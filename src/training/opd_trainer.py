"""OPD (Offline Preference Distillation) Trainer.

On-policy reverse KL distillation from expert models to a student (Unified) model.

Algorithm per step:
  1. Student generates response (on-policy)
  2. Full sequence = prompt + student_response
  3. Forward student on full sequence → student_logits
  4. Forward expert on SAME full sequence → expert_logits
  5. Reverse KL: D_KL(student || expert) with temperature
  6. Backward → only student LoRA updates

Key design (per paper):
  - On-policy: input sequence MUST be student-generated, not expert-generated
  - Reverse KL: D_KL(S || E) — student learns expert's high-probability regions
  - Temperature: temp=1.0~1.5 to soften distributions
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from tqdm import tqdm


class OPDDataset(Dataset):
    """Dataset for OPD training.

    Each sample:
        prompt_text: str — formatted prompt (with chat template, add_generation_prompt=True)
        task_type: str — "box" or "point"/"maze"
    """

    def __init__(self, data: List[Dict[str, Any]], processor: AutoProcessor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        task_type = sample.get("task_type", "box")

        # Build prompt messages
        messages = [
            {
                "role": "system",
                "content": "You are a helpful visual reasoning assistant. Think step by step.",
            },
            {
                "role": "user",
                "content": sample["prompt"],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Tokenize prompt
        prompt_inputs = self.processor(
            text=[prompt_text],
            return_tensors="pt",
            padding=False,
        )
        prompt_ids = prompt_inputs["input_ids"][0]

        return {
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "task_type": task_type,
            "sample": sample,  # Keep original sample for image loading if needed
        }


def _save_opd_checkpoint(
    student_model,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    output_dir: str,
    logger: logging.Logger,
):
    """Save OPD training checkpoint."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save adapter weights
    student_model.save_pretrained(ckpt_dir)

    # Save optimizer + scheduler + training state
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
    }
    torch.save(state, os.path.join(ckpt_dir, "opd_state.pt"))
    logger.info(f"  Saved OPD checkpoint at step {global_step} -> {ckpt_dir}")

    # Prune old checkpoints (keep latest 2)
    import glob
    import shutil
    ckpt_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                       key=lambda d: int(d.split("-")[-1]))
    while len(ckpt_dirs) > 2:
        old_dir = ckpt_dirs.pop(0)
        shutil.rmtree(old_dir, ignore_errors=True)
        logger.info(f"  Removed old checkpoint: {old_dir}")


def _load_opd_checkpoint(
    student_model,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    logger: logging.Logger,
) -> tuple:
    """Load OPD training checkpoint. Returns (global_step, epoch, step_in_epoch)."""
    logger.info(f"Resuming OPD from checkpoint: {checkpoint_dir}")

    # Load adapter weights
    from peft import PeftModel
    if isinstance(student_model, PeftModel):
        student_model.load_adapter(checkpoint_dir, adapter_name="default", is_trainable=True)
        student_model.set_adapter("default")
    else:
        state_dict = torch.load(os.path.join(checkpoint_dir, "adapter_model.bin"),
                                map_location="cpu", weights_only=False)
        student_model.load_state_dict(state_dict, strict=False)

    # Load optimizer + scheduler + training state
    state_path = os.path.join(checkpoint_dir, "opd_state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    global_step = state["global_step"]
    epoch = state["epoch"]
    step_in_epoch = state["step_in_epoch"]
    logger.info(
        f"  Resumed at global_step={global_step}, epoch={epoch+1}, "
        f"step_in_epoch={step_in_epoch}"
    )
    return global_step, epoch, step_in_epoch


def train_opd(
    student_model,
    box_expert,
    point_expert,
    processor: AutoProcessor,
    train_data: List[Dict[str, Any]],
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    save_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    logger: logging.Logger | None = None,
):
    """Run OPD training with reverse KL distillation.

    Student generates on-policy, experts score the same sequence.
    Task routing: box → Box Expert, point/maze → Point Expert.

    Experts are loaded one at a time to save VRAM.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Freeze experts
    for param in box_expert.parameters():
        param.requires_grad = False
    box_expert.eval()

    for param in point_expert.parameters():
        param.requires_grad = False
    point_expert.eval()

    # Student should be in train mode
    student_model.train()

    # Build dataset
    dataset = OPDDataset(data=train_data, processor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        drop_last=False,
    )
    logger.info(f"OPD dataset: {len(dataset)} samples, {len(dataloader)} batches")

    # Optimizer: only student LoRA params
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable student params: {n_params:,} ({n_params/1e6:.1f}M)")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * num_epochs
    )

    # Get pad/eos token ids
    pad_token_id = processor.tokenizer.pad_token_id or 0
    eos_token_id = processor.tokenizer.eos_token_id

    global_step = 0
    start_epoch = 0
    start_step_in_epoch = 0

    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.isdir(resume_from_checkpoint):
        global_step, start_epoch, start_step_in_epoch = _load_opd_checkpoint(
            student_model, optimizer, scheduler, resume_from_checkpoint, logger,
        )
    elif resume_from_checkpoint:
        logger.warning(f"Checkpoint not found: {resume_from_checkpoint}, starting from scratch")

    for epoch in range(start_epoch, num_epochs):
        epoch_kl = 0.0
        epoch_t0 = time.time()

        pbar = tqdm(dataloader, desc=f"OPD Epoch {epoch+1}/{num_epochs}", unit="batch")
        for step, batch in enumerate(pbar):
            # Skip already-processed steps when resuming within an epoch
            if epoch == start_epoch and step < start_step_in_epoch:
                continue

            global_step += 1

            # Warmup
            if global_step <= warmup_steps:
                lr = learning_rate * global_step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr

            task_type = batch["task_type"][0]  # batch_size=1
            prompt_ids = batch["prompt_ids"][0].to(student_model.device)

            # 1. Student generates response (on-policy)
            with torch.no_grad():
                generated = student_model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
            full_ids = generated[0]  # prompt + student response

            # 2. Select expert by task type
            if task_type == "box":
                expert = box_expert
            else:
                expert = point_expert

            # 3. Forward student on full sequence
            student_outputs = student_model(
                input_ids=full_ids.unsqueeze(0),
                labels=full_ids.unsqueeze(0),
            )
            # Get logits excluding the last position (no prediction after final token)
            student_logits = student_outputs.logits[:, :-1, :]  # [1, seq-1, vocab]

            # 4. Forward expert on SAME full sequence (frozen, no grad)
            with torch.no_grad():
                expert_outputs = expert(
                    input_ids=full_ids.unsqueeze(0),
                )
                expert_logits = expert_outputs.logits[:, :-1, :]  # [1, seq-1, vocab]

            # Align lengths
            min_len = min(student_logits.shape[1], expert_logits.shape[1])
            student_logits = student_logits[:, :min_len, :]
            expert_logits = expert_logits[:, :min_len, :]

            # 5. Compute reverse KL: D_KL(student || expert)
            # p_s = softmax(student_logits / temp)
            # kl = sum(p_s * (log(p_s) - log(p_e)))
            temp = max(temperature, 0.1)
            log_p_s = F.log_softmax(student_logits / temp, dim=-1)
            log_p_e = F.log_softmax(expert_logits / temp, dim=-1)
            p_s = F.softmax(student_logits / temp, dim=-1)

            kl_per_token = (p_s * (log_p_s - log_p_e)).sum(dim=-1)  # [1, min_len]
            kl_loss = kl_per_token.mean()

            # 6. Backward
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
            optimizer.step()
            optimizer.zero_grad()

            if global_step > warmup_steps:
                scheduler.step()

            kl_val = kl_loss.item()
            epoch_kl += kl_val

            pbar.set_postfix({
                "kl": f"{kl_val:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "task": task_type,
            })

            if global_step % logging_steps == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | Step {global_step} | "
                    f"KL: {kl_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # Save checkpoint
            if global_step % save_steps == 0:
                _save_opd_checkpoint(
                    student_model, optimizer, scheduler,
                    global_step, epoch, step + 1,
                    output_dir, logger,
                )

        avg_kl = epoch_kl / max(len(dataloader), 1)
        logger.info(
            f"OPD Epoch {epoch+1}/{num_epochs} complete. "
            f"Avg KL: {avg_kl:.4f} | Time: {time.time() - epoch_t0:.1f}s"
        )

    logger.info("OPD training complete.")
