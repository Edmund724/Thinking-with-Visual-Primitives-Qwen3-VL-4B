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
  - Temperature: DEFAULT_DISTILL_TEMPERATURE (paper range 1.0~1.5) to soften distributions
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

from ..data.datasets.image_loader import load_image
from ..utils.constants import DEFAULT_DISTILL_TEMPERATURE
from ..utils.conversation_builder import ConversationBuilder


class OPDDataset(Dataset):
    """Dataset for OPD training.

    Each sample:
        prompt_text: str — formatted prompt (with chat template, add_generation_prompt=True)
        task_type: str — "box" or "point"/"maze"
    """

    def __init__(self, data: List[Dict[str, Any]], processor: AutoProcessor):
        self.data = data
        self.processor = processor
        self._conv_builder = ConversationBuilder("opd")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        task_type = sample.get("task_type", "box")
        image = load_image(sample.get("image"))

        # Build prompt messages with image embedded in multimodal content blocks
        messages = self._conv_builder.build_prompt(sample["prompt"], image)

        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Process prompt with image so pixel_values / image_grid_thw are available
        if image is not None:
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
        else:
            prompt_inputs = self.processor(
                text=[prompt_text],
                return_tensors="pt",
                padding=False,
            )

        # Remove batch dimension
        prompt_inputs = {k: v.squeeze(0) for k, v in prompt_inputs.items()}

        return {
            "prompt_text": prompt_text,
            "task_type": task_type,
            **prompt_inputs,
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


def _opd_collate(features: list) -> Dict[str, Any]:
    """Collate OPD batch; pixel_values are concatenated, not stacked.

    Qwen3-VL expects pixel_values concatenated along the patch dimension,
    with image_grid_thw indicating each image's grid size.
    """
    batch: Dict[str, Any] = {}
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())

    for key in all_keys:
        if key in ("prompt_text", "task_type"):
            batch[key] = [f[key] for f in features]
        elif key == "pixel_values":
            present = [f[key] for f in features if key in f]
            if present:
                batch[key] = torch.cat(present, dim=0)
        elif key == "image_grid_thw":
            present = [f[key] for f in features if key in f]
            if present:
                batch[key] = torch.stack(present, dim=0)
        else:
            # Stack tensors present in all samples
            if all(key in f for f in features):
                batch[key] = torch.stack([f[key] for f in features], dim=0)
    return batch


def train_opd(
    student_model,
    expert,
    processor: AutoProcessor,
    train_data: List[Dict[str, Any]],
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = DEFAULT_DISTILL_TEMPERATURE,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    save_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    logger: logging.Logger | None = None,
):
    """Run OPD training with reverse KL distillation for one expert.

    Student generates on-policy, the provided expert scores the same sequence.
    Call this function once per expert (box expert on box data, point expert on
    point/maze data) to keep only one teacher in GPU memory at a time, matching
    the paper's two-teacher full-vocabulary distillation (Sec 2.5.4).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Freeze expert
    for param in expert.parameters():
        param.requires_grad = False
    expert.eval()

    # Student should be in train mode
    student_model.train()

    # Build dataset
    dataset = OPDDataset(data=train_data, processor=processor)
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_opd_collate,
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

            # Build prompt inputs and move to device
            prompt_inputs = {
                k: v[0].to(student_model.device)
                for k, v in batch.items()
                if k not in ("task_type", "prompt_text")
            }
            prompt_ids = prompt_inputs["input_ids"]
            image_kwargs = {
                k: v for k, v in prompt_inputs.items()
                if k in ("pixel_values", "image_grid_thw")
            }

            # 1. Student generates response (on-policy)
            with torch.no_grad():
                generated = student_model.generate(
                    input_ids=prompt_ids.unsqueeze(0),
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.1),
                    do_sample=True,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    **image_kwargs,
                )
            full_ids = generated[0]  # prompt + student response

            # 2. Forward student on full sequence
            student_outputs = student_model(
                input_ids=full_ids.unsqueeze(0),
                labels=full_ids.unsqueeze(0),
                **image_kwargs,
            )
            # Get logits excluding the last position (no prediction after final token)
            student_logits = student_outputs.logits[:, :-1, :]  # [1, seq-1, vocab]

            # 3. Forward expert on SAME full sequence (frozen, no grad)
            with torch.no_grad():
                expert_outputs = expert(
                    input_ids=full_ids.unsqueeze(0),
                    **image_kwargs,
                )
                expert_logits = expert_outputs.logits[:, :-1, :]  # [1, seq-1, vocab]

            # Align lengths
            min_len = min(student_logits.shape[1], expert_logits.shape[1])
            student_logits = student_logits[:, :min_len, :]
            expert_logits = expert_logits[:, :min_len, :]

            # 4. Compute reverse KL: D_KL(student || expert)
            # p_s = softmax(student_logits / temp)
            # kl = sum(p_s * (log(p_s) - log(p_e)))
            temp = max(temperature, 0.1)
            log_p_s = F.log_softmax(student_logits / temp, dim=-1)
            log_p_e = F.log_softmax(expert_logits / temp, dim=-1)
            p_s = F.softmax(student_logits / temp, dim=-1)

            kl_per_token = (p_s * (log_p_s - log_p_e)).sum(dim=-1)  # [1, min_len]
            kl_loss = kl_per_token.mean()

            # 5. Backward
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

        # Clear cache between epochs to combat fragmentation from variable-length
        # student completions (same pattern as stage 4 GRPO rounds).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("OPD training complete.")


def train_opd_parallel(
    student_model,
    box_expert,
    point_expert,
    processor: AutoProcessor,
    box_data: List[Dict[str, Any]],
    point_data: List[Dict[str, Any]],
    output_dir: str,
    num_epochs: int = 2,
    learning_rate: float = 1e-6,
    per_device_batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = DEFAULT_DISTILL_TEMPERATURE,
    warmup_steps: int = 100,
    logging_steps: int = 20,
    save_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    logger: logging.Logger | None = None,
):
    """Run OPD with gradient accumulation simulating parallel distillation.

    The paper (Sec 2.5.4) defines: L = w1*D_KL(π_θ||π_E_box) + w2*D_KL(π_θ||π_E_point).
    This function approximates that by:
      1. Process box batches with box expert → backward (accumulate, no step)
      2. Swap to point expert → process point batches → backward (accumulate)
      3. optimizer.step() — gradient is the sum of both expert signals
    Only one expert is in GPU memory at a time, keeping VRAM constant.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Freeze both experts
    for expert in (box_expert, point_expert):
        for param in expert.parameters():
            param.requires_grad = False
        expert.eval()

    student_model.train()

    # Build datasets
    box_dataset = OPDDataset(data=box_data, processor=processor)
    point_dataset = OPDDataset(data=point_data, processor=processor)
    box_loader = DataLoader(box_dataset, batch_size=per_device_batch_size, shuffle=True,
                            drop_last=False, collate_fn=_opd_collate)
    point_loader = DataLoader(point_dataset, batch_size=per_device_batch_size, shuffle=True,
                              drop_last=False, collate_fn=_opd_collate)

    total_steps_per_epoch = len(box_loader) + len(point_loader)
    logger.info(
        f"OPD parallel: {len(box_dataset)} box + {len(point_dataset)} point samples, "
        f"{total_steps_per_epoch} batches/epoch"
    )

    # Optimizer: only student LoRA params
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable student params: {n_params:,} ({n_params/1e6:.1f}M)")
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps_per_epoch * num_epochs
    )

    pad_token_id = processor.tokenizer.pad_token_id or 0
    eos_token_id = processor.tokenizer.eos_token_id

    global_step = 0
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_kl = 0.0
        epoch_t0 = time.time()
        batch_count = 0

        # Phase 1: Box expert on box data (accumulate gradients, no step)
        logger.info(f"Epoch {epoch+1}: Box expert phase...")
        box_expert.to(student_model.device)
        pbar = tqdm(box_loader, desc=f"OPD Box E{epoch+1}", unit="batch")
        for step, batch in enumerate(pbar):
            kl_val = _opd_single_batch(
                student_model, box_expert, batch, processor,
                max_new_tokens, temperature, pad_token_id, eos_token_id,
                do_step=False,  # accumulate only
            )
            global_step += 1
            batch_count += 1
            epoch_kl += kl_val
            if global_step <= warmup_steps:
                lr = learning_rate * global_step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
            if global_step > warmup_steps:
                scheduler.step()
            pbar.set_postfix({"kl": f"{kl_val:.4f}", "phase": "box"})

        # Release box expert, load point expert
        box_expert.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        point_expert.to(student_model.device)

        # Phase 2: Point expert on point data (accumulate, then step)
        logger.info(f"Epoch {epoch+1}: Point expert phase...")
        pbar = tqdm(point_loader, desc=f"OPD Point E{epoch+1}", unit="batch")
        for step, batch in enumerate(pbar):
            is_last_batch = (step == len(point_loader) - 1)
            kl_val = _opd_single_batch(
                student_model, point_expert, batch, processor,
                max_new_tokens, temperature, pad_token_id, eos_token_id,
                do_step=is_last_batch,  # step on last batch of epoch
                optimizer=optimizer if is_last_batch else None,
                trainable_params=trainable_params,
            )
            global_step += 1
            batch_count += 1
            epoch_kl += kl_val
            if global_step <= warmup_steps:
                lr = learning_rate * global_step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr
            if global_step > warmup_steps:
                scheduler.step()
            pbar.set_postfix({"kl": f"{kl_val:.4f}", "phase": "point"})

            if global_step % logging_steps == 0:
                logger.info(
                    f"  Epoch {epoch+1} | Step {global_step} | "
                    f"KL: {kl_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            if global_step % save_steps == 0:
                _save_opd_checkpoint(
                    student_model, optimizer, scheduler,
                    global_step, epoch, step + 1,
                    output_dir, logger,
                )

        # Release point expert for next epoch
        point_expert.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_kl = epoch_kl / max(batch_count, 1)
        logger.info(
            f"OPD Epoch {epoch+1}/{num_epochs} complete. "
            f"Avg KL: {avg_kl:.4f} | Time: {time.time() - epoch_t0:.1f}s"
        )

    logger.info("OPD parallel training complete.")


def _opd_single_batch(
    student_model,
    expert,
    batch,
    processor,
    max_new_tokens,
    temperature,
    pad_token_id,
    eos_token_id,
    do_step: bool = False,
    optimizer=None,
    trainable_params=None,
) -> float:
    """Process a single OPD batch: generate, forward, compute KL, backward.

    Returns the KL loss value.
    """
    prompt_inputs = {
        k: v[0].to(student_model.device)
        for k, v in batch.items()
        if k not in ("task_type", "prompt_text")
    }
    prompt_ids = prompt_inputs["input_ids"]
    image_kwargs = {
        k: v for k, v in prompt_inputs.items()
        if k in ("pixel_values", "image_grid_thw")
    }

    # 1. Student generates (on-policy)
    with torch.no_grad():
        generated = student_model.generate(
            input_ids=prompt_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.1),
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **image_kwargs,
        )
    full_ids = generated[0]

    # 2. Forward student
    student_outputs = student_model(
        input_ids=full_ids.unsqueeze(0),
        labels=full_ids.unsqueeze(0),
        **image_kwargs,
    )
    student_logits = student_outputs.logits[:, :-1, :]

    # 3. Forward expert (frozen)
    with torch.no_grad():
        expert_outputs = expert(
            input_ids=full_ids.unsqueeze(0),
            **image_kwargs,
        )
        expert_logits = expert_outputs.logits[:, :-1, :]

    # Align lengths
    min_len = min(student_logits.shape[1], expert_logits.shape[1])
    student_logits = student_logits[:, :min_len, :]
    expert_logits = expert_logits[:, :min_len, :]

    # 4. Reverse KL
    temp = max(temperature, 0.1)
    log_p_s = F.log_softmax(student_logits / temp, dim=-1)
    log_p_e = F.log_softmax(expert_logits / temp, dim=-1)
    p_s = F.softmax(student_logits / temp, dim=-1)
    kl_per_token = (p_s * (log_p_s - log_p_e)).sum(dim=-1)
    kl_loss = kl_per_token.mean()

    # 5. Backward (accumulate if not stepping)
    kl_loss.backward()

    if do_step and optimizer is not None and trainable_params is not None:
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
        optimizer.step()
        optimizer.zero_grad()

    return kl_loss.item()
