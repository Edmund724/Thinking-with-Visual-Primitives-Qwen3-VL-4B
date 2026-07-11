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
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from tqdm import tqdm

from ..data.datasets.image_loader import load_image
from ..models.qwen_vl_loader import (
    _get_use_cache_states,
    _set_use_cache_deep,
    _set_use_cache_states,
    load_qlora_model,
)
from ..utils.constants import (
    DEFAULT_DISTILL_TEMPERATURE,
    SPECIAL_TOKENS,
)
from ..utils.conversation_builder import ConversationBuilder
from .memory_utils import clear_memory, get_gpu_memory_gb, get_gpu_memory_reserved_gb


def _freeze_opd_embeddings(model: torch.nn.Module, logger: logging.Logger) -> int:
    """Freeze embed_tokens and lm_head so OPD only updates LoRA adapters.

    IMPORTANT: This is safe ONLY because the Stage 5 student checkpoint already
    learned the visual-primitive embeddings in earlier stages.  In Stage 1-3,
    ``embed_tokens`` / ``lm_head`` are kept trainable via ``modules_to_save``;
    without that training, the added special tokens stay at their random
    initialization and the model emits garbage / non-Latin characters.  OPD is
    pure reverse-KL distillation of the expert policies, so it does not need to
    relearn those embeddings.

    Keeping ``embed_tokens`` / ``lm_head`` trainable in Stage 6 adds ~390M
    trainable parameters and ~3GB of AdamW state, which pushes the two-model OPD
    setup (student + one expert) past 24GB.  Freezing them is consistent with the
    paper's reverse-KL distillation objective and leaves only LoRA parameters to
    be optimized.

    A safety check verifies that the special-token embeddings look trained (L2
    norm well above random-init magnitude) before freezing.  If they appear
    uninitialized, a warning is emitted and the caller should abort and re-run
    the preceding stages.
    """
    n = 0
    for name, p in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            p.requires_grad = False
            n += p.numel()

    # Guard: ensure the special-token embeddings were actually trained upstream.
    # Randomly-initialized embeddings typically have an L2 norm ~ sqrt(dim) * sigma,
    # where sigma ~ 0.02 for Qwen3-VL-4B (embed_dim=2048 -> norm ~ 0.9).  After even
    # a small amount of training the norms cluster around 0.3-0.5.  We use a
    # conservative threshold of 0.05 to catch pathological zero/untrained weights
    # while tolerating very weak training.
    _warn_if_embeddings_untrained(model, logger, min_norm=0.05)
    return n


def _warn_if_embeddings_untrained(
    model: torch.nn.Module,
    logger: logging.Logger,
    min_norm: float = 0.05,
) -> None:
    """Log a warning if visual-primitive embeddings look randomly initialized."""
    base = model.base_model.model if hasattr(model, "base_model") else model
    tokenizer = getattr(model, "_tvp_processor_tokenizer", None)
    if tokenizer is None:
        # Try to locate the tokenizer through the processor attribute if available.
        processor = getattr(model, "_tvp_processor", None)
        if processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)

    # Find embed_tokens / lm_head modules.
    embed = None
    lm_head = None
    for name, module in base.named_modules():
        if "embed_tokens" in name and hasattr(module, "weight"):
            embed = module
        if "lm_head" in name and hasattr(module, "weight"):
            lm_head = module
        if embed is not None and lm_head is not None and name.count(".") == 0:
            # Only need the first (actual) ones; lm_head may also be under
            # language_model, but the named_modules traversal will pick it up.
            pass

    if embed is None:
        logger.warning("Could not locate embed_tokens; skipping special-token embedding check.")
        return

    # If we don't have a tokenizer, just check the last len(SPECIAL_TOKENS) rows.
    ids_to_check: list[int] = []
    if tokenizer is not None:
        for tok in SPECIAL_TOKENS:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tok_id, int):
                ids_to_check.append(tok_id)
    else:
        # Fallback: assume special tokens are the last rows added to the vocab.
        ids_to_check = list(range(embed.num_embeddings - len(SPECIAL_TOKENS), embed.num_embeddings))
        ids_to_check = [i for i in ids_to_check if 0 <= i < embed.num_embeddings]

    if not ids_to_check:
        return

    norms = []
    with torch.no_grad():
        rows = embed.weight[ids_to_check]
        norms = rows.norm(dim=-1).tolist()

    low_norm_tokens = []
    for tok_id, norm in zip(ids_to_check, norms):
        if norm < min_norm:
            token_str = tokenizer.convert_ids_to_tokens([tok_id])[0] if tokenizer is not None else str(tok_id)
            low_norm_tokens.append((token_str, norm))

    if low_norm_tokens:
        details = ", ".join(f"{tok} (norm={norm:.4f})" for tok, norm in low_norm_tokens)
        logger.warning(
            "Special-token embeddings appear untrained (L2 norm below %.4f): %s. "
            "This usually means embed_tokens/lm_head were frozen in earlier stages. "
            "OPD will freeze them, which is unsafe with untrained embeddings and can "
            "cause garbage / non-Latin output. Please re-train Stage 1-3/5 before OPD.",
            min_norm, details,
        )
    else:
        logger.info(
            "Special-token embedding check passed (min norm=%.4f). Safe to freeze embed_tokens/lm_head for OPD.",
            min(norms) if norms else 0.0,
        )


def _cast_frozen_norms_to_bf16(model: torch.nn.Module) -> int:
    """Cast frozen text RMSNorm modules back to bfloat16.

    ``prepare_model_for_kbit_training`` casts normalization layers to fp32 for
    training stability.  In OPD those text layers are frozen, and their fp32
    outputs trigger the flash-attn "Casting fp32 inputs back to bfloat16"
    warning.  Restoring text RMSNorms to the model's native dtype removes the
    warning without affecting gradients (only LoRA parameters are trainable).

    Vision LayerNorms are left in fp32 because the vision backbone expects
    fp32 inputs for its normalization blocks.
    """
    n = 0
    for module in model.modules():
        if "TextRMSNorm" in module.__class__.__name__:
            if not any(p.requires_grad for p in module.parameters(recurse=False)):
                module.to(torch.bfloat16)
                n += 1
    return n


@contextmanager
def _no_gradient_checkpointing(model: torch.nn.Module):
    """Temporarily disable gradient checkpointing and enable KV-cache for generation.

    Qwen3-VL's generation is memory-hungry with long multimodal sequences. When
    gradient checkpointing is active, ``transformers`` forces ``use_cache=False``
    (and prints a warning), causing every autoregressive step to recompute
    attention over the full prefix. This context manager disables gradient
    checkpointing and temporarily sets ``use_cache=True`` on all nested config
    objects during ``model.generate()``, then restores both states so the
    subsequent training forward still benefits from checkpointing.
    """
    was_enabled = getattr(model, "is_gradient_checkpointing", False)
    old_use_cache = _get_use_cache_states(model)
    if was_enabled:
        model.gradient_checkpointing_disable()
    _set_use_cache_deep(model, True)
    try:
        yield
    finally:
        if was_enabled:
            model.gradient_checkpointing_enable()
        _set_use_cache_states(model, old_use_cache)


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
    load_adapter: bool = True,
) -> tuple:
    """Load OPD training checkpoint. Returns (global_step, epoch, step_in_epoch)."""
    logger.info(f"Resuming OPD from checkpoint: {checkpoint_dir}")

    # Load adapter weights
    if load_adapter:
        from peft import PeftModel
        if isinstance(student_model, PeftModel):
            # Replace the current "default" adapter weights with the checkpoint
            # weights in-place. Deleting and re-adding the adapter can leave PEFT
            # auxiliary modules with no active adapter and raise
            # "Please specify at least one adapter to set".
            from peft.utils import load_peft_weights, set_peft_model_state_dict
            adapter_weights = load_peft_weights(checkpoint_dir)
            set_peft_model_state_dict(student_model, adapter_weights, adapter_name="default")
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
    norm_cast_expert = _cast_frozen_norms_to_bf16(expert)
    if norm_cast_expert:
        logger.info(f"Cast {norm_cast_expert} frozen norm layer(s) on expert back to bfloat16")

    # Student should be in train mode
    student_model.train()

    # Freeze special-token embeddings; they were already learned in earlier stages.
    frozen = _freeze_opd_embeddings(student_model, logger)
    if frozen:
        logger.info(f"Frozen embed_tokens/lm_head ({frozen:,} params) for OPD")
    norm_cast = _cast_frozen_norms_to_bf16(student_model)
    if norm_cast:
        logger.info(f"Cast {norm_cast} frozen norm layer(s) back to bfloat16")

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

    # Optimizer: only student LoRA params; use 8-bit AdamW to fit in 24GB.
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable student params: {n_params:,} ({n_params/1e6:.1f}M)")
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=learning_rate, weight_decay=0.0)
        logger.info("Using 8-bit AdamW for OPD")
    except Exception as exc:  # pragma: no cover - bnb may be unavailable
        logger.warning(f"8-bit AdamW unavailable ({exc}); falling back to fp32 AdamW")
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
    box_expert_path: str,
    point_expert_path: str,
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
    Experts are loaded on demand and released immediately after use so that only
    one teacher (plus the student) resides in GPU memory at any time.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    student_model.train()

    # Freeze special-token embeddings; they were already learned in earlier stages.
    frozen = _freeze_opd_embeddings(student_model, logger)
    if frozen:
        logger.info(f"Frozen embed_tokens/lm_head ({frozen:,} params) for OPD")
    norm_cast = _cast_frozen_norms_to_bf16(student_model)
    if norm_cast:
        logger.info(f"Cast {norm_cast} frozen norm layer(s) back to bfloat16")

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

    # Optimizer: only student LoRA params; use 8-bit AdamW to fit in 24GB.
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable student params: {n_params:,} ({n_params/1e6:.1f}M)")
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable_params, lr=learning_rate, weight_decay=0.0)
        logger.info("Using 8-bit AdamW for OPD")
    except Exception as exc:  # pragma: no cover - bnb may be unavailable
        logger.warning(f"8-bit AdamW unavailable ({exc}); falling back to fp32 AdamW")
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

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
        batch_count = 0

        # Phase 1: Box expert on box data (accumulate gradients, no step)
        logger.info(f"Epoch {epoch+1}: Loading Box Expert from {box_expert_path}...")
        box_expert, _ = load_qlora_model(model_name=box_expert_path)
        for param in box_expert.parameters():
            param.requires_grad = False
        box_expert.eval()
        norm_cast_box = _cast_frozen_norms_to_bf16(box_expert)
        if norm_cast_box:
            logger.info(f"Cast {norm_cast_box} frozen norm layer(s) on box expert back to bfloat16")
        # Experts are frozen teachers; gradient checkpointing is unnecessary and
        # produces checkpoint warnings when called under no_grad.
        if getattr(box_expert, "is_gradient_checkpointing", False):
            box_expert.gradient_checkpointing_disable()
        box_expert.to(student_model.device)
        allocated = get_gpu_memory_gb()
        reserved = get_gpu_memory_reserved_gb()
        logger.info(
            f"Epoch {epoch+1} Box expert loaded: "
            f"GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
        )
        logger.info(f"Epoch {epoch+1}: Box expert phase...")
        pbar = tqdm(box_loader, desc=f"OPD Box E{epoch+1}", unit="batch")
        for step, batch in enumerate(pbar):
            # Skip already-processed steps when resuming within an epoch
            if epoch == start_epoch and step < start_step_in_epoch:
                continue
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
            pbar.set_postfix({"kl": f"{kl_val:.4f}", "phase": "box"})

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

        # Release box expert before loading point expert
        del box_expert
        clear_memory()
        allocated = get_gpu_memory_gb()
        reserved = get_gpu_memory_reserved_gb()
        logger.info(
            f"Epoch {epoch+1} Box expert released: "
            f"GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
        )

        # Phase 2: Point expert on point data (accumulate, then step)
        logger.info(f"Epoch {epoch+1}: Loading Point Expert from {point_expert_path}...")
        point_expert, _ = load_qlora_model(model_name=point_expert_path)
        for param in point_expert.parameters():
            param.requires_grad = False
        point_expert.eval()
        norm_cast_point = _cast_frozen_norms_to_bf16(point_expert)
        if norm_cast_point:
            logger.info(f"Cast {norm_cast_point} frozen norm layer(s) on point expert back to bfloat16")
        if getattr(point_expert, "is_gradient_checkpointing", False):
            point_expert.gradient_checkpointing_disable()
        point_expert.to(student_model.device)
        allocated = get_gpu_memory_gb()
        reserved = get_gpu_memory_reserved_gb()
        logger.info(
            f"Epoch {epoch+1} Point expert loaded: "
            f"GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
        )
        logger.info(f"Epoch {epoch+1}: Point expert phase...")
        pbar = tqdm(point_loader, desc=f"OPD Point E{epoch+1}", unit="batch")
        for step, batch in enumerate(pbar):
            # Skip already-processed steps when resuming within an epoch.
            # Note: start_step_in_epoch counts across the full epoch (box + point).
            total_steps_before_point = len(box_loader)
            if epoch == start_epoch and (step + total_steps_before_point) < start_step_in_epoch:
                continue
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
            elif is_last_batch:
                # Step the scheduler exactly once per actual optimizer update.
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
                    global_step, epoch, step + 1 + total_steps_before_point,
                    output_dir, logger,
                )

        # Release point expert before next epoch
        del point_expert
        clear_memory()
        allocated = get_gpu_memory_gb()
        reserved = get_gpu_memory_reserved_gb()
        logger.info(
            f"Epoch {epoch+1} Point expert released: "
            f"GPU allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
        )

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

    Returns the KL loss value. When ``do_step=True`` the optimizer is stepped
    before returning.
    """
    # Convert batch tensors to the student device. For image_grid_thw we keep
    # the collated [num_images, 3] shape; using v[0] would collapse it to [3]
    # and break Qwen3-VL's vision bilinear sampling (it expects rows of [t,h,w]).
    prompt_inputs = {}
    for k, v in batch.items():
        if k in ("task_type", "prompt_text"):
            continue
        if k == "image_grid_thw":
            prompt_inputs[k] = v.to(student_model.device)
        else:
            prompt_inputs[k] = v[0].to(student_model.device)

    # Add batch dimension to sequence tensors so generate()/forward() receive
    # 2D inputs as expected by Qwen3-VL (input_ids and attention_mask must both
    # be [1, seq_len] to avoid IndexError in M-RoPE position-ids computation).
    prompt_inputs["input_ids"] = prompt_inputs["input_ids"].unsqueeze(0)
    if "attention_mask" in prompt_inputs:
        prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"].unsqueeze(0)
    if "mm_token_type_ids" in prompt_inputs:
        prompt_inputs["mm_token_type_ids"] = prompt_inputs["mm_token_type_ids"].unsqueeze(0)

    prompt_ids = prompt_inputs["input_ids"]
    # Qwen3-VL needs mm_token_type_ids alongside image_grid_thw to compute M-RoPE.
    image_kwargs = {
        k: v for k, v in prompt_inputs.items()
        if k in ("pixel_values", "image_grid_thw", "mm_token_type_ids")
    }

    # 1. Student generates (on-policy)
    # Generation is under no_grad and does not benefit from gradient
    # checkpointing; temporarily disable it to avoid use_cache/checkpoint warnings.
    with torch.no_grad(), _no_gradient_checkpointing(student_model):
        generated = student_model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.1),
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **image_kwargs,
        )
    full_ids = generated[0]

    # 2. Forward student
    # The full sequence (prompt + generated response) is longer than the prompt,
    # but image kwargs from the processor describe the prompt only. Extend
    # mm_token_type_ids with zeros for generated tokens so M-RoPE can compute
    # position_ids for the full length.
    full_len = full_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    if "mm_token_type_ids" in image_kwargs and full_len > prompt_len:
        mm = image_kwargs["mm_token_type_ids"]
        generated_only_len = full_len - prompt_len
        mm_padding = torch.zeros(
            1, generated_only_len, dtype=mm.dtype, device=mm.device,
        )
        image_kwargs = {
            **image_kwargs,
            "mm_token_type_ids": torch.cat([mm, mm_padding], dim=1),
        }

    # 2. Forward student on full sequence (no internal loss; we compute KL below)
    student_outputs = student_model(
        input_ids=full_ids.unsqueeze(0),
        use_cache=False,
        **image_kwargs,
    )
    student_logits = student_outputs.logits[:, :-1, :]

    # 3. Forward expert (frozen)
    with torch.no_grad():
        expert_outputs = expert(
            input_ids=full_ids.unsqueeze(0),
            use_cache=False,
            **image_kwargs,
        )
        expert_logits = expert_outputs.logits[:, :-1, :]

    # Restore image_kwargs to prompt-length for the next batch
    if "mm_token_type_ids" in image_kwargs and full_len > prompt_len:
        image_kwargs = {
            **image_kwargs,
            "mm_token_type_ids": mm,
        }
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

    # Variable-length sequences can leave large reserved blocks behind;
    # release them after each batch to keep fragmentation in check.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return kl_loss.item()
