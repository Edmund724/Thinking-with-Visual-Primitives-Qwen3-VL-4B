"""SFT Trainer wrapper for QLoRA training of visual primitive models."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from ...data.datasets.sft_dataset import SFTDataset
from ...utils.constants import MAX_GPU_MEMORY_GB

logger = logging.getLogger(__name__)


class WeightedSFTTrainer(Trainer):
    """SFT Trainer that applies per-token loss weights.

    Expects each sample to contain a ``loss_weight`` tensor aligned with
    ``labels``. Prompt/padding positions should have weight 0 so they do not
    contribute to the mean.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_weight = inputs.pop("loss_weight", None)

        outputs = model(**inputs)
        logits = outputs.logits

        # Shift so that logits predict the next token.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weight[..., 1:].contiguous() if loss_weight is not None else None

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        if shift_weights is not None:
            losses = losses * shift_weights.view(-1)

        # Mean over non-masked assistant tokens, weighted by per-token loss_weight.
        # Use the sum of weights as denominator so that up-weighting format tokens
        # changes the gradient emphasis without inflating the reported loss value.
        mask = (shift_labels != -100).view(-1)
        if mask.any():
            weights = shift_weights.view(-1)[mask] if shift_weights is not None else torch.ones_like(losses[mask])
            loss = losses[mask].sum() / weights.sum()
        else:
            loss = losses.sum() * 0.0

        return (loss, outputs) if return_outputs else loss


def create_sft_trainer(
    model: PeftModel,
    processor: AutoProcessor,
    train_data: list,
    output_dir: str,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 500,
    warmup_steps: int = 100,
    use_wandb: bool = True,
    additional_callbacks: Optional[list] = None,
    format_token_weight: float = 5.0,
) -> Trainer:
    """Create a Trainer for SFT training."""

    train_dataset = SFTDataset(
        data=train_data,
        processor=processor,
        max_length=max_seq_length,
        format_token_weight=format_token_weight,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="no",
        report_to="wandb" if use_wandb else "none",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    )

    # Collect callbacks
    callbacks = list(additional_callbacks or [])
    from ..callbacks import MemoryMonitorCallback
    callbacks.append(MemoryMonitorCallback(max_memory_gb=MAX_GPU_MEMORY_GB))

    # Get pad_token_id from processor (C3: avoid hardcoding)
    pad_token_id = processor.tokenizer.pad_token_id or 0

    # Ensure use_cache is disabled on ALL nested configs — incompatible with
    # gradient checkpointing. The Qwen3VLTextModel deep inside reads its own
    # config.use_cache during forward, which a top-level set won't reach.
    from ...models.qwen_vl_loader import _set_use_cache_deep
    _set_use_cache_deep(model)

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=lambda features: _collate_sft(features, pad_token_id=pad_token_id),
        callbacks=callbacks,
    )

    return trainer


def _collate_sft(features: list, pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate SFT batch with dynamic padding (pad to longest in batch).

    Qwen3-VL expects pixel_values concatenated along the patch dimension
    (not stacked), with image_grid_thw indicating the grid size per image.

    Handles mixed batches of text-only and image samples:
    - pixel_values / image_grid_thw are only collected from samples that have them.
    - If no sample in the batch has images, these keys are omitted.
    """
    batch = {}
    # Union of all keys across samples
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())

    # Find max sequence length in this batch for dynamic padding
    pad_keys = {"input_ids", "attention_mask", "labels", "position_ids", "mm_token_type_ids", "loss_weight"}
    max_len = max(f["input_ids"].shape[0] for f in features)

    for key in all_keys:
        if key == "pixel_values":
            present = [f[key] for f in features if key in f]
            if present:
                batch[key] = torch.cat(present, dim=0)
        elif key == "image_grid_thw":
            present = [f[key] for f in features if key in f]
            if present:
                batch[key] = torch.stack(present, dim=0)
        elif key in pad_keys:
            # Dynamic pad each sequence to max_len in this batch
            # Skip if key is missing from some samples (e.g. position_ids)
            if not all(key in f for f in features):
                continue
            padded = []
            for f in features:
                seq = f[key]
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    if key == "labels":
                        pad_val = -100
                    elif key == "attention_mask":
                        pad_val = 0
                    elif key == "loss_weight":
                        pad_val = 0.0
                    else:
                        pad_val = pad_token_id
                    seq = F.pad(seq, (0, pad_len), value=pad_val)
                padded.append(seq)
            batch[key] = torch.stack(padded, dim=0)
        else:
            # Only stack keys present in ALL samples with matching shapes
            if all(key in f for f in features):
                shapes = [f[key].shape for f in features]
                if all(s == shapes[0] for s in shapes):
                    batch[key] = torch.stack([f[key] for f in features], dim=0)
                # else: skip variable-shape non-sequence keys (log warning)
                else:
                    logger.warning(f"Skipping key '{key}': variable shapes {shapes}")
    return batch
