"""SFT Trainer wrapper for QLoRA training of visual primitive models."""

import logging
from typing import Any, Dict, Optional

import torch
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
) -> Trainer:
    """Create a Trainer for SFT training."""

    train_dataset = SFTDataset(
        data=train_data,
        processor=processor,
        max_length=max_seq_length,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        data_collator=lambda features: _collate_sft(features),
        callbacks=callbacks,
    )

    return trainer


def _collate_sft(features: list) -> Dict[str, torch.Tensor]:
    """Collate SFT batch with variable-size image tensors (Qwen3-VL).

    Qwen3-VL expects pixel_values concatenated along the patch dimension
    (not stacked), with image_grid_thw indicating the grid size per image.
    """
    batch = {}
    keys = features[0].keys()
    for key in keys:
        if key == "pixel_values":
            # Different images → different num_patches; concatenate along dim 0
            batch[key] = torch.cat([f[key] for f in features], dim=0)
        elif key == "image_grid_thw":
            # Stack to [batch_size, 3]
            batch[key] = torch.stack([f[key] for f in features], dim=0)
        else:
            # input_ids, attention_mask, labels: already padded to max_length
            batch[key] = torch.stack([f[key] for f in features], dim=0)
    return batch
