"""Pretrain trainer — minimal training loop for embedding-only pretrain.

Uses a plain PyTorch training loop (NOT HuggingFace Trainer) because
Trainer rejects fully quantized models even when only embedding is trainable.

Simple: DataLoader + optimizer.step() + loss.backward().
Backbone is frozen; only embed_tokens.weight receives gradients.
"""

import logging
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """Dataset for pure-text pretrain (no images).

    Each sample:
        conversations: [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]

    Pre-tokenizes all samples at init time (fast, cheap since no images).
    Labels mask user tokens with -100.
    """

    def __init__(
        self,
        data: list,
        processor: AutoProcessor,
        max_length: int = 256,
    ):
        self.processor = processor
        self.max_length = max_length
        self._tokenized = []

        pad_token_id = processor.tokenizer.pad_token_id or 0

        for sample in data:
            messages = sample["conversations"]
            full_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            enc = processor.tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"][0]
            labels = input_ids.clone()

            # Mask user tokens: find where assistant content starts
            assistant_content = messages[1]["content"]
            try:
                idx = full_text.index(assistant_content)
                prefix = full_text[:idx]
                prefix_enc = processor.tokenizer(prefix, return_tensors="pt", padding=False)
                prompt_len = prefix_enc["input_ids"].shape[1]
                labels[:prompt_len] = -100
            except ValueError:
                labels[: len(labels) // 2] = -100

            # Pad to max_length
            if len(input_ids) < max_length:
                pad = torch.full((max_length - len(input_ids),), pad_token_id)
                input_ids = torch.cat([input_ids, pad])
                pad_labels = torch.full((max_length - len(labels),), -100)
                labels = torch.cat([labels, pad_labels])
            else:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            attention_mask = (input_ids != pad_token_id).long()

            self._tokenized.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self._tokenized)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self._tokenized[idx]


def train_pretrain(
    model,
    processor: AutoProcessor,
    train_data: list,
    output_dir: str,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    max_length: int = 256,
    warmup_steps: int = 200,
    logging_steps: int = 50,
    save_steps: int | None = None,
):
    """Run embedding-only pretrain with a minimal PyTorch training loop.

    Only embed_tokens (and lm_head if not tied) are trainable.
    Backbone is frozen — gradients never flow through it.

    Uses mixed precision (torch.cuda.amp) for speed.
    """
    dataset = PretrainDataset(
        data=train_data,
        processor=processor,
        max_length=max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        drop_last=False,
    )
    logger.info(
        f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch "
        f"(batch_size={per_device_batch_size})"
    )

    # Optimizer: only trainable parameters (embed_tokens ± lm_head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * num_epochs
    )

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            global_step += 1

            # Warmup
            if global_step <= warmup_steps:
                lr = learning_rate * global_step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr

            # Move to GPU
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # Forward + backward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.3)
            optimizer.step()
            optimizer.zero_grad()

            # Cosine schedule after warmup
            if global_step > warmup_steps:
                scheduler.step()

            epoch_loss += loss.item()

            if global_step % logging_steps == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{num_epochs} | "
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} complete. "
            f"Avg loss: {avg_loss:.4f}"
        )

    logger.info("Pretrain training complete.")
