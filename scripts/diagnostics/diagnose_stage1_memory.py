#!/usr/bin/env python3
"""Diagnose Stage 1 GPU memory allocation and Windows-shared-memory-like behavior.

Reproduces the Stage 1 model load + one training step with the same settings as
configs/stage1_visual_pretrain.yaml, then prints detailed CUDA memory stats.

Run examples:
    python scripts/diagnostics/diagnose_stage1_memory.py
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True python scripts/diagnostics/diagnose_stage1_memory.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import _collate_sft
from src.utils.constants import MAX_GPU_MEMORY_GB


def _smi() -> str:
    """Return a compact nvidia-smi memory line."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def log_memory(label: str, print_summary: bool = True) -> None:
    """Log allocated/reserved memory and optionally a compact summary."""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"\n[{label}] torch allocated={allocated:.2f}GB  reserved={reserved:.2f}GB  gap={reserved - allocated:.2f}GB")
    print(f"[{label}] nvidia-smi used/free/total (MB): {_smi()}")
    if print_summary:
        print(torch.cuda.memory_summary(device=torch.cuda.current_device(), abbreviated=True))


def build_tiny_dataset(processor, max_length: int):
    """Build a tiny COCO box+point dataset for diagnosis."""
    print("Generating 4 COCO box + 4 COCO point samples...")
    box_data = generate_coco_box_samples(
        image_dir="data/coco/train2017",
        ann_file="data/coco/annotations/instances_train2017.json",
        num_samples=4,
        use_thinking=False,
    )
    for d in box_data:
        d["task_type"] = "box"

    point_data = generate_coco_point_samples(
        image_dir="data/coco/train2017",
        ann_file="data/coco/annotations/instances_train2017.json",
        num_samples=4,
        use_thinking=False,
    )
    for d in point_data:
        d["task_type"] = "point"

    return SFTDataset(box_data + point_data, processor, max_length=max_length)


def run_training_step(model, batch, pad_token_id: int):
    """Run one forward/backward/optimizer step mirroring WeightedSFTTrainer."""
    labels = batch.pop("labels")
    loss_weight = batch.pop("loss_weight", None)

    outputs = model(**batch)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = (
        loss_weight[..., 1:].contiguous() if loss_weight is not None else None
    )

    losses = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    mask = (shift_labels != -100).view(-1)
    if mask.any():
        weights = (
            shift_weights.view(-1)[mask]
            if shift_weights is not None
            else torch.ones_like(losses[mask])
        )
        loss = (losses[mask] * weights).sum() / weights.sum()
    else:
        loss = losses.sum() * 0.0

    loss.backward()
    return loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alloc_conf",
        type=str,
        default=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        help="Value to set for PYTORCH_CUDA_ALLOC_CONF.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--skip_optimizer", action="store_true")
    parser.add_argument(
        "--test_callback_fraction",
        type=float,
        default=None,
        help="If set, run MemoryMonitorCallback smoke test with this reserved-memory fraction threshold.",
    )
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.alloc_conf
    print(f"PYTORCH_CUDA_ALLOC_CONF={args.alloc_conf}")
    print(f"Settings: batch_size={args.batch_size}, max_seq_length={args.max_seq_length}, lora_r={args.lora_r}, lora_alpha={args.lora_alpha}")

    torch.cuda.empty_cache()
    log_memory("START", print_summary=False)

    # 1. Load model exactly like Stage 1.
    print("\nLoading Qwen3-VL-4B-Thinking with QLoRA (this may take a while)...")
    model, processor = load_qlora_model(
        model_name="models/Qwen3-VL-4B-Thinking",
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing=True,
        unfreeze_vit_layers=0,
    )
    log_memory("AFTER MODEL LOAD")

    # 2. Build a tiny dataset and one batch.
    dataset = build_tiny_dataset(processor, args.max_seq_length)
    features = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    pad_token_id = processor.tokenizer.pad_token_id or 0
    batch = _collate_sft(features, pad_token_id=pad_token_id)

    # Move tensors to GPU; keep lists (image_grid_thw is a tensor so it moves).
    batch = {
        k: v.cuda(non_blocking=False) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    log_memory("AFTER BATCH TO GPU", print_summary=False)

    # 3. Forward + backward.
    model.train()
    loss_val = run_training_step(model, batch, pad_token_id)
    print(f"\nOne-step loss: {loss_val:.4f}")
    log_memory("AFTER BACKWARD")

    # 4. Optional optimizer step to see optimizer-state footprint.
    # Stage 1 uses paged_adamw_8bit; use it here for a realistic measurement.
    if not args.skip_optimizer:
        try:
            from bitsandbytes.optim import PagedAdamW8bit

            optimizer = PagedAdamW8bit(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-6,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            print("Using PagedAdamW8bit (matches Stage 1 TrainingArguments).")
        except Exception as exc:
            print(f"PagedAdamW8bit unavailable ({exc}), falling back to AdamW.")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-6
            )
        optimizer.step()
        log_memory("AFTER OPTIMIZER STEP")

    # 5. Optional callback smoke test: verify MemoryMonitorCallback can detect
    # high reserved memory and clear the cache.
    if args.test_callback_fraction is not None:
        from src.training.callbacks import MemoryMonitorCallback

        print(
            f"\nSmoke-testing MemoryMonitorCallback with "
            f"max_reserved_fraction={args.test_callback_fraction}..."
        )
        log_memory("BEFORE CALLBACK TEST", print_summary=False)
        callback = MemoryMonitorCallback(
            max_memory_gb=MAX_GPU_MEMORY_GB,
            max_reserved_fraction=args.test_callback_fraction,
        )
        state = argparse.Namespace(global_step=1)
        control = argparse.Namespace()
        callback.on_step_begin(None, state, control)
        log_memory("AFTER CALLBACK TEST", print_summary=False)

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
