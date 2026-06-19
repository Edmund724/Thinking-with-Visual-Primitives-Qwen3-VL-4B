#!/usr/bin/env python3
"""Diagnose why Stage 2 resume loss differs from pre-interruption loss.

Loads a checkpoint and evaluates the *same fixed batch* with three loss modes:
1. Unweighted (all assistant tokens weight=1) — should match original loss ~0.8.
2. Weighted with current implementation — reports current resume loss.
3. Weighted with sum-of-weights denominator — corrected weighted average.

Also dumps checkpoint state to verify optimizer/lr_scheduler restoration.
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor

from src.models.qwen_vl_loader import load_qlora_model
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.data.datasets.sft_dataset import SFTDataset
from src.training.trainers.sft_trainer import _collate_sft


def compute_loss_unweighted(logits, labels):
    """CrossEntropy mean over non-masked tokens (original behaviour)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    mask = (shift_labels != -100).view(-1)
    return (losses[mask].sum() / mask.sum()).item()


def compute_loss_weighted_current(logits, labels, loss_weight):
    """Current WeightedSFTTrainer implementation."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weight[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    losses = losses * shift_weights.view(-1)
    mask = (shift_labels != -100).view(-1)
    return (losses[mask].sum() / mask.sum()).item()


def compute_loss_weighted_fixed(logits, labels, loss_weight):
    """Weighted average using sum of weights as denominator."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weight[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    losses = losses * shift_weights.view(-1)
    mask = (shift_labels != -100).view(-1)
    weights = shift_weights.view(-1)[mask]
    return (losses[mask].sum() / weights.sum()).item()


def main(args):
    print(f"Loading checkpoint: {args.checkpoint}")
    model, processor = load_qlora_model(
        model_name=args.checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        pretrain_embedding_path=None,
        old_vocab_size=None,
    )
    model.eval()

    # Load trainer state to compare.
    state_path = os.path.join(args.checkpoint, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"Checkpoint global_step: {state.get('global_step')}")
        print(f"Checkpoint epoch: {state.get('epoch')}")
        print(f"Last logged loss: {state['log_history'][-1]}")

    # Generate a tiny deterministic dataset (same generator, fixed seed if possible).
    torch.manual_seed(args.seed)
    data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_samples,
        use_thinking=False,
    )
    for d in data:
        d["task_type"] = "box"

    dataset = SFTDataset(
        data=data,
        processor=processor,
        max_length=args.max_seq_length,
        format_token_weight=args.format_token_weight,
    )
    batch = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    collated = _collate_sft(batch, pad_token_id=processor.tokenizer.pad_token_id)

    # Move to device.
    device = next(model.parameters()).device
    collated = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in collated.items()}

    labels = collated.pop("labels")
    loss_weight = collated.pop("loss_weight", None)

    with torch.no_grad():
        outputs = model(**collated)
        logits = outputs.logits

    print("\n--- Loss comparison on same fixed batch ---")
    print(f"Unweighted (original behaviour):        {compute_loss_unweighted(logits, labels):.4f}")
    if loss_weight is not None:
        print(f"Weighted current (token-count denom):   {compute_loss_weighted_current(logits, labels, loss_weight):.4f}")
        print(f"Weighted fixed (sum-of-weights denom):  {compute_loss_weighted_fixed(logits, labels, loss_weight):.4f}")
        print(f"Format token weight used:               {args.format_token_weight}")
        fmt_frac = (loss_weight > 1.0).float().mean().item()
        print(f"Fraction of positions with weight > 1:  {fmt_frac:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default=None)
    parser.add_argument("--coco_ann_file", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None, help="Samples to draw from COCO")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--format_token_weight", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
