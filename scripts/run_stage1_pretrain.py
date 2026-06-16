#!/usr/bin/env python3
"""Stage 1: Text Pretrain — Initialize visual primitive token embeddings.

Following the paper's curriculum:
  Stage 1: Learn "hand movement" (stable embedding for new tokens)
  Stage 2+: Learn "when and how to think" (integrate into Chain-of-Thought)

Pure text-only training. No images. No QLoRA.
Only embed_tokens (and lm_head if not tied) are trained.

Optionally mix in real COCO categories + coordinates (--coco_grounding_ratio)
to move beyond purely random coordinates while still keeping Stage 1 lightweight.

Expected: ~30 min on RTX 5090D for 25K samples × 3 epochs.
"""

import os

# Mitigate CUDA memory fragmentation during training.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.pretrain_loader import (
    load_pretrain_model,
    save_pretrain_state,
)
from src.training.pretrain_trainer import train_pretrain
from src.utils.config_utils import apply_yaml_defaults
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage1_pretrain.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 1: Text Pretrain — Embedding-only Training")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Generate pretrain data if not exists
    if not os.path.exists(args.data_path):
        logger.info(f"Data not found. Generating {args.num_samples} samples...")
        from scripts.generate_pretrain_data import generate_dataset, export_for_training

        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        data = generate_dataset(
            n=args.num_samples,
            seed=42,
            coco_ann_file=args.coco_ann_file,
            coco_grounding_ratio=args.coco_grounding_ratio,
            curriculum=args.curriculum,
        )
        export_for_training(data, args.data_path)
    else:
        logger.info(f"Loading existing data from {args.data_path}...")
        with open(args.data_path, "r") as f:
            data = json.load(f)
        # Trim to requested sample count
        if args.num_samples < len(data):
            data = data[:args.num_samples]
        logger.info(f"Loaded {len(data)} pretrain conversation samples")

    # 2. Load model (4-bit quantized, embedding-only trainable)
    logger.info("Loading model (4-bit, ~2GB RAM)...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model, processor, old_vocab_size = load_pretrain_model(
        model_name=args.model_path,
        attn_impl=attn_impl,
    )

    # 3. Train
    logger.info("Starting embedding-only pretrain...")
    train_pretrain(
        model=model,
        processor=processor,
        train_data=data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        logger=logger,
    )

    # 4. Save only trained embedding weights
    logger.info("Saving pretrained embedding state...")
    save_pretrain_state(
        model=model,
        processor=processor,
        output_dir=args.output_dir,
        old_vocab_size=old_vocab_size,
    )

    logger.info("=" * 60)
    logger.info(f"Stage 1 complete. State saved to {args.output_dir}/")
    logger.info(f"Next: run Stage 2 with --pretrain_embedding_path {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Text Pretrain (Embedding)")
    parser.add_argument("--config", type=str, default="configs/stage1_pretrain.yaml",
                        help="YAML config path; values are used as defaults unless overridden by CLI flags.")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-VL-4B-Thinking")
    parser.add_argument("--data_path", type=str, default="data/pretrain/pretrain_data.json")
    parser.add_argument("--output_dir", type=str, default="outputs/stage1_pretrain")
    parser.add_argument("--num_samples", type=int, default=25000)
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json",
                        help="Optional COCO annotations to mix real categories/coordinates into pretrain data")
    parser.add_argument("--coco_grounding_ratio", type=float, default=0.0,
                        help="Fraction of pretrain samples to draw from COCO (0.0~0.5)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort pretrain data from short/simple to long/complex")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps for embedding-only pretrain")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=200)
    args = parser.parse_args()
    apply_yaml_defaults(args, parser, args.config)
    main(args)
