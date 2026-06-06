#!/usr/bin/env python3
"""Merge Stage 2 LoRA adapter into base model.

After Stage 2 visual pretrain, the LoRA weights need to be merged into
the base model so that Stage 3a/3b can start from a clean base with
visual grounding ability baked in (no nested LoRA adapters).

Usage:
    python scripts/merge_stage2.py \
        --base_model models/Qwen3-VL-4B-Thinking \
        --adapter_path outputs/stage2_visual_pretrain \
        --output_dir outputs/stage2_merged_base
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/merge_stage2.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Merging Stage 2 LoRA into base model")
    logger.info("=" * 60)

    # Resolve relative paths to absolute
    args.base_model = os.path.abspath(args.base_model)
    args.adapter_path = os.path.abspath(args.adapter_path)
    args.output_dir = os.path.abspath(args.output_dir)

    logger.info(f"Loading base model: {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Also save tokenizer from adapter (has special tokens)
    processor = AutoProcessor.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )
    processor.save_pretrained(args.output_dir)

    logger.info(f"Merge complete. Merged model saved to {args.output_dir}")
    logger.info("Next: Stage 3a/3b will load from this merged base.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Stage 2 LoRA into base")
    parser.add_argument("--base_model", type=str, default="models/Qwen3-VL-4B-Thinking")
    parser.add_argument("--adapter_path", type=str, default="outputs/stage2_visual_pretrain")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_merged_base")
    args = parser.parse_args()
    main(args)
