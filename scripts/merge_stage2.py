#!/usr/bin/env python3
"""Merge Stage 2 LoRA adapter into base model.

After Stage 1 unified visual grounding pretrain, the LoRA weights need to
be merged into the base model so that Stage 3a/3b can start from a clean
base with visual grounding ability baked in (no nested LoRA adapters).

Special token embeddings were learned during Stage 1 visual pretrain
together with the LoRA adapter — no separate pretrain embedding injection
is needed.

Usage:
    python scripts/merge_stage2.py \
        --base_model models/Qwen3-VL-4B-Thinking \
        --adapter_path outputs/stage1_visual_pretrain \
        --output_dir outputs/stage2_merged_base
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.utils.constants import SPECIAL_TOKENS
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/merge_stage2.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Merging Stage 1 LoRA into base model")
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

    # Load tokenizer from adapter (has special tokens) and add them to base model
    processor = AutoProcessor.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )
    special_tokens_dict = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = processor.tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"Added {num_added} special tokens: {SPECIAL_TOKENS}")

    current_embed_size = model.get_input_embeddings().num_embeddings
    new_tokenizer_len = len(processor.tokenizer)
    if new_tokenizer_len > current_embed_size:
        model.resize_token_embeddings(new_tokenizer_len)
        logger.info(f"Resized embeddings: {current_embed_size} → {new_tokenizer_len}")
    else:
        logger.info(
            f"No resize needed: embedding ({current_embed_size}) covers tokenizer ({new_tokenizer_len})"
        )

    # Align config with tokenizer
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.bos_token_id = processor.tokenizer.bos_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
        model.generation_config.bos_token_id = processor.tokenizer.bos_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    logger.info(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Merge complete. Merged model saved to {args.output_dir}")
    logger.info("Next: Stage 3a/3b will load from this merged base.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Stage 1 LoRA into base")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
