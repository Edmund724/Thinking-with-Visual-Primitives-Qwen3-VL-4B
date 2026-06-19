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
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from src.models.pretrain_loader import inject_pretrained_embeddings
from src.utils.constants import BASE_VOCAB_SIZE, SPECIAL_TOKENS
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

    # CRITICAL: inject Stage 1 pretrained embeddings before merging. Stage 2 was
    # trained on top of these embeddings, so losing them here would corrupt the
    # special-token representations.
    pretrain_path = (
        os.path.abspath(args.pretrain_embedding_path)
        if args.pretrain_embedding_path
        else None
    )
    if pretrain_path is not None and os.path.exists(pretrain_path):
        inject_pretrained_embeddings(
            model=model,
            pretrain_path=pretrain_path,
            old_vocab_size=BASE_VOCAB_SIZE,
        )
    else:
        logger.warning(
            f"No pretrained embedding state found at {pretrain_path}. "
            "Special-token embeddings will remain random."
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
    parser = argparse.ArgumentParser(description="Merge Stage 2 LoRA into base")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--pretrain_embedding_path", type=str,
                        default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
