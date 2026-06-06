#!/usr/bin/env python3
"""Stage 3a: Specialized SFT — Box Expert.

Trains a LoRA adapter specialized for box-type tasks (localization + counting).
Loads from merged Stage 2 base.

Data ratio: 70% general + 30% box-only visual primitives.
"""

import argparse
import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage3a_sft_box.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 3a: Specialized SFT — Box Expert")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load from merged Stage 2 base
    logger.info(f"Loading from merged base: {args.model_path}")
    model, processor = load_qlora_model(
        model_name=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("After model loading:")

    # 2. Generate training data
    logger.info("Generating training data (70% general + 30% box)...")

    # 30% box data
    box_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_box,
    )
    for d in box_data:
        d["task_type"] = "box"
    logger.info(f"  Box samples: {len(box_data)}")

    # 70% general data (text-only pretrain data)
    general_data = []
    if os.path.exists(args.general_data_path):
        import json
        with open(args.general_data_path, "r") as f:
            raw_general = json.load(f)
        # Convert from conversations format to SFT format
        for item in raw_general:
            convs = item.get("conversations", [])
            user_msg = next((c["content"] for c in convs if c["role"] == "user"), "")
            asst_msg = next((c["content"] for c in convs if c["role"] == "assistant"), "")
            general_data.append({
                "prompt": user_msg,
                "reasoning": "",
                "answer": asst_msg,
                "image": item.get("image", None),
                "task_type": "general",
            })
        # Trim to ~70% ratio
        target_general = int(len(box_data) * 7 / 3)
        if len(general_data) > target_general:
            general_data = general_data[:target_general]
        logger.info(f"  General samples: {len(general_data)}")
    else:
        logger.warning(f"General data not found at {args.general_data_path}, using 100% box")

    all_data = general_data + box_data
    random.shuffle(all_data)
    logger.info(f"Total training samples: {len(all_data)}")

    # 3. Train
    trainer = create_sft_trainer(
        model=model,
        processor=processor,
        train_data=all_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        use_wandb=False,
    )

    logger.info("Starting Box Expert SFT training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 3a complete. Model saved to {args.output_dir}")
    log_memory_status("Stage 3a complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3a: Box Expert SFT")
    parser.add_argument("--model_path", type=str, default="outputs/stage2_merged_base")
    parser.add_argument("--output_dir", type=str, default="outputs/stage3a_sft_box")
    parser.add_argument("--general_data_path", type=str, default="data/pretrain/pretrain_data.json")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_box", type=int, default=15000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    main(args)
