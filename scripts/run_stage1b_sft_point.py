#!/usr/bin/env python3
"""Stage 1b: Specialized SFT — Point Expert.

Trains a LoRA adapter specialized for point-type tasks (point grounding + maze navigation).
Loads from merged Stage 0.5 base.

Data ratio: 70% general + 30% visual primitives (20% maze + 10% point).
"""

import argparse
import logging
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import generate_coco_point_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage1b_sft_point.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 1b: Specialized SFT — Point Expert")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load from merged Stage 0.5 base
    logger.info(f"Loading from merged base: {args.model_path}")
    model, processor = load_qlora_model(
        model_name=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("After model loading:")

    # 2. Generate training data
    logger.info("Generating training data (70% general + 30% point/maze)...")

    # 10% point data (COCO object centers)
    point_data = generate_coco_point_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_point,
    )
    for d in point_data:
        d["task_type"] = "point"
    logger.info(f"  Point samples: {len(point_data)}")

    # 20% maze data
    maze_data = generate_maze_dataset(
        n=args.num_maze,
        seed=42,
    )
    for d in maze_data:
        d["task_type"] = "maze"
    logger.info(f"  Maze samples: {len(maze_data)}")

    # 70% general data (text-only pretrain data)
    general_data = []
    if os.path.exists(args.general_data_path):
        import json
        with open(args.general_data_path, "r") as f:
            general_data = json.load(f)
        # 70% of total = general, 30% = visual primitives
        visual_count = len(point_data) + len(maze_data)
        target_general = int(visual_count * 7 / 3)
        if len(general_data) > target_general:
            general_data = general_data[:target_general]
        for d in general_data:
            d["task_type"] = "general"
            if "image" not in d:
                d["image"] = None
        logger.info(f"  General samples: {len(general_data)}")
    else:
        logger.warning(f"General data not found at {args.general_data_path}, using 100% visual")

    all_data = general_data + point_data + maze_data
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

    logger.info("Starting Point Expert SFT training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 1b complete. Model saved to {args.output_dir}")
    log_memory_status("Stage 1b complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1b: Point Expert SFT")
    parser.add_argument("--model_path", type=str, default="outputs/stage0_5_merged_base")
    parser.add_argument("--output_dir", type=str, default="outputs/stage1b_sft_point")
    parser.add_argument("--general_data_path", type=str, default="data/pretrain/pretrain_data.json")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_point", type=int, default=10000)
    parser.add_argument("--num_maze", type=int, default=50000)
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
