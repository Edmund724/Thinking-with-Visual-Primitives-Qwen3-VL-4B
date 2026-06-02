#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Stage 1: SFT Unified — train box + maze + path tasks together."""

import argparse
import logging
import random
import yaml

import torch

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.data.generators.synthetic_path import generate_path_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage1_sft_unified.log")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Stage 1: SFT Unified (Box + Maze + Path)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load model + processor
    logger.info("Loading model...")
    model, processor = load_qlora_model(
        model_name=config["base_model"],
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
    )
    log_memory_status("After model loading:")

    # 2. Generate mixed training data
    logger.info("Generating training data...")

    all_data = []

    # COCO box samples
    coco_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_coco or config.get("num_coco_samples", 40000),
    )
    for d in coco_data:
        d["task_type"] = "box"
    all_data.extend(coco_data)
    logger.info(f"  COCO box samples: {len(coco_data)}")

    # Maze samples
    maze_data = generate_maze_dataset(
        n=args.num_maze or config.get("num_maze_samples", 50000),
        seed=42,
    )
    for d in maze_data:
        d["task_type"] = "maze"
    all_data.extend(maze_data)
    logger.info(f"  Maze samples: {len(maze_data)}")

    # Path samples
    path_data = generate_path_dataset(
        n=args.num_path or config.get("num_path_samples", 15000),
        seed=42,
    )
    for d in path_data:
        d["task_type"] = "point"
    all_data.extend(path_data)
    logger.info(f"  Path samples: {len(path_data)}")

    # Shuffle all data
    random.shuffle(all_data)
    logger.info(f"Total training samples: {len(all_data)}")

    # 3. Train
    trainer = create_sft_trainer(
        model=model,
        processor=processor,
        train_data=all_data,
        output_dir=config["output_dir"],
        num_epochs=config.get("num_train_epochs", 1),
        learning_rate=config.get("learning_rate", 1e-4),
        per_device_batch_size=config.get("per_device_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        max_seq_length=config.get("max_seq_length", 2048),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        warmup_steps=config.get("warmup_steps", 100),
        use_wandb=config.get("report_to") == "wandb",
    )

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(config["output_dir"])
    processor.save_pretrained(config["output_dir"])

    logger.info(f"Stage 1 complete. Model saved to {config['output_dir']}")
    log_memory_status("Stage 1 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: SFT Unified")
    parser.add_argument("--config", type=str, default="configs/stage1_sft_unified.yaml")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_coco", type=int, default=None)
    parser.add_argument("--num_maze", type=int, default=None)
    parser.add_argument("--num_path", type=int, default=None)
    main(parser.parse_args())
