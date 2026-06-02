#!/usr/bin/env python3
"""Stage 0: Pretrain — COCO Box Grounding (no thinking chain).

Following the paper's curriculum: first teach the model to output stable
box coordinates reliably, then add Chain-of-Thought reasoning in Stage 1.

Only uses COCO box data; no maze, no path, no <think> tags.
"""

import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage0_pretrain.log")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Stage 0: Pretrain (COCO Box Grounding — No Thinking)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load model + LoRA adapter
    logger.info("Loading model...")
    model, processor = load_qlora_model(
        model_name=config["base_model"],
        lora_r=config.get("lora_r", 256),
        lora_alpha=config.get("lora_alpha", 512),
    )
    log_memory_status("After model loading:")

    # 2. Generate COCO box-only training data
    logger.info("Generating training data (COCO box only)...")

    all_data = []

    coco_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_coco or config.get("num_coco_samples", 40000),
    )
    for d in coco_data:
        d["task_type"] = "box"
    all_data.extend(coco_data)
    logger.info(f"  COCO box samples: {len(coco_data)}")

    # Shuffle
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
        save_steps=config.get("save_steps", 200),
        warmup_steps=config.get("warmup_steps", 100),
        use_wandb=config.get("report_to") == "wandb",
    )

    logger.info("Starting SFT training (Pretrain)...")
    trainer.train()
    # For PEFT models, use model.save_pretrained to save adapter only
    model.save_pretrained(config["output_dir"])
    processor.save_pretrained(config["output_dir"])

    logger.info(f"Stage 0 complete. Model saved to {config['output_dir']}")
    log_memory_status("Stage 0 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Pretrain (COCO Box Grounding)")
    parser.add_argument("--config", type=str, default="configs/stage0_pretrain.yaml")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument(
        "--coco_ann_file", type=str,
        default="data/coco/annotations/instances_train2017.json",
    )
    parser.add_argument("--num_coco", type=int, default=None)
    main(parser.parse_args())
