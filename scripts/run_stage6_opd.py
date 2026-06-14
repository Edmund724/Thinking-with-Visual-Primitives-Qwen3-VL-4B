#!/usr/bin/env python3
"""Stage 6: OPD (Offline Preference Distillation) — Reverse KL Distillation.

Student = Unified RFT model (Stage 5 output)
Experts = Box Expert + Point Expert (Stage 4 output, frozen teachers)

For each sample:
  1. Student generates response (on-policy)
  2. Student and Expert forward on same sequence
  3. Reverse KL: D_KL(student || expert)
  4. Student LoRA updates to match expert distribution
"""

import os

# Mitigate CUDA memory fragmentation from variable-length OPD completions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import logging
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.opd_trainer import train_opd
from src.training.memory_utils import log_memory_status, clear_memory
from src.utils.logging_utils import setup_logging

logger = setup_logging(log_file="logs/stage6_opd.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 6: OPD (Offline Preference Distillation)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load Student (Unified RFT model)
    logger.info(f"Loading student from {args.student_path}...")
    student_model, processor = load_qlora_model(
        model_name=args.student_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Student loaded:")

    # 2. Load Box Expert (frozen teacher)
    logger.info(f"Loading Box Expert from {args.box_expert_path}...")
    box_expert, _ = load_qlora_model(
        model_name=args.box_expert_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Box Expert loaded:")

    # 3. Load Point Expert (frozen teacher)
    logger.info(f"Loading Point Expert from {args.point_expert_path}...")
    point_expert, _ = load_qlora_model(
        model_name=args.point_expert_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Point Expert loaded:")

    # 4. Generate or load cached OPD training data
    cache_path = os.path.join(args.output_dir, "train_data_cache.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached training data from {cache_path}")
        with open(cache_path, "rb") as f:
            all_data = pickle.load(f)
        logger.info(f"  Loaded {len(all_data)} samples from cache")
    else:
        logger.info("Generating OPD training data...")
        all_data = []

        box_data = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_box,
        )
        for d in box_data:
            d["task_type"] = "box"
        all_data.extend(box_data)
        logger.info(f"  Box: {len(box_data)}")

        point_data = generate_coco_point_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_point,
        )
        for d in point_data:
            d["task_type"] = "point"
        all_data.extend(point_data)
        logger.info(f"  Point: {len(point_data)}")

        maze_data = generate_maze_dataset(
            n=args.num_maze,
            seed=99,
        )
        for d in maze_data:
            d["task_type"] = "maze"
        all_data.extend(maze_data)
        logger.info(f"  Maze: {len(maze_data)}")

        logger.info(f"Total OPD samples: {len(all_data)}")

        # Save cache for future runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Cached training data to {cache_path}")

    # 5. Run OPD training
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt and not os.path.isdir(resume_ckpt):
        logger.error(f"Checkpoint not found: {resume_ckpt}")
        sys.exit(1)

    logger.info("Starting OPD training (reverse KL distillation)...")
    train_opd(
        student_model=student_model,
        box_expert=box_expert,
        point_expert=point_expert,
        processor=processor,
        train_data=all_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        resume_from_checkpoint=resume_ckpt,
        logger=logger,
    )

    # Release experts before saving final student to free VRAM
    logger.info("Releasing expert models after OPD training...")
    del box_expert
    del point_expert
    gc.collect()
    clear_memory()
    log_memory_status("Expert models released:")

    # Save student model
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 6 complete. Final model saved to {args.output_dir}")
    log_memory_status("Stage 6 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: OPD")
    parser.add_argument("--student_path", type=str, default="outputs/stage5_rft_unified/final_model")
    parser.add_argument("--box_expert_path", type=str, default="outputs/stage4a_grpo_box")
    parser.add_argument("--point_expert_path", type=str, default="outputs/stage4b_grpo_point")
    parser.add_argument("--output_dir", type=str, default="outputs/stage6_opd")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_box", type=int, default=3000)
    parser.add_argument("--num_point", type=int, default=2000)
    parser.add_argument("--num_maze", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from, e.g. outputs/stage6_opd/checkpoint-500")
    args = parser.parse_args()
    main(args)
