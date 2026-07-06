#!/usr/bin/env python3
"""Stage 6: OPD (On-Policy Distillation) — Reverse KL Distillation.

Student = Unified RFT model (Stage 5 output)
Experts = Box Expert + Point Expert (Stage 4 output, frozen teachers)

Following the paper (Sec 2.5.4), we distill from each expert separately:
  1. Distill Box Expert on box-only data.
  2. Distill Point Expert on point+maze data.

Only one expert is kept in GPU memory at a time to avoid VRAM pressure.
"""

import hashlib
import os
import sys

import torch

from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.training.stage_runner import StageRunner
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.opd_trainer import train_opd_parallel
from src.training.memory_utils import log_memory_status
from src.utils.constants import DEFAULT_DISTILL_TEMPERATURE


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Determine resume checkpoint (explicit flag or latest auto checkpoint)
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt and not os.path.isdir(resume_ckpt):
        logger.error(f"Requested checkpoint not found: {resume_ckpt}")
        sys.exit(1)
    if not resume_ckpt:
        latest = runner.latest_checkpoint(args.output_dir)
        if latest:
            logger.info(f"Auto-resuming from latest checkpoint: {latest}")
            resume_ckpt = latest
        else:
            logger.info(f"No checkpoint found; starting fresh from {args.student_path}")

    # 2. Load Student (Unified RFT model) from checkpoint when resuming
    load_path = resume_ckpt if resume_ckpt else args.student_path
    logger.info(f"Loading student from: {load_path}")
    student_model, processor = load_qlora_model(
        model_name=load_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Student loaded:")

    # 2. Generate or load cached OPD training data
    cache_key = (
        f"{args.num_box}|{args.num_point}|{args.num_maze}|"
        f"{args.coco_image_dir}|{args.coco_ann_file}"
    )
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(
        args.output_dir, f"train_data_cache_{cache_hash}.pkl"
    )

    if args.regenerate_data and os.path.exists(cache_path):
        logger.info(f"--regenerate_data set; removing old cache {cache_path}")
        os.remove(cache_path)

    def generate_opd_data():
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
        return all_data

    all_data = runner.cached_data(cache_path, generate_opd_data)

    # Split by teacher routing
    box_data = [d for d in all_data if d.get("task_type") == "box"]
    point_data = [d for d in all_data if d.get("task_type") in ("point", "maze", "path")]
    logger.info(f"Routing: {len(box_data)} box samples, {len(point_data)} point/maze/path samples")

    # 3. Parallel OPD with gradient accumulation; experts are loaded on demand
    #    so that only one teacher (plus the student) is in GPU memory at a time.
    logger.info("Starting parallel OPD (gradient accumulation mode)...")
    train_opd_parallel(
        student_model=student_model,
        box_expert_path=args.box_expert_path,
        point_expert_path=args.point_expert_path,
        processor=processor,
        box_data=box_data,
        point_data=point_data,
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

    # Save student model
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 6 complete. Final model saved to {args.output_dir}")
    log_memory_status("Stage 6 complete:")


if __name__ == "__main__":
    runner = StageRunner(
        "stage6_opd",
        "configs/stage6_opd.yaml",
        description="Stage 6: OPD (On-Policy Distillation)",
    )
    runner.add_arg("--student_path", type=str, default=None)
    runner.add_arg("--box_expert_path", type=str, default=None)
    runner.add_arg("--point_expert_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_box", type=int, default=None)
    runner.add_arg("--num_point", type=int, default=None)
    runner.add_arg("--num_maze", type=int, default=None)
    runner.add_arg("--num_epochs", type=int, default=None)
    runner.add_arg("--learning_rate", type=float, default=None)
    runner.add_arg("--batch_size", type=int, default=None)
    runner.add_arg("--max_new_tokens", type=int, default=None)
    runner.add_arg("--temperature", type=float, default=DEFAULT_DISTILL_TEMPERATURE)
    runner.add_arg("--lora_r", type=int, default=None)
    runner.add_arg("--lora_alpha", type=int, default=None)
    runner.add_arg("--logging_steps", type=int, default=None)
    runner.add_arg("--warmup_steps", type=int, default=None)
    runner.add_arg("--save_steps", type=int, default=None)
    runner.add_arg("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume from, e.g. outputs/stage6_opd/checkpoint-500")
    runner.add_arg("--regenerate_data", action="store_true",
                   help="Force regeneration of training data and ignore existing cache.")
    runner.run(train)
