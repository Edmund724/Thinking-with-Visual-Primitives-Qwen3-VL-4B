#!/usr/bin/env python3
"""Stage 3a: Specialized SFT — Box Expert.

Trains a LoRA adapter specialized for box-type tasks (localization + counting).
Loads from merged Stage 2 base.

Data ratio: 70% general + 30% box-only visual primitives.
"""

import hashlib
import os
import random

import torch

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.training.stage_runner import StageRunner
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
    generate_coco_negative_box_samples,
)
from src.data.formatters.primitive_formatter import clean_primitive_tags
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.callbacks import TimeLoggingCallback
from src.training.memory_utils import log_memory_status


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Determine resume checkpoint (explicit flag or latest auto checkpoint)
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt and not os.path.isdir(resume_ckpt):
        logger.error(f"Requested checkpoint not found: {resume_ckpt}")
        return
    if not resume_ckpt:
        latest = runner.latest_checkpoint(args.output_dir)
        if latest:
            logger.info(f"Auto-resuming from latest checkpoint: {latest}")
            resume_ckpt = latest
        else:
            logger.info(f"No checkpoint found; starting fresh from {args.model_path}")

    # 2. Load model from checkpoint when resuming, otherwise from merged Stage 2 base
    load_path = resume_ckpt if resume_ckpt else args.model_path
    logger.info(f"Loading model from: {load_path}")
    model, processor = load_qlora_model(
        model_name=load_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("After model loading:")

    # 2. Generate or load cached training data
    def _generate_sft_data():
        logger.info("Generating training data (70% general + 30% box reasoning)...")

        # 30% specialized box-type visual primitive data
        box_data = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_box,
        )
        for d in box_data:
            d["task_type"] = "box"
        logger.info(f"  Box localization samples: {len(box_data)}")

        counting_data = generate_coco_counting_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_counting,
            attribute_constraint_ratio=args.counting_attribute_ratio,
        )
        for d in counting_data:
            d["task_type"] = "box"
        logger.info(f"  Coarse-grained counting samples: {len(counting_data)}")

        clevr_data = generate_clevr_spatial_dataset(
            n=args.num_clevr,
            seed=44,
            cache_dir=os.path.join(args.output_dir, "clevr_cache"),
            negative_ratio=args.clevr_negative_ratio,
        )
        for d in clevr_data:
            d["task_type"] = "box"
        logger.info(f"  CLEVR spatial/VQA samples: {len(clevr_data)}")

        negative_box_data = generate_coco_negative_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_negative_box,
            seed=46,
        )
        for d in negative_box_data:
            d["task_type"] = "box"
        logger.info(f"  COCO negative box samples: {len(negative_box_data)}")

        visual_data = box_data + counting_data + clevr_data + negative_box_data

        # 70% general data (text-only pretrain data)
        general_data = []
        if args.general_data_path and os.path.exists(args.general_data_path):
            import json
            with open(args.general_data_path, "r") as f:
                raw_general = json.load(f)
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
            target_general = int(len(visual_data) * 7 / 3)
            if len(general_data) > target_general:
                general_data = general_data[:target_general]
            logger.info(f"  General samples: {len(general_data)}")
        else:
            logger.warning(f"General data not found at {args.general_data_path}, using 100% visual")

        all_data = general_data + visual_data

        # Data cleaning: fix any wrong-order / duplicate primitive tags
        cleaned_count = 0
        for d in all_data:
            if d.get("task_type") in ("box", "point"):
                original = d.get("reasoning", "")
                cleaned = clean_primitive_tags(original, task_type=d.get("task_type", "box"))
                if cleaned != original:
                    d["reasoning"] = cleaned
                    cleaned_count += 1
        if cleaned_count > 0:
            logger.info(f"  Cleaned primitive tags in {cleaned_count} samples")

        return all_data

    cache_key = (
        f"{args.num_box}|{args.num_counting}|{args.num_clevr}|{args.num_negative_box}|"
        f"{args.counting_attribute_ratio}|{args.clevr_negative_ratio}|"
        f"{args.coco_image_dir}|{args.coco_ann_file}|{args.general_data_path}"
    )
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    cache_path = os.path.join(
        args.output_dir, f"train_data_cache_{cache_hash}.pkl"
    )

    if args.regenerate_data and os.path.exists(cache_path):
        logger.info(f"--regenerate_data set; removing old cache {cache_path}")
        os.remove(cache_path)

    all_data = runner.cached_data(cache_path, _generate_sft_data)

    random.seed(42)
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
        max_grad_norm=args.max_grad_norm,
        format_token_weight=args.format_token_weight,
        additional_callbacks=[TimeLoggingCallback()],
    )

    logger.info("Starting Box Expert SFT training...")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 3a complete. Model saved to {args.output_dir}")
    log_memory_status("Stage 3a complete:")


if __name__ == "__main__":
    runner = StageRunner(
        "stage3a_sft_box",
        "configs/stage3a_sft_box.yaml",
        description="Stage 3a: Specialized SFT — Box Expert",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--general_data_path", type=str, default="data/pretrain/pretrain_data.json")
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_box", type=int, default=None)
    runner.add_arg("--num_counting", type=int, default=None,
                   help="Number of coarse-grained counting samples")
    runner.add_arg("--counting_attribute_ratio", type=float, default=None,
                   help="Fraction of counting samples with color/size attribute constraints")
    runner.add_arg("--num_clevr", type=int, default=None,
                   help="Number of CLEVR-style spatial/VQA samples")
    runner.add_arg("--clevr_negative_ratio", type=float, default=None,
                   help="Fraction of CLEVR samples that are faithful-refusal negatives")
    runner.add_arg("--num_negative_box", type=int, default=None,
                   help="Number of COCO negative box samples (category not present)")
    runner.add_arg("--num_epochs", type=int, default=None)
    runner.add_arg("--learning_rate", type=float, default=None)
    runner.add_arg("--batch_size", type=int, default=None)
    runner.add_arg("--gradient_accumulation_steps", type=int, default=None)
    runner.add_arg("--max_seq_length", type=int, default=None)
    runner.add_arg("--lora_r", type=int, default=None)
    runner.add_arg("--lora_alpha", type=int, default=None)
    runner.add_arg("--logging_steps", type=int, default=None)
    runner.add_arg("--save_steps", type=int, default=None)
    runner.add_arg("--warmup_steps", type=int, default=None)
    runner.add_arg("--resume_from_checkpoint", type=str, default=None,
                   help="Path to checkpoint dir to resume from, e.g. outputs/stage3a_sft_box/checkpoint-500")
    runner.add_arg("--format_token_weight", type=float, default=None,
                   help="Loss weight multiplier for visual-primitive / think format tokens.")
    runner.add_arg("--max_grad_norm", type=float, default=None,
                   help="Maximum gradient norm for clipping.")
    runner.add_arg("--regenerate_data", action="store_true",
                   help="Force regeneration of training data and ignore existing cache.")
    runner.run(train)
