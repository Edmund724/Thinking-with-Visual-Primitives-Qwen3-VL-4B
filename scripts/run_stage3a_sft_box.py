#!/usr/bin/env python3
"""Stage 3a: Specialized SFT — Box Expert.

Trains a LoRA adapter specialized for box-type tasks (localization + counting).
Loads from merged Stage 2 base.

Data ratio: 70% general + 30% box-only visual primitives.
"""

import os

# Mitigate CUDA memory fragmentation from variable-length SFT completions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
    generate_coco_negative_box_samples,
)
from src.data.formatters.primitive_formatter import clean_primitive_tags
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.config_utils import apply_yaml_defaults
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
        target_general = int(len(visual_data) * 7 / 3)
        if len(general_data) > target_general:
            general_data = general_data[:target_general]
        logger.info(f"  General samples: {len(general_data)}")
    else:
        logger.warning(f"General data not found at {args.general_data_path}, using 100% visual")

    all_data = general_data + visual_data

    # Data cleaning: fix any wrong-order / duplicate primitive tags in the
    # SFT targets so the model is not trained on corrupted syntax.
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
        format_token_weight=args.format_token_weight,
    )

    logger.info("Starting Box Expert SFT training...")
    if args.resume_from_checkpoint and not os.path.isdir(args.resume_from_checkpoint):
        logger.error(f"Checkpoint not found: {args.resume_from_checkpoint}")
        sys.exit(1)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 3a complete. Model saved to {args.output_dir}")
    log_memory_status("Stage 3a complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3a: Box Expert SFT")
    parser.add_argument("--config", type=str, default="configs/stage3a_sft_box.yaml",
                        help="YAML config path; values are used as defaults unless overridden by CLI flags.")
    parser.add_argument("--model_path", type=str, default="outputs/stage2_merged_base")
    parser.add_argument("--output_dir", type=str, default="outputs/stage3a_sft_box")
    parser.add_argument("--general_data_path", type=str, default="data/pretrain/pretrain_data.json")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_box", type=int, default=15000)
    parser.add_argument("--num_counting", type=int, default=10000,
                        help="Number of coarse-grained counting samples")
    parser.add_argument("--counting_attribute_ratio", type=float, default=0.3,
                        help="Fraction of counting samples with color/size attribute constraints")
    parser.add_argument("--num_clevr", type=int, default=5000,
                        help="Number of CLEVR-style spatial/VQA samples")
    parser.add_argument("--clevr_negative_ratio", type=float, default=0.25,
                        help="Fraction of CLEVR samples that are faithful-refusal negatives")
    parser.add_argument("--num_negative_box", type=int, default=2000,
                        help="Number of COCO negative box samples (category not present)")
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from, e.g. outputs/stage3a_sft_box/checkpoint-500")
    parser.add_argument("--format_token_weight", type=float, default=5.0,
                        help="Loss weight multiplier for visual-primitive / think format tokens.")
    args = parser.parse_args()
    apply_yaml_defaults(args, parser, args.config)
    main(args)
