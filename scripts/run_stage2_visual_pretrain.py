#!/usr/bin/env python3
"""Stage 2: Visual Pretrain — Establish visual feature → coordinate mapping.

Uses COCO images + box/point annotations to teach the model real visual
grounding, not just token syntax (which Stage 1 already handled).

Trainable:
  - ViT: FROZEN
  - Vision-Language Projection: TRAINABLE
  - LLM layers: TRAINABLE via LoRA (low LR)
  - Special token embeddings: TRAINABLE

Data:
  - Box: COCO train2017, ~50K samples
  - Point: COCO object centers, ~10K samples
  - Total: ~60K samples

After training: run scripts/merge_stage2.py to merge LoRA into base.
"""

import os

# Mitigate CUDA memory fragmentation from variable-length visual sequences.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.config_utils import apply_yaml_defaults
from src.utils.logging_utils import setup_logging
from src.utils.constants import BASE_VOCAB_SIZE

logger = setup_logging(log_file="logs/stage2_visual_pretrain.log")


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 2: Visual Pretrain")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load base model + inject Stage 1 pretrain embeddings
    base_model = args.model_path
    pretrain_path = args.pretrain_embedding_path

    logger.info(
        f"Loading from {base_model} + injecting pretrained embeddings from {pretrain_path}..."
    )
    model, processor = load_qlora_model(
        model_name=base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        pretrain_embedding_path=os.path.join(pretrain_path, "pretrain_state_dict.pt"),
        old_vocab_size=BASE_VOCAB_SIZE,
    )
    log_memory_status("After model loading:")

    # 2. Generate visual pretrain data
    logger.info("Generating visual pretrain data...")

    all_data = []

    # COCO box samples
    box_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_box,
        use_thinking=False,
    )
    for d in box_data:
        d["task_type"] = "box"
    all_data.extend(box_data)
    logger.info(f"  COCO box samples: {len(box_data)}")

    # COCO point samples (object centers)
    point_data = generate_coco_point_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_point,
        use_thinking=False,
    )
    for d in point_data:
        d["task_type"] = "point"
    all_data.extend(point_data)
    logger.info(f"  COCO point samples: {len(point_data)}")

    if args.curriculum:
        # Sort from simple (few boxes/points) to complex.
        def _visual_complexity(d):
            text = d.get("reasoning", "") + " " + d.get("answer", "")
            return (
                text.count("<|box|>")
                + text.count("<|point|>")
                + len(text.split())
            )
        all_data.sort(key=_visual_complexity)
        logger.info("Applied Stage 2 curriculum: short-to-long token sequences.")
    else:
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

    logger.info("Starting visual pretrain...")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 2 complete. Model saved to {args.output_dir}")
    logger.info(f"Next: merge LoRA into base with scripts/merge_stage2.py")
    log_memory_status("Stage 2 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Visual Pretrain")
    parser.add_argument("--config", type=str, default="configs/stage2_visual_pretrain.yaml",
                        help="YAML config path; values are used as defaults unless overridden by CLI flags.")
    parser.add_argument("--model_path", type=str, default="models/Qwen3-VL-4B-Thinking")
    parser.add_argument("--pretrain_embedding_path", type=str, default="outputs/stage1_pretrain")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2_visual_pretrain")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_box", type=int, default=50000)
    parser.add_argument("--num_point", type=int, default=10000)
    parser.add_argument("--curriculum", action="store_true",
                        help="Sort visual pretrain data from simple to complex")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()
    apply_yaml_defaults(args, parser, args.config)
    main(args)
