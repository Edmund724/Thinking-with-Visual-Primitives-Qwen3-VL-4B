#!/usr/bin/env python3
"""Stage 1: Unified Visual Grounding Pretrain.

Establishes the foundational "visual feature → coordinate" mapping
directly on images.  Special tokens (<|box|>, <|point|>, etc.) are
randomly initialized and learned alongside the LoRA adapter during
training — no separate text-only format pretrain needed.

Trainable:
  - ViT: last N layers optionally unfrozen (--unfreeze_vit_layers)
  - Vision-Language Projection: TRAINABLE
  - LLM layers: TRAINABLE via LoRA (low LR)
  - Special token embeddings: TRAINABLE

Data:
  - COCO: box localization, point (object centers), coarse-grained counting
  - CLEVR: synthetic spatial reasoning / VQA

After training: run scripts/merge_stage1.py to merge LoRA into base.
"""

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
    generate_coco_point_samples,
)
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.memory_utils import log_memory_status
from src.training.trainers.sft_trainer import create_sft_trainer


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Load model with LoRA. Special tokens are randomly initialized —
    #    they will be learned during visual pretrain (no pretrain embedding
    #    injection needed in the unified flow).
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    vit_unfreeze = getattr(args, "unfreeze_vit_layers", 0)
    if resume_ckpt:
        logger.info(f"Resuming Stage 1 from checkpoint: {resume_ckpt}")
        model, processor = load_qlora_model(
            model_name=resume_ckpt,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            pretrain_embedding_path=None,
            old_vocab_size=None,
            unfreeze_vit_layers=vit_unfreeze,
        )
    else:
        logger.info(
            f"Loading base model from {args.model_path} "
            f"(special tokens will be randomly initialized)"
        )
        model, processor = load_qlora_model(
            model_name=args.model_path,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            pretrain_embedding_path=None,
            old_vocab_size=None,
            unfreeze_vit_layers=vit_unfreeze,
        )
    log_memory_status("After model loading:")

    # 2. Generate visual pretrain data (COCO + CLEVR)
    logger.info("Generating unified visual pretrain data...")

    all_data = []

    # COCO box samples (localization)
    box_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_box,
        use_thinking=False,  # simplified format: learn "see → mark"
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

    # CLEVR spatial reasoning / VQA (adds shape/color/material diversity)
    if args.num_clevr > 0:
        clevr_data = generate_clevr_spatial_dataset(
            n=args.num_clevr,
            seed=43,
            cache_dir=os.path.join(args.output_dir, "clevr_cache"),
        )
        for d in clevr_data:
            d["task_type"] = "box"
        all_data.extend(clevr_data)
        logger.info(f"  CLEVR spatial samples: {len(clevr_data)}")

    if args.curriculum:
        def _visual_complexity(d):
            text = d.get("reasoning", "") + " " + d.get("answer", "")
            return (
                text.count("<|box|>")
                + text.count("<|point|>")
                + len(text.split())
            )
        all_data.sort(key=_visual_complexity)
        logger.info("Applied Stage 1 curriculum: short-to-long token sequences.")
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

    logger.info("Starting unified visual pretrain...")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 1 complete. Model saved to {args.output_dir}")
    logger.info(f"Next: merge LoRA into base with scripts/merge_stage2.py")
    log_memory_status("Stage 1 complete:")


if __name__ == "__main__":
    runner = StageRunner(
        "stage1_visual_pretrain",
        "configs/stage1_visual_pretrain.yaml",
        description="Stage 1: Unified Visual Grounding Pretrain",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_box", type=int, default=None)
    runner.add_arg("--num_point", type=int, default=None)
    runner.add_arg("--num_clevr", type=int, default=None,
                   help="Number of CLEVR-style spatial/VQA samples (0 to skip)")
    runner.add_arg("--curriculum", action="store_true",
                   help="Sort pretrain data from simple to complex")
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
                   help="Path to a Stage 1 checkpoint directory to resume from.")
    runner.add_arg("--unfreeze_vit_layers", type=int, default=None,
                   help="Unfreeze last N ViT blocks + merger (0 = all frozen)")
    runner.run(train)
