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
import random

import torch

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.models.qwen_vl_loader import load_qlora_model
from src.training.memory_utils import log_memory_status
from src.training.stage_runner import StageRunner
from src.training.trainers.sft_trainer import create_sft_trainer
from src.utils.constants import BASE_VOCAB_SIZE


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Load model. If resuming, load from the adapter checkpoint so that
    #    special-token embeddings remain as trained in Stage 2 instead of
    #    being overwritten by Stage 1 embeddings.
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    vit_unfreeze = getattr(args, "unfreeze_vit_layers", 0)
    if resume_ckpt:
        logger.info(f"Resuming Stage 2 from checkpoint: {resume_ckpt}")
        model, processor = load_qlora_model(
            model_name=resume_ckpt,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            pretrain_embedding_path=None,
            old_vocab_size=None,
            unfreeze_vit_layers=vit_unfreeze,
        )
    else:
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
            unfreeze_vit_layers=vit_unfreeze,
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
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    logger.info(f"Stage 2 complete. Model saved to {args.output_dir}")
    logger.info(f"Next: merge LoRA into base with scripts/merge_stage2.py")
    log_memory_status("Stage 2 complete:")


if __name__ == "__main__":
    runner = StageRunner(
        "stage2_visual_pretrain",
        "configs/stage2_visual_pretrain.yaml",
        description="Stage 2: Visual Pretrain",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--pretrain_embedding_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_box", type=int, default=None)
    runner.add_arg("--num_point", type=int, default=None)
    runner.add_arg("--curriculum", action="store_true",
                   help="Sort visual pretrain data from simple to complex")
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
                   help="Path to a Stage 2 checkpoint directory to resume from.")
    runner.add_arg("--unfreeze_vit_layers", type=int, default=None,
                   help="Unfreeze last N ViT blocks + merger (0 = all frozen)")
    runner.add_arg("--vit_lr", type=float, default=None,
                   help="LR for unfrozen ViT layers (very low)")
    runner.run(train)
