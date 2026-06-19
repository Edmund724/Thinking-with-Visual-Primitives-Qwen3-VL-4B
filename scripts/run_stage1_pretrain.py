#!/usr/bin/env python3
"""Stage 1: Text + Visual Pretrain — Initialize visual primitive token embeddings.

Following the paper's curriculum:
  Stage 1: Learn "hand movement" (stable embedding for new tokens)
  Stage 2+: Learn "when and how to think" (integrate into Chain-of-Thought)

Two-phase training:
  Phase 1: Pure text format pretrain (teaches token format and embedding)
  Phase 2 (optional): Visual grounding pretrain with real COCO images
           (teaches visual feature → coordinate mapping; enabled via --visual_data_ratio > 0)

Optionally unfreeze last ViT layers (--unfreeze_vit_layers 1-2) with very low LR
(--vit_lr 1e-6) to experimentally improve coordinate precision.

Expected: ~30 min text + optional visual phase on RTX 5090D.
"""

import json
import os

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_point_samples,
)
from src.models.pretrain_loader import (
    load_pretrain_model,
    save_pretrain_state,
)
from src.training.pretrain_trainer import train_pretrain, train_pretrain_visual
from src.training.stage_runner import StageRunner


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # ── Phase 1: Text pretrain data ───────────────────────────────────
    if not os.path.exists(args.data_path):
        logger.info(f"Data not found. Generating {args.num_samples} samples...")
        from scripts.generate_pretrain_data import generate_dataset, export_for_training

        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        data = generate_dataset(
            n=args.num_samples,
            seed=42,
            coco_ann_file=args.coco_ann_file,
            coco_grounding_ratio=args.coco_grounding_ratio,
            curriculum=args.curriculum,
        )
        export_for_training(data, args.data_path)
    else:
        logger.info(f"Loading existing data from {args.data_path}...")
        with open(args.data_path, "r") as f:
            data = json.load(f)
        if args.num_samples < len(data):
            data = data[:args.num_samples]
        logger.info(f"Loaded {len(data)} pretrain conversation samples")

    # ── Phase 1: Load model ───────────────────────────────────────────
    logger.info("Loading model (4-bit, ~2GB RAM)...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    vit_unfreeze = getattr(args, "unfreeze_vit_layers", 0)
    model, processor, old_vocab_size = load_pretrain_model(
        model_name=args.model_path,
        attn_impl=attn_impl,
        unfreeze_vit_layers=vit_unfreeze,
    )

    # ── Phase 1: Text format pretrain ─────────────────────────────────
    logger.info("Phase 1: Text format pretrain (embedding + decoder layers)...")
    train_pretrain(
        model=model,
        processor=processor,
        train_data=data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        vit_lr=args.vit_lr,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_steps=args.warmup_steps,
        logger=logger,
    )

    # ── Phase 2: Visual grounding pretrain (optional) ─────────────────
    if args.visual_data_ratio > 0:
        logger.info("=" * 60)
        logger.info("Phase 2: Visual grounding pretrain with COCO images")
        logger.info(f"  Visual data ratio: {args.visual_data_ratio}")
        logger.info(f"  ViT layers unfrozen: {vit_unfreeze}, ViT LR: {args.vit_lr}")
        logger.info("=" * 60)

        logger.info("Generating COCO visual pretrain data...")
        visual_data = []
        coco_dir = args.coco_image_dir
        coco_ann = args.coco_ann_file

        # Box grounding samples
        if args.visual_num_box > 0:
            box_samples = generate_coco_box_samples(
                image_dir=coco_dir,
                ann_file=coco_ann,
                num_samples=args.visual_num_box,
                use_thinking=False,  # simplified format for Stage 1
            )
            for d in box_samples:
                d["task_type"] = "box"
            visual_data.extend(box_samples)
            logger.info(f"  COCO box samples: {len(box_samples)}")

        # Point grounding samples
        if args.visual_num_point > 0:
            point_samples = generate_coco_point_samples(
                image_dir=coco_dir,
                ann_file=coco_ann,
                num_samples=args.visual_num_point,
                use_thinking=False,
            )
            for d in point_samples:
                d["task_type"] = "point"
            visual_data.extend(point_samples)
            logger.info(f"  COCO point samples: {len(point_samples)}")

        logger.info(f"Total visual samples: {len(visual_data)}")

        # Run visual pretrain
        train_pretrain_visual(
            model=model,
            processor=processor,
            train_data=visual_data,
            output_dir=args.output_dir,
            num_epochs=args.visual_epochs,
            learning_rate=args.visual_learning_rate,
            vit_lr=args.vit_lr,
            per_device_batch_size=args.visual_batch_size,
            gradient_accumulation_steps=args.visual_gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            warmup_steps=max(10, args.warmup_steps // 4),
            logger=logger,
        )

    # ── Save ──────────────────────────────────────────────────────────
    logger.info("Saving pretrained embedding state...")
    save_pretrain_state(
        model=model,
        processor=processor,
        output_dir=args.output_dir,
        old_vocab_size=old_vocab_size,
    )

    logger.info("=" * 60)
    logger.info(f"Stage 1 complete. State saved to {args.output_dir}/")
    logger.info(f"Next: run Stage 2 with --pretrain_embedding_path {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    runner = StageRunner(
        "stage1_pretrain",
        "configs/stage1_pretrain.yaml",
        description="Stage 1: Text + Visual Pretrain (Embedding)",
    )
    # ── Text pretrain flags ───────────────────────────────────────────
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--data_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--num_samples", type=int, default=None)
    runner.add_arg("--coco_ann_file", type=str, default=None,
                   help="COCO annotations for text pretrain category mixing and visual pretrain")
    runner.add_arg("--coco_grounding_ratio", type=float, default=None)
    runner.add_arg("--curriculum", action="store_true")
    runner.add_arg("--num_epochs", type=int, default=None)
    runner.add_arg("--learning_rate", type=float, default=None)
    runner.add_arg("--vit_lr", type=float, default=None,
                   help="LR for unfrozen ViT layers (very low)")
    runner.add_arg("--batch_size", type=int, default=None)
    runner.add_arg("--gradient_accumulation_steps", type=int, default=None)
    runner.add_arg("--max_length", type=int, default=None)
    runner.add_arg("--warmup_steps", type=int, default=None)
    runner.add_arg("--unfreeze_vit_layers", type=int, default=None,
                   help="Unfreeze last N ViT blocks + merger (0 = all frozen)")
    # ── Visual grounding pretrain flags ───────────────────────────────
    runner.add_arg("--visual_data_ratio", type=float, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--visual_num_box", type=int, default=None)
    runner.add_arg("--visual_num_point", type=int, default=None)
    runner.add_arg("--visual_epochs", type=int, default=None)
    runner.add_arg("--visual_learning_rate", type=float, default=None)
    runner.add_arg("--visual_batch_size", type=int, default=None)
    runner.add_arg("--visual_gradient_accumulation_steps", type=int, default=None)
    runner.add_arg("--max_seq_length", type=int, default=None)

    runner.run(train)
