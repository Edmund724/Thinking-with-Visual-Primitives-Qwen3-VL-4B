#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Stage 2: GRPO — Group Relative Policy Optimization for visual primitives.

Uses TRL's GRPOTrainer with multimodal support.
3 rounds with tightening reward thresholds.
Reference model = frozen Stage 1 SFT checkpoint.
"""

import argparse
import logging
import yaml
from pathlib import Path

import torch
from trl import GRPOConfig, GRPOTrainer

from src.data.datasets.grpo_dataset import GRPODataset
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.data.generators.synthetic_path import generate_path_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging
from src.utils.metrics import process_reward

logger = setup_logging(log_file="logs/stage2_grpo.log")


def make_reward_fn(iou_threshold: float, point_dist_threshold: float):
    """Factory: create a reward function with given thresholds."""

    def grpo_reward(completions, prompts=None, **kwargs):
        inputs = kwargs.get("inputs", [])
        rewards = []
        for i, completion in enumerate(completions):
            if i >= len(inputs):
                rewards.append(0.0)
                continue
            inp = inputs[i]
            task_type = inp.get("task_type", "box")
            r = process_reward(
                pred_text=completion,
                gt_text=inp["gt_text"],
                task_type=task_type,
                iou_threshold=iou_threshold,
                point_dist_threshold=point_dist_threshold,
                maze_grid=inp.get("maze_grid"),
            )
            score = 0.0
            if r["answer_correct"]:
                score += 1.0
                if r["syntax_valid"]:
                    score += 0.2
                if task_type == "box":
                    score += r.get("box_avg_iou", 0.0)
                elif task_type in ("point", "maze"):
                    dist = r.get("point_avg_dist", float("inf"))
                    score += max(0, 1.0 - min(dist, 100.0) / 100.0)
                if task_type == "maze":
                    if r.get("wall_collision_count", 0) == 0:
                        score += 0.3
                    if not r.get("backtracking_missing", False):
                        score += 0.1
            rewards.append(score)
        return rewards

    return grpo_reward


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Stage 2: GRPO (Group Relative Policy Optimization)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load policy model from Stage 1
    policy_path = args.model_path or config["stage1_checkpoint"]
    logger.info(f"Loading policy model from {policy_path}...")
    policy_model, processor = load_qlora_model(
        model_name=policy_path,
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
    )
    # 2. GRPOTrainer with PEFT handles reference internally (adapter disable)
    log_memory_status("Policy loaded:")

    # 2. Generate shared GRPO data (mixed box + maze + path)
    logger.info("Generating GRPO training data...")
    all_data = []

    coco_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_samples or config.get("num_samples_per_round", 5000),
    )
    for d in coco_data:
        d["task_type"] = "box"
    all_data.extend(coco_data)
    logger.info(f"  COCO: {len(coco_data)}")

    maze_data = generate_maze_dataset(
        n=args.num_maze or max(1000, (args.num_samples or 5000) // 2),
        seed=42,
    )
    for d in maze_data:
        d["task_type"] = "maze"
    all_data.extend(maze_data)
    logger.info(f"  Maze: {len(maze_data)}")

    path_data = generate_path_dataset(
        n=args.num_path or max(500, (args.num_samples or 5000) // 5),
        seed=42,
    )
    for d in path_data:
        d["task_type"] = "point"
    all_data.extend(path_data)
    logger.info(f"  Path: {len(path_data)}")

    logger.info(f"Total GRPO samples: {len(all_data)}")

    num_rounds = args.num_rounds or config.get("num_rounds", 3)
    iou_thresholds = config.get("iou_thresholds", [0.3, 0.5, 0.7])
    point_dist_thresholds = config.get("point_dist_thresholds", [20.0, 10.0, 5.0])

    for round_idx in range(num_rounds):
        iou_th = iou_thresholds[round_idx] if round_idx < len(iou_thresholds) else 0.7
        dist_th = point_dist_thresholds[round_idx] if round_idx < len(point_dist_thresholds) else 5.0

        round_dir = f"{config['output_dir']}/round_{round_idx + 1}"
        Path(round_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info(f"GRPO Round {round_idx + 1}/{num_rounds}")
        logger.info(f"  IoU threshold: {iou_th}, Point dist threshold: {dist_th}")
        logger.info(f"{'='*60}")

        reward_fn = make_reward_fn(iou_th, dist_th)

        grpo_config = GRPOConfig(
            output_dir=round_dir,
            num_train_epochs=config.get("num_train_epochs", 1),
            per_device_train_batch_size=config.get("per_device_batch_size", 1),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 1e-6),
            warmup_steps=config.get("warmup_steps", 50),
            logging_steps=config.get("logging_steps", 5),
            save_steps=config.get("save_steps", 200),
            save_total_limit=config.get("save_total_limit", 2),
            bf16=True,
            optim=config.get("optimizer", "paged_adamw_8bit"),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to=config.get("report_to", "none"),
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            num_generations=config.get("num_generations", 5),
            generation_batch_size=config.get("num_generations", 5),
            max_completion_length=args.max_completion_length or config.get("max_completion_length", 1024),
            beta=config.get("beta", 0.04),
            temperature=config.get("temperature", 1.0),
            scale_rewards=config.get("scale_rewards", "group"),
        )

        dataset = GRPODataset(all_data)

        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=dataset,
            processing_class=processor,
        )

        logger.info("Training GRPO...")
        trainer.train()
        trainer.save_model(round_dir)
        processor.save_pretrained(round_dir)

        log_memory_status(f"Round {round_idx + 1} complete:")

        # Reload policy model from this round's output for next round
        try:
            policy_model, processor = load_qlora_model(
                model_name=round_dir,
                lora_r=config.get("lora_r", 64),
                lora_alpha=config.get("lora_alpha", 128),
            )
        except Exception as e:
            logger.warning(f"Could not load round checkpoint: {e}, continuing with current model")

    logger.info(f"Stage 2 complete. Checkpoints in {config['output_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: GRPO")
    parser.add_argument("--config", type=str, default="configs/stage2_grpo.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to policy model (default: stage1_checkpoint from config)")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of GRPO training samples")
    parser.add_argument("--num_maze", type=int, default=None)
    parser.add_argument("--num_path", type=int, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    main(parser.parse_args())
