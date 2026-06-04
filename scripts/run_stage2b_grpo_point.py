#!/usr/bin/env python3
"""Stage 2b: Specialized GRPO — Point Expert.

Continues training the Point Expert LoRA adapter with GRPO on point+maze data.
Uses Format RM + Accuracy RM with difficulty grading (Normal only).
"""

import argparse
import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trl import GRPOConfig, GRPOTrainer

from src.data.datasets.grpo_dataset import GRPODataset
from src.data.generators.coco_box_generator import generate_coco_point_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_total_reward

logger = setup_logging(log_file="logs/stage2b_grpo_point.log")


def make_point_reward_fn(point_dist_threshold: float):
    """Factory: point+maze reward with Format RM + Point/Maze Accuracy RM."""

    def grpo_reward(completions, prompts=None, **kwargs):
        inputs = kwargs.get("inputs", [])
        rewards = []
        for i, completion in enumerate(completions):
            if i >= len(inputs):
                rewards.append(0.0)
                continue
            inp = inputs[i]
            try:
                total = compute_total_reward(
                    pred_text=completion,
                    gt_text=inp["gt_text"],
                    task_type=inp.get("task_type", "point"),
                    point_dist_threshold=point_dist_threshold,
                    maze_grid=inp.get("maze_grid"),
                )
                if total["difficulty"] == "normal":
                    rewards.append(total["total_reward"])
                else:
                    rewards.append(0.1 if total["difficulty"] == "easy" else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    return grpo_reward


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 2b: Specialized GRPO — Point Expert")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load Point Expert from Stage 1b
    policy_path = args.model_path
    logger.info(f"Loading Point Expert from {policy_path}...")
    policy_model, processor = load_qlora_model(
        model_name=policy_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Policy loaded:")

    # 2. Generate GRPO data (broader pool)
    logger.info("Generating GRPO training data (point+maze)...")
    all_data = []

    point_data = generate_coco_point_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_point,
    )
    for d in point_data:
        d["task_type"] = "point"
    all_data.extend(point_data)
    logger.info(f"  Point samples: {len(point_data)}")

    maze_data = generate_maze_dataset(
        n=args.num_maze,
        seed=42,
    )
    for d in maze_data:
        d["task_type"] = "maze"
    all_data.extend(maze_data)
    logger.info(f"  Maze samples: {len(maze_data)}")

    logger.info(f"Total GRPO samples: {len(all_data)}")

    num_rounds = args.num_rounds
    dist_thresholds = [20.0, 10.0, 5.0]

    for round_idx in range(num_rounds):
        dist_th = dist_thresholds[round_idx] if round_idx < len(dist_thresholds) else 5.0
        round_dir = Path(args.output_dir) / f"round_{round_idx + 1}"
        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info(f"GRPO Round {round_idx + 1}/{num_rounds} (Dist threshold: {dist_th})")
        logger.info(f"{'='*60}")

        reward_fn = make_point_reward_fn(dist_th)

        grpo_config = GRPOConfig(
            output_dir=str(round_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=2,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            num_generations=args.num_generations,
            generation_batch_size=args.num_generations,
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            temperature=args.temperature,
            scale_rewards="group",
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
        trainer.save_model(str(round_dir))
        processor.save_pretrained(str(round_dir))

        log_memory_status(f"Round {round_idx + 1} complete:")

        try:
            policy_model, processor = load_qlora_model(
                model_name=str(round_dir),
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        except Exception as e:
            logger.warning(f"Could not reload: {e}, continuing")

    logger.info(f"Stage 2b complete. Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2b: Point Expert GRPO")
    parser.add_argument("--model_path", type=str, default="outputs/stage1b_sft_point")
    parser.add_argument("--output_dir", type=str, default="outputs/stage2b_grpo_point")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_point", type=int, default=2000)
    parser.add_argument("--num_maze", type=int, default=5000)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=5)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
