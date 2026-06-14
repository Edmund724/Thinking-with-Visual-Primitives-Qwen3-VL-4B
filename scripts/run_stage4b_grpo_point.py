#!/usr/bin/env python3
"""Stage 4b: Specialized GRPO — Point Expert.

Continues training the Point Expert LoRA adapter with GRPO on point+maze data.
Uses Format RM + Accuracy RM with difficulty grading (Normal only).
"""

import os

# Mitigate CUDA memory fragmentation from variable-length GRPO completions.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trl import GRPOConfig, GRPOTrainer

from src.data.datasets.grpo_dataset import GRPODataset
from src.training.grpo_fixes import apply_grpo_fixes
from src.training.grpo_utils import extract_completion_text
from src.data.generators.coco_box_generator import generate_coco_point_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model, _set_use_cache_deep
from src.training.memory_utils import log_memory_status, clear_memory, GPUMemoryMonitor
from src.utils.constants import GPU_MEMORY_WARNING_GB
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_total_reward, length_reward

logger = setup_logging(log_file="logs/stage4b_grpo_point.log")


def _latest_checkpoint(round_dir: Path) -> Path | None:
    """Return the latest checkpoint-* directory inside a round dir, or None."""
    checkpoints = [p for p in round_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None

    def _step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return -1

    return max(checkpoints, key=_step)


def make_point_reward_fn(point_dist_threshold: float, tokenizer=None):
    """Factory: point+maze reward with Format RM + Point/Maze Accuracy RM."""

    def grpo_reward(completions, prompts=None, **kwargs):
        # Support both training format (inputs=dict list) and test format (separate kwargs)
        inputs = kwargs.get("inputs", [])
        gt_texts = kwargs.get("gt_text", [])
        task_types = kwargs.get("task_type", [])
        maze_grids = kwargs.get("maze_grid", [])
        completion_ids_list = kwargs.get("completion_ids", [])

        rewards = []
        for i, completion in enumerate(completions):
            # Get gt_text and task_type from either format
            if i < len(inputs):
                gt_text = inputs[i].get("gt_text", "")
                task_type = inputs[i].get("task_type", "point")
                maze_grid = inputs[i].get("maze_grid")
            elif i < len(gt_texts):
                gt_text = gt_texts[i]
                task_type = task_types[i] if i < len(task_types) else "point"
                maze_grid = maze_grids[i] if i < len(maze_grids) else None
            else:
                rewards.append(0.0)
                continue

            # Extract completion text: prefer re-decoding from IDs (preserves special tokens)
            comp_id = completion_ids_list[i] if i < len(completion_ids_list) else None
            pred_text = extract_completion_text(
                completion, tokenizer=tokenizer, completion_id=comp_id
            )

            try:
                total = compute_total_reward(
                    pred_text=pred_text,
                    gt_text=gt_text,
                    task_type=task_type,
                    point_dist_threshold=point_dist_threshold,
                    maze_grid=maze_grid,
                )
                # Use raw total reward without difficulty-based collapsing to preserve
                # within-group variance. Add a gentle length penalty to differentiate
                # completions of different conciseness.
                comp_id = completion_ids_list[i] if i < len(completion_ids_list) else None
                comp_len = len(comp_id) if comp_id is not None else len(pred_text.split())
                target_len = 150 if task_type == "maze" else 80
                length_r = length_reward(comp_len, target_length=target_len, max_penalty=0.1)
                rewards.append(total["total_reward"] + length_r)
            except Exception as e:
                logger.warning(f"Reward computation failed for sample {i}: {e}")
                rewards.append(0.0)
        return rewards

    return grpo_reward


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 4b: Specialized GRPO — Point Expert")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load Point Expert from Stage 3b
    policy_path = args.model_path
    logger.info(f"Loading Point Expert from {policy_path}...")
    policy_model, processor = load_qlora_model(
        model_name=policy_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Policy loaded:")

    # 2. Generate or load cached GRPO data
    cache_path = os.path.join(args.output_dir, "train_data_cache.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached training data from {cache_path}")
        with open(cache_path, "rb") as f:
            all_data = pickle.load(f)
        logger.info(f"  Loaded {len(all_data)} samples from cache")
    else:
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

        # Save cache for future runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Cached training data to {cache_path}")

    num_rounds = args.num_rounds
    dist_thresholds = [20.0, 10.0, 5.0]

    # Apply monkey-patches once, before training. Applying inside the round loop
    # would nest wrappers on each iteration.
    apply_grpo_fixes(GRPOTrainer)

    for round_idx in range(num_rounds):
        dist_th = dist_thresholds[round_idx] if round_idx < len(dist_thresholds) else 5.0
        round_dir = Path(args.output_dir) / f"round_{round_idx + 1}"

        # Skip already-completed rounds
        round_adapter = round_dir / "adapter_model.safetensors"
        if round_adapter.exists():
            logger.info(f"Round {round_idx + 1}/{num_rounds} already done ({round_adapter}), skipping.")
            try:
                policy_model, processor = load_qlora_model(
                    model_name=str(round_dir),
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                )
            except Exception as e:
                logger.warning(f"Could not reload round {round_idx + 1}: {e}")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"{'='*60}")
        logger.info(f"GRPO Round {round_idx + 1}/{num_rounds} (Dist threshold: {dist_th})")
        logger.info(f"{'='*60}")

        # If this round was interrupted, resume from the latest checkpoint-*
        # instead of restarting the whole round.
        resume_from = _latest_checkpoint(round_dir)
        if resume_from is not None:
            logger.info(
                f"Found checkpoint {resume_from.name} for round {round_idx + 1}, resuming."
            )
            try:
                policy_model, processor = load_qlora_model(
                    model_name=str(resume_from),
                    lora_r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                )
                log_memory_status(f"Loaded checkpoint {resume_from.name}:")
            except Exception as e:
                logger.warning(f"Could not load checkpoint {resume_from}: {e}, starting from scratch.")
                resume_from = None

        reward_fn = make_point_reward_fn(dist_th, tokenizer=processor.tokenizer)

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
            generation_batch_size=args.batch_size * args.num_generations,
            max_completion_length=args.max_completion_length,
            beta=args.beta,
            temperature=args.temperature,
            scale_rewards="group",
        )

        dataset = GRPODataset(all_data)

        # use_cache is incompatible with gradient checkpointing; disable on all nested configs
        _set_use_cache_deep(policy_model)

        # Memory monitor threshold: 85% of total VRAM (5090D = ~27 GB) so we
        # clear cache before fragmentation pushes us into OOM territory.
        total_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        mem_threshold_gb = max(GPU_MEMORY_WARNING_GB, total_vram_gb * 0.85)

        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=[reward_fn],
            args=grpo_config,
            train_dataset=dataset,
            processing_class=processor,
            callbacks=[GPUMemoryMonitor(clear_threshold_gb=mem_threshold_gb)],
        )

        logger.info("Training GRPO...")
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
        trainer.save_model(str(round_dir))
        processor.save_pretrained(str(round_dir))

        log_memory_status(f"Round {round_idx + 1} complete:")

        # Explicitly free the trainer + old model before reloading to avoid
        # carrying fragmented/accumulated memory into the next round.
        del trainer
        del policy_model
        gc.collect()
        clear_memory()

        try:
            policy_model, processor = load_qlora_model(
                model_name=str(round_dir),
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        except Exception as e:
            logger.warning(f"Could not reload: {e}, continuing")

    logger.info(f"Stage 4b complete. Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4b: Point Expert GRPO")
    parser.add_argument("--model_path", type=str, default="outputs/stage3b_sft_point")
    parser.add_argument("--output_dir", type=str, default="outputs/stage4b_grpo_point")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_point", type=int, default=2000)
    parser.add_argument("--num_maze", type=int, default=5000)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--max_completion_length", type=int, default=768)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
