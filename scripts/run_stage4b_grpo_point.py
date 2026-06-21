#!/usr/bin/env python3
"""Stage 4b: Specialized GRPO — Point Expert.

Continues training the Point Expert LoRA adapter with GRPO on point+maze data.
Uses Format RM + Accuracy RM with difficulty grading (Normal only).
"""

import os

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.training.grpo_runner import run_grpo_rounds
from src.training.grpo_utils import extract_completion_text
from src.data.generators.coco_box_generator import generate_coco_point_samples
from src.data.generators.path_tracing import generate_path_tracing_dataset
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.callbacks import maybe_compile_model
from src.training.memory_utils import log_memory_status
from src.training.stage_runner import StageRunner
from src.utils.difficulty import filter_normal_level_data
from src.utils.quality_rm_api import make_quality_reward_api_fn
from src.utils.reward.accuracy_rm import compute_total_reward, length_reward
from src.utils.reward.quality_rm import make_quality_reward_fn


def make_point_reward_fn(point_dist_threshold: float, tokenizer=None, logger=None):
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
                gt_curve = inputs[i].get("gt_curve")
            elif i < len(gt_texts):
                gt_text = gt_texts[i]
                task_type = task_types[i] if i < len(task_types) else "point"
                maze_grid = maze_grids[i] if i < len(maze_grids) else None
                gt_curve = None
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
                    gt_curve=gt_curve,
                )
                # Use raw total reward without difficulty-based collapsing to preserve
                # within-group variance. Add a gentle length penalty to differentiate
                # completions of different conciseness.
                comp_id = completion_ids_list[i] if i < len(completion_ids_list) else None
                comp_len = len(comp_id) if comp_id is not None else len(pred_text.split())
                target_len = 150 if task_type == "maze" else (120 if task_type == "path" else 80)
                length_r = length_reward(comp_len, target_length=target_len, max_penalty=0.1)
                rewards.append(total["total_reward"] + length_r)
            except Exception as e:
                logger.warning(f"Reward computation failed for sample {i}: {e}")
                rewards.append(0.0)
        return rewards

    return grpo_reward


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Load Point Expert from Stage 3b
    policy_path = args.model_path
    logger.info(f"Loading Point Expert from {policy_path}...")
    policy_model, processor = load_qlora_model(
        model_name=policy_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Policy loaded:")

    # Optional torch.compile (best-effort).
    policy_model = maybe_compile_model(policy_model, enable=args.compile_model)

    # 2. Generate or load cached GRPO data
    def _generate_raw_data():
        logger.info("Generating GRPO training data (point+maze+path)...")
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

        path_data = generate_path_tracing_dataset(
            n=args.num_path,
            seed=43,
            cache_dir=os.path.join(args.output_dir, "path_tracing_cache"),
        )
        # task_type is already "path" from generator; no override needed
        all_data.extend(path_data)
        logger.info(f"  Path tracing samples: {len(path_data)}")

        logger.info(f"Total GRPO samples: {len(all_data)}")
        return all_data

    all_data = runner.cached_data(
        os.path.join(args.output_dir, "train_data_cache.pkl"),
        _generate_raw_data,
    )

    num_rounds = args.num_rounds

    # Difficulty filtering: keep only Normal-level samples (paper Sec 2.5.2).
    if args.skip_difficulty_filter:
        logger.info("--skip_difficulty_filter is set; using all samples without filtering")
    else:
        def _filter_data():
            logger.info("Difficulty filtering: retaining only Normal-level samples...")
            return filter_normal_level_data(
                model=policy_model,
                processor=processor,
                data=all_data,
                num_generations=args.num_generations,
                max_completion_length=args.filter_max_completion_length,
                task_type="point",
                point_dist_threshold=args.filter_point_dist_threshold,
                batch_size=args.filter_batch_size,
                empty_cache_every=args.filter_empty_cache_every,
                reward_threshold=args.filter_reward_threshold,
                logger=logger,
            )

        all_data = runner.cached_data(
            os.path.join(args.output_dir, "filtered_train_data_cache.pkl"),
            _filter_data,
        )

    # Build quality RM factory (captures use_quality_rm_api flag in closure).
    quality_fn_factory = (
        make_quality_reward_api_fn if args.use_quality_rm_api
        else make_quality_reward_fn
    )

    # Reward factory: point-specific reward with maze support.
    def reward_fn_factory(threshold: float, **kwargs):
        return make_point_reward_fn(threshold, **kwargs)

    run_grpo_rounds(
        policy_model=policy_model,
        processor=processor,
        train_data=all_data,
        output_dir=args.output_dir,
        num_rounds=num_rounds,
        reward_fn_factory=reward_fn_factory,
        thresholds=[20.0, 10.0, 5.0],
        quality_fn_factory=quality_fn_factory,
        quality_task_type="point",
        args=args,
        logger=logger,
    )

    logger.info(f"Stage 4b complete. Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    runner = StageRunner(
        "stage4b_grpo_point",
        "configs/stage4b_grpo_point.yaml",
        description="Stage 4b: Point Expert GRPO",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_point", type=int, default=None)
    runner.add_arg("--num_maze", type=int, default=None)
    runner.add_arg("--num_path", type=int, default=None,
                   help="Number of path tracing samples")
    runner.add_arg("--num_rounds", type=int, default=None)
    runner.add_arg("--num_epochs", type=int, default=None)
    runner.add_arg("--learning_rate", type=float, default=None)
    runner.add_arg("--batch_size", type=int, default=None)
    runner.add_arg("--gradient_accumulation_steps", type=int, default=None)
    runner.add_arg("--lora_r", type=int, default=None)
    runner.add_arg("--lora_alpha", type=int, default=None)
    runner.add_arg("--logging_steps", type=int, default=None)
    runner.add_arg("--save_steps", type=int, default=None)
    runner.add_arg("--warmup_steps", type=int, default=None)
    runner.add_arg("--num_generations", type=int, default=None)
    runner.add_arg("--generation_batch_size", type=int, default=None,
                   help="Prompts per generation batch (must be divisible by num_generations). Defaults to num_generations.")
    runner.add_arg("--filter_batch_size", type=int, default=None,
                   help="Batch size for difficulty-filter generation (prompts per batch)")
    runner.add_arg("--filter_max_completion_length", type=int, default=None,
                   help="Max completion length used only during difficulty filtering")
    runner.add_arg("--filter_point_dist_threshold", type=float, default=None,
                   help="Point distance threshold used only during difficulty filtering")
    runner.add_arg("--filter_reward_threshold", type=float, default=None,
                   help="Reward threshold for correctness in difficulty filtering (overrides binary check)")
    runner.add_arg("--filter_empty_cache_every", type=int, default=None)
    runner.add_arg("--skip_difficulty_filter", action="store_true",
                   help="Skip difficulty filtering and use all generated samples")
    runner.add_arg("--max_completion_length", type=int, default=None)
    runner.add_arg("--beta", type=float, default=None)
    runner.add_arg("--temperature", type=float, default=None)
    runner.add_arg(
        "--use_quality_rm_api",
        action="store_true",
        help="Use OpenAI-compatible API for Quality RM (requires .env key).",
    )
    runner.add_arg(
        "--compile_model",
        action="store_true",
        help="Try torch.compile on the policy model (best-effort).",
    )
    runner.add_arg(
        "--early_stopping_subset_size",
        type=int,
        default=None,
        help="Validation subset size for early stopping (0 to disable).",
    )
    runner.add_arg(
        "--early_stopping_eval_steps",
        type=int,
        default=None,
        help="Evaluate validation subset every N steps.",
    )
    runner.add_arg(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Stop after this many evals without improvement.",
    )
    runner.run(train)
