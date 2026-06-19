#!/usr/bin/env python3
"""Stage 4a: Specialized GRPO — Box Expert.

Continues training the Box Expert LoRA adapter with GRPO on box-only data.
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
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
)
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.memory_utils import log_memory_status
from src.training.callbacks import maybe_compile_model
from src.training.stage_runner import StageRunner
from src.utils.difficulty import filter_normal_level_data
from src.utils.reward.accuracy_rm import compute_total_reward, length_reward
from src.utils.reward.quality_rm import make_quality_reward_fn
from src.utils.quality_rm_api import make_quality_reward_api_fn

logger = None  # Set by train() from runner.logger


def make_box_reward_fn(iou_threshold: float, tokenizer=None):
    """Factory: box-only reward with Format RM + Box Accuracy RM + difficulty grading."""

    def grpo_reward(completions, prompts=None, **kwargs):
        # Support both training format (inputs=dict list) and test format (separate kwargs)
        inputs = kwargs.get("inputs", [])
        gt_texts = kwargs.get("gt_text", [])
        completion_ids_list = kwargs.get("completion_ids", [])

        rewards = []
        for i, completion in enumerate(completions):
            # Get gt_text from either format
            if i < len(inputs):
                gt_text = inputs[i].get("gt_text", "")
            elif i < len(gt_texts):
                gt_text = gt_texts[i]
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
                    task_type="box",
                    iou_threshold=iou_threshold,
                )
                # Use raw total reward without difficulty-based collapsing to preserve
                # within-group variance. Add a gentle length penalty to differentiate
                # completions of different conciseness.
                comp_id = completion_ids_list[i] if i < len(completion_ids_list) else None
                comp_len = len(comp_id) if comp_id is not None else len(pred_text.split())
                # Box completions often need >120 tokens (multiple boxes/counting).
                # Use a more permissive target and a small penalty to avoid
                # punishing valid long chains.
                length_r = length_reward(comp_len, target_length=240, max_penalty=0.05)
                rewards.append(total["total_reward"] + length_r)
            except Exception as e:
                logger.warning(f"Reward computation failed for sample {i}: {e}")
                rewards.append(0.0)
        return rewards

    return grpo_reward


def train(runner: StageRunner) -> None:
    global logger
    args, logger = runner.args, runner.logger

    # 1. Load Box Expert from Stage 3a
    policy_path = args.model_path
    logger.info(f"Loading Box Expert from {policy_path}...")
    policy_model, processor = load_qlora_model(
        model_name=policy_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Policy loaded:")

    # Optional torch.compile (best-effort).
    policy_model = maybe_compile_model(policy_model, enable=args.compile_model)

    # 2. Generate or load cached GRPO data
    def _generate_data():
        logger.info("Generating GRPO training data (box + counting + spatial/VQA)...")
        data = []

        box_data = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_samples,
        )
        for d in box_data:
            d["task_type"] = "box"
        data.extend(box_data)
        logger.info(f"  Box localization samples: {len(box_data)}")

        counting_data = generate_coco_counting_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_counting,
        )
        for d in counting_data:
            d["task_type"] = "box"
        data.extend(counting_data)
        logger.info(f"  Coarse-grained counting samples: {len(counting_data)}")

        clevr_data = generate_clevr_spatial_dataset(
            n=args.num_clevr,
            seed=44,
            cache_dir=os.path.join(args.output_dir, "clevr_cache"),
        )
        for d in clevr_data:
            d["task_type"] = "box"
        data.extend(clevr_data)
        logger.info(f"  CLEVR spatial/VQA samples: {len(clevr_data)}")

        logger.info(f"Total GRPO samples: {len(data)}")
        return data

    cache_path = os.path.join(args.output_dir, "train_data_cache.pkl")
    all_data = runner.cached_data(cache_path, _generate_data)

    num_rounds = args.num_rounds

    # Difficulty filtering: keep only Normal-level samples (paper Sec 2.5.2).
    filtered_cache_path = os.path.join(args.output_dir, "filtered_train_data_cache.pkl")
    if args.skip_difficulty_filter:
        logger.info("--skip_difficulty_filter is set; using all samples without filtering")
    else:
        all_data = runner.cached_data(filtered_cache_path, lambda: filter_normal_level_data(
            model=policy_model,
            processor=processor,
            data=all_data,
            num_generations=args.num_generations,
            max_completion_length=args.filter_max_completion_length,
            task_type="box",
            iou_threshold=0.3 if num_rounds > 0 else 0.5,
            batch_size=args.filter_batch_size,
            empty_cache_every=args.filter_empty_cache_every,
            logger=logger,
        ))

    # Build quality RM factory (captures use_quality_rm_api flag in closure).
    quality_fn_factory = (
        make_quality_reward_api_fn if args.use_quality_rm_api
        else make_quality_reward_fn
    )

    # Reward factory: box-specific reward with accuracy + length penalty.
    def reward_fn_factory(threshold: float, **kwargs):
        return make_box_reward_fn(threshold, **kwargs)

    run_grpo_rounds(
        policy_model=policy_model,
        processor=processor,
        train_data=all_data,
        output_dir=args.output_dir,
        num_rounds=num_rounds,
        reward_fn_factory=reward_fn_factory,
        thresholds=[0.3, 0.5, 0.7],
        quality_fn_factory=quality_fn_factory,
        quality_task_type="box",
        args=args,
        logger=logger,
    )

    logger.info(f"Stage 4a complete. Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    runner = StageRunner(
        "stage4a_grpo_box",
        "configs/stage4a_grpo_box.yaml",
        description="Stage 4a: Box Expert GRPO",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_samples", type=int, default=None)
    runner.add_arg("--num_counting", type=int, default=None,
                   help="Number of coarse-grained counting samples")
    runner.add_arg("--num_clevr", type=int, default=None,
                   help="Number of CLEVR-style spatial/VQA samples")
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
    runner.add_arg("--filter_batch_size", type=int, default=None,
                   help="Batch size for difficulty-filter generation (prompts per batch)")
    runner.add_arg("--filter_max_completion_length", type=int, default=None,
                   help="Max completion length used only during difficulty filtering")
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
    runner.add_arg(
        "--no_console_log",
        action="store_true",
        help="Only log to file; suppress console output to avoid Terminal crashes.",
    )
    runner.add_arg(
        "--disable_tqdm",
        action="store_true",
        help="Disable progress bars to reduce console output.",
    )
    runner.run(train)
