#!/usr/bin/env python3
"""Stage 4a: Specialized GRPO — Box Expert.

Continues training the Box Expert LoRA adapter with GRPO on box-only data.
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
from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
)
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model, _set_use_cache_deep
from src.training.memory_utils import log_memory_status, clear_memory, GPUMemoryMonitor
from src.utils.constants import GPU_MEMORY_WARNING_GB
from src.utils.logging_utils import setup_logging
from src.utils.metrics import (
    compute_total_reward,
    filter_normal_level_data,
    length_reward,
    make_quality_reward_fn,
)

logger = logging.getLogger("stage4a_grpo_box")


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
                length_r = length_reward(comp_len, target_length=120, max_penalty=0.1)
                rewards.append(total["total_reward"] + length_r)
            except Exception as e:
                logger.warning(f"Reward computation failed for sample {i}: {e}")
                rewards.append(0.0)
        return rewards

    return grpo_reward


def main(args):
    setup_logging(
        log_file="logs/stage4a_grpo_box.log", console=not args.no_console_log
    )

    logger.info("=" * 60)
    logger.info("Stage 4a: Specialized GRPO — Box Expert")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load Box Expert from Stage 3a
    policy_path = args.model_path
    logger.info(f"Loading Box Expert from {policy_path}...")
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
        logger.info("Generating GRPO training data (box + counting + spatial/VQA)...")
        all_data = []

        box_data = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_samples,
        )
        for d in box_data:
            d["task_type"] = "box"
        all_data.extend(box_data)
        logger.info(f"  Box localization samples: {len(box_data)}")

        counting_data = generate_coco_counting_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_counting,
        )
        for d in counting_data:
            d["task_type"] = "box"
        all_data.extend(counting_data)
        logger.info(f"  Coarse-grained counting samples: {len(counting_data)}")

        clevr_data = generate_clevr_spatial_dataset(
            n=args.num_clevr,
            seed=44,
            cache_dir=os.path.join(args.output_dir, "clevr_cache"),
        )
        for d in clevr_data:
            d["task_type"] = "box"
        all_data.extend(clevr_data)
        logger.info(f"  CLEVR spatial/VQA samples: {len(clevr_data)}")

        logger.info(f"Total GRPO samples: {len(all_data)}")

        # Save cache for future runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Cached training data to {cache_path}")

    num_rounds = args.num_rounds
    iou_thresholds = [0.3, 0.5, 0.7]

    # Difficulty filtering: keep only Normal-level samples (paper Sec 2.5.2).
    filtered_cache_path = os.path.join(args.output_dir, "filtered_train_data_cache.pkl")
    if os.path.exists(filtered_cache_path):
        logger.info(f"Loading filtered training data from {filtered_cache_path}")
        with open(filtered_cache_path, "rb") as f:
            all_data = pickle.load(f)
        logger.info(f"  Loaded {len(all_data)} Normal-difficulty samples")
    else:
        logger.info("Difficulty filtering: retaining only Normal-level samples...")
        all_data = filter_normal_level_data(
            model=policy_model,
            processor=processor,
            data=all_data,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            task_type="box",
            iou_threshold=iou_thresholds[0] if num_rounds > 0 else 0.5,
            logger=logger,
        )
        with open(filtered_cache_path, "wb") as f:
            pickle.dump(all_data, f)
        logger.info(f"Cached filtered training data to {filtered_cache_path}")

    # Apply monkey-patches once, before training. Applying inside the round loop
    # would nest wrappers on each iteration.
    apply_grpo_fixes(GRPOTrainer)

    for round_idx in range(num_rounds):
        iou_th = iou_thresholds[round_idx] if round_idx < len(iou_thresholds) else 0.7
        round_dir = Path(args.output_dir) / f"round_{round_idx + 1}"

        # Skip already-completed rounds
        round_adapter = round_dir / "adapter_model.safetensors"
        if round_adapter.exists():
            logger.info(f"Round {round_idx + 1}/{num_rounds} already done ({round_adapter}), skipping.")
            # Reload for next round
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
        logger.info(f"GRPO Round {round_idx + 1}/{num_rounds} (IoU threshold: {iou_th})")
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

        reward_fn = make_box_reward_fn(iou_th, tokenizer=processor.tokenizer)

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
            disable_tqdm=args.disable_tqdm,
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

        quality_fn = make_quality_reward_fn(
            tokenizer=processor.tokenizer, task_type_default="box"
        )

        trainer = GRPOTrainer(
            model=policy_model,
            reward_funcs=[reward_fn, quality_fn],
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

        # Reload for next round
        try:
            policy_model, processor = load_qlora_model(
                model_name=str(round_dir),
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
        except Exception as e:
            logger.warning(f"Could not reload: {e}, continuing")

    logger.info(f"Stage 4a complete. Checkpoints in {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4a: Box Expert GRPO")
    parser.add_argument("--model_path", type=str, default="outputs/stage3a_sft_box")
    parser.add_argument("--output_dir", type=str, default="outputs/stage4a_grpo_box")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--num_counting", type=int, default=3000,
                        help="Number of coarse-grained counting samples")
    parser.add_argument("--num_clevr", type=int, default=2000,
                        help="Number of CLEVR-style spatial/VQA samples")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=6)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--no_console_log",
        action="store_true",
        help="Only log to file; suppress console output to avoid Terminal crashes.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Disable progress bars to reduce console output.",
    )
    args = parser.parse_args()
    main(args)
