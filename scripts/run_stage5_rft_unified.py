#!/usr/bin/env python3
"""Stage 5: Unified RFT — Experts as Generators, Unified as Learner.

Key design (corrected per paper):
  - EXPERTS generate rollouts (not Unified model)
  - Box Expert generates for box prompts
  - Point Expert generates for point/maze prompts
  - Difficulty grading: Easy/Normal/Hard → only Normal used
  - Unified model (re-init from merged Stage 2 base) SFTs on filtered data
"""

import os

# Mitigate CUDA memory fragmentation from variable-length expert rollouts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
    generate_coco_point_samples,
)
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.data.datasets.image_loader import load_image
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status, clear_memory
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_total_reward, extract_answer

logger = setup_logging(log_file="logs/stage5_rft_unified.log")


def generate_with_expert(expert_model, processor, sample, num_rollouts, max_new_tokens):
    """Expert generates N rollouts for a given sample."""
    image = load_image(sample["image"])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful visual reasoning assistant. Think step by step.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]
    gt_text = sample["reasoning"] + f"\n</think>\n\nThe answer is {sample.get('answer', '')}."

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(expert_model.device) for k, v in inputs.items()}

    rollouts = []
    for _ in range(num_rollouts):
        with torch.inference_mode():
            outputs = expert_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_len:]
        pred = processor.tokenizer.decode(new_tokens, skip_special_tokens=False)
        rollouts.append(pred)

    return rollouts, gt_text, sample.get("task_type", "box"), sample.get("maze_grid")


def difficulty_grading(rollouts, gt_text, task_type, maze_grid, iou_threshold, dist_threshold):
    """Grade rollouts as Easy/Normal/Hard based on reward scores."""
    scores = []
    for rollout in rollouts:
        try:
            total = compute_total_reward(
                pred_text=rollout,
                gt_text=gt_text,
                task_type=task_type,
                iou_threshold=iou_threshold,
                point_dist_threshold=dist_threshold,
                maze_grid=maze_grid,
            )
            scores.append(total["total_reward"])
        except Exception:
            scores.append(0.0)

    if not scores:
        return "hard", None, 0.0

    avg_score = sum(scores) / len(scores)
    # Difficulty classification
    correct = sum(1 for s in scores if s >= 1.5)  # All correct
    partial = sum(1 for s in scores if 0.5 <= s < 1.5)  # Partial
    wrong = sum(1 for s in scores if s < 0.5)  # All wrong

    if correct == len(scores):
        difficulty = "easy"
    elif wrong == len(scores):
        difficulty = "hard"
    else:
        difficulty = "normal"

    # Pick the best rollout for SFT data
    best_idx = scores.index(max(scores))
    return difficulty, rollouts[best_idx], avg_score


def main(args):
    logger.info("=" * 60)
    logger.info("Stage 5: Unified RFT (Experts as Generators)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load Unified model from merged Stage 2 base (fresh LoRA)
    logger.info(f"Loading Unified model from merged base: {args.model_path}")
    unified_model, processor = load_qlora_model(
        model_name=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Unified model loaded:")

    # 2. Generate or load cached prompts for rejection sampling
    cache_path = os.path.join(args.output_dir, "prompts_cache.pkl")
    if os.path.exists(cache_path):
        logger.info(f"Loading cached prompts from {cache_path}")
        with open(cache_path, "rb") as f:
            all_prompts = pickle.load(f)
        logger.info(f"  Loaded {len(all_prompts)} prompts from cache")
    else:
        logger.info("Generating prompts for rejection sampling...")
        all_prompts = []

        box_prompts = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_box_prompts,
        )
        for d in box_prompts:
            d["task_type"] = "box"
        all_prompts.extend(box_prompts)
        logger.info(f"  Box prompts: {len(box_prompts)}")

        counting_prompts = generate_coco_counting_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_counting_prompts,
        )
        for d in counting_prompts:
            d["task_type"] = "box"
        all_prompts.extend(counting_prompts)
        logger.info(f"  Counting prompts: {len(counting_prompts)}")

        clevr_prompts = generate_clevr_spatial_dataset(
            n=args.num_clevr_prompts,
            seed=47,
            cache_dir=os.path.join(args.output_dir, "clevr_prompt_cache"),
        )
        for d in clevr_prompts:
            d["task_type"] = "box"
        all_prompts.extend(clevr_prompts)
        logger.info(f"  CLEVR prompts: {len(clevr_prompts)}")

        point_prompts = generate_coco_point_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_point_prompts,
        )
        for d in point_prompts:
            d["task_type"] = "point"
        all_prompts.extend(point_prompts)
        logger.info(f"  Point prompts: {len(point_prompts)}")

        maze_prompts = generate_maze_dataset(
            n=args.num_maze_prompts,
            seed=45,
        )
        for d in maze_prompts:
            d["task_type"] = "maze"
        all_prompts.extend(maze_prompts)
        logger.info(f"  Maze prompts: {len(maze_prompts)}")

        random.shuffle(all_prompts)
        logger.info(f"Total prompts: {len(all_prompts)}")

        # Save cache for future runs
        os.makedirs(args.output_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(all_prompts, f)
        logger.info(f"Cached prompts to {cache_path}")

    # 3. Load Box Expert (teacher)
    logger.info(f"Loading Box Expert from {args.box_expert_path}...")
    box_expert, _ = load_qlora_model(
        model_name=args.box_expert_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # 4. Load Point Expert (teacher)
    logger.info(f"Loading Point Expert from {args.point_expert_path}...")
    point_expert, _ = load_qlora_model(
        model_name=args.point_expert_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # 5. Expert generation + difficulty grading
    logger.info("Running expert generation with difficulty grading...")
    filtered_data = []
    easy_samples = []
    hard_count = 0

    for i, sample in enumerate(all_prompts):
        task_type = sample.get("task_type", "box")

        # Select expert based on task type
        if task_type == "box":
            expert = box_expert
        else:
            expert = point_expert

        rollouts, gt_text, _, maze_grid = generate_with_expert(
            expert, processor, sample,
            args.num_rollouts, args.max_new_tokens,
        )

        difficulty, best_rollout, avg_score = difficulty_grading(
            rollouts, gt_text, sample.get("task_type", "box"),
            sample.get("maze_grid"),
            args.iou_threshold, args.point_dist_threshold,
        )

        record = {
            "image": sample["image"],
            "prompt": sample["prompt"],
            "reasoning": best_rollout or sample["reasoning"],
            "answer": extract_answer(best_rollout) or sample["answer"],
            "task_type": sample.get("task_type", "box"),
        }

        if difficulty == "normal":
            filtered_data.append(record)
        elif difficulty == "easy":
            easy_samples.append(record)
        else:
            hard_count += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"  {i + 1}/{len(all_prompts)}: "
                f"{len(filtered_data)} normal kept, {len(easy_samples)} easy, {hard_count} hard"
            )

    # Retain all Normal + 5% Easy to mitigate catastrophic forgetting (paper Sec 2.5.3)
    if easy_samples:
        random.shuffle(easy_samples)
        retained_easy_count = max(1, int(len(easy_samples) * 0.05))
        filtered_data.extend(easy_samples[:retained_easy_count])
        logger.info(
            f"Retained {retained_easy_count} Easy samples ({0.05*100:.0f}%) alongside Normal data"
        )

    logger.info(
        f"Difficulty grading done: {len(filtered_data)} total kept "
        f"(Normal + 5% Easy), {len(easy_samples)} Easy total, {hard_count} Hard skipped"
    )

    # Release expert models before SFT to free VRAM for Unified model training.
    logger.info("Releasing expert models before Unified SFT training...")
    del box_expert
    del point_expert
    gc.collect()
    clear_memory()
    log_memory_status("Expert models released:")

    if len(filtered_data) < 100:
        logger.warning("Too few Normal samples — consider adjusting thresholds")
        return

    # 6. SFT Unified model on Normal-difficulty expert data
    logger.info("Training Unified model on Normal-difficulty expert data...")
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt and not os.path.isdir(resume_ckpt):
        logger.error(f"Checkpoint not found: {resume_ckpt}")
        sys.exit(1)

    trainer = create_sft_trainer(
        model=unified_model,
        processor=processor,
        train_data=filtered_data,
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

    trainer.train(resume_from_checkpoint=resume_ckpt)
    final_dir = f"{args.output_dir}/final_model"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    logger.info(f"Stage 5 complete. Final model saved to {final_dir}")
    log_memory_status("Stage 5 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Unified RFT")
    parser.add_argument("--model_path", type=str, default="outputs/stage2_merged_base")
    parser.add_argument("--output_dir", type=str, default="outputs/stage5_rft_unified")
    parser.add_argument("--box_expert_path", type=str, default="outputs/stage4a_grpo_box")
    parser.add_argument("--point_expert_path", type=str, default="outputs/stage4b_grpo_point")
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_box_prompts", type=int, default=4000)
    parser.add_argument("--num_counting_prompts", type=int, default=3000,
                        help="Number of counting prompts for rejection sampling")
    parser.add_argument("--num_clevr_prompts", type=int, default=2000,
                        help="Number of CLEVR spatial/VQA prompts for rejection sampling")
    parser.add_argument("--num_point_prompts", type=int, default=3000)
    parser.add_argument("--num_maze_prompts", type=int, default=3000)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--point_dist_threshold", type=float, default=10.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume SFT training, e.g. outputs/stage5_rft_unified/checkpoint-500")
    args = parser.parse_args()
    main(args)
