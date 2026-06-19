#!/usr/bin/env python3
"""Stage 5: Unified RFT — Experts as Generators, Unified as Learner.

Key design (corrected per paper):
  - EXPERTS generate rollouts (not Unified model)
  - Box Expert generates for box prompts
  - Point Expert generates for point/maze prompts
  - Difficulty grading: Easy/Normal/Hard → only Normal used
  - Unified model (re-init from merged Stage 2 base) SFTs on filtered data
"""

import gc
import os
import random
import sys

import torch

import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from src.training.stage_runner import StageRunner
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
from src.utils.reward.accuracy_rm import compute_total_reward
from src.models.visual_primitive_parser import PrimitiveParser
from src.utils.conversation_builder import ConversationBuilder


def generate_with_expert(expert_model, processor, sample, num_rollouts, max_new_tokens):
    """Expert generates N rollouts for a given sample."""
    image = load_image(sample["image"])
    messages = ConversationBuilder("opd").build_prompt(sample["prompt"], image)
    gt_text = ConversationBuilder.build_gt_text(
        sample["reasoning"], sample.get("answer", "")
    )

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
    """Grade rollouts as Easy/Normal/Hard based on the number of correct rollouts.

    Matches the paper's Specialized RL / Unified RFT data selection (Sec 2.5.2
    and 2.5.3): a rollout is considered correct only when its final answer is
    correct AND its output satisfies basic syntax constraints. We then classify
    the prompt by the count of correct rollouts among the N generated samples.
    """
    from src.utils.difficulty import is_rollout_correct

    correct_flags = []
    for rollout in rollouts:
        try:
            correct_flags.append(
                is_rollout_correct(
                    pred_text=rollout,
                    gt_text=gt_text,
                    task_type=task_type,
                    iou_threshold=iou_threshold,
                    point_dist_threshold=dist_threshold,
                    maze_grid=maze_grid,
                )
            )
        except Exception:
            correct_flags.append(False)

    if not correct_flags:
        return "hard", None, 0.0

    correct_count = sum(correct_flags)
    avg_score = correct_count / len(correct_flags)

    # Difficulty classification by correct rollout count (paper Sec 2.5.2):
    #   Easy:   all rollouts correct.
    #   Hard:   all rollouts incorrect.
    #   Normal: at least one correct and at least one incorrect.
    if correct_count == len(correct_flags):
        difficulty = "easy"
    elif correct_count == 0:
        difficulty = "hard"
    else:
        difficulty = "normal"

    # Pick the best rollout for SFT data (highest continuous reward as tie-breaker)
    try:
        scores = []
        for rollout in rollouts:
            total = compute_total_reward(
                pred_text=rollout,
                gt_text=gt_text,
                task_type=task_type,
                iou_threshold=iou_threshold,
                point_dist_threshold=dist_threshold,
                maze_grid=maze_grid,
            )
            scores.append(total["total_reward"])
        best_idx = scores.index(max(scores))
    except Exception:
        best_idx = 0

    return difficulty, rollouts[best_idx], avg_score


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Load Unified model from merged Stage 2 base (fresh LoRA)
    logger.info(f"Loading Unified model from merged base: {args.model_path}")
    unified_model, processor = load_qlora_model(
        model_name=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    log_memory_status("Unified model loaded:")

    # 2. Generate or load cached prompts for rejection sampling
    def _generate_prompts():
        logger.info("Generating prompts for rejection sampling...")
        prompts = []

        box_prompts = generate_coco_box_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_box_prompts,
        )
        for d in box_prompts:
            d["task_type"] = "box"
        prompts.extend(box_prompts)
        logger.info(f"  Box prompts: {len(box_prompts)}")

        counting_prompts = generate_coco_counting_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_counting_prompts,
        )
        for d in counting_prompts:
            d["task_type"] = "box"
        prompts.extend(counting_prompts)
        logger.info(f"  Counting prompts: {len(counting_prompts)}")

        clevr_prompts = generate_clevr_spatial_dataset(
            n=args.num_clevr_prompts,
            seed=47,
            cache_dir=os.path.join(args.output_dir, "clevr_prompt_cache"),
        )
        for d in clevr_prompts:
            d["task_type"] = "box"
        prompts.extend(clevr_prompts)
        logger.info(f"  CLEVR prompts: {len(clevr_prompts)}")

        point_prompts = generate_coco_point_samples(
            image_dir=args.coco_image_dir,
            ann_file=args.coco_ann_file,
            num_samples=args.num_point_prompts,
        )
        for d in point_prompts:
            d["task_type"] = "point"
        prompts.extend(point_prompts)
        logger.info(f"  Point prompts: {len(point_prompts)}")

        maze_prompts = generate_maze_dataset(
            n=args.num_maze_prompts,
            seed=45,
        )
        for d in maze_prompts:
            d["task_type"] = "maze"
        prompts.extend(maze_prompts)
        logger.info(f"  Maze prompts: {len(maze_prompts)}")

        random.shuffle(prompts)
        logger.info(f"Total prompts: {len(prompts)}")
        return prompts

    all_prompts = runner.cached_data(
        os.path.join(args.output_dir, "prompts_cache.pkl"),
        _generate_prompts,
    )

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
            "answer": PrimitiveParser.extract_answer(best_rollout) or sample["answer"],
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
    runner = StageRunner(
        "stage5_rft_unified",
        "configs/stage5_rft_unified.yaml",
        description="Stage 5: Unified RFT",
    )
    runner.add_arg("--model_path", type=str, default=None)
    runner.add_arg("--output_dir", type=str, default=None)
    runner.add_arg("--box_expert_path", type=str, default=None)
    runner.add_arg("--point_expert_path", type=str, default=None)
    runner.add_arg("--coco_image_dir", type=str, default=None)
    runner.add_arg("--coco_ann_file", type=str,
                   default=None)
    runner.add_arg("--num_box_prompts", type=int, default=None)
    runner.add_arg("--num_counting_prompts", type=int, default=None,
                   help="Number of counting prompts for rejection sampling")
    runner.add_arg("--num_clevr_prompts", type=int, default=None,
                   help="Number of CLEVR spatial/VQA prompts for rejection sampling")
    runner.add_arg("--num_point_prompts", type=int, default=None)
    runner.add_arg("--num_maze_prompts", type=int, default=None)
    runner.add_arg("--num_rollouts", type=int, default=None)
    runner.add_arg("--max_new_tokens", type=int, default=None)
    runner.add_arg("--iou_threshold", type=float, default=None)
    runner.add_arg("--point_dist_threshold", type=float, default=None)
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
                   help="Path to checkpoint dir to resume SFT training, e.g. outputs/stage5_rft_unified/checkpoint-500")
    runner.run(train)
