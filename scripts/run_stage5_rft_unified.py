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
import hashlib
import os
import pickle
import random
import sys

import torch

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
from src.data.generators.path_tracing import generate_path_tracing_dataset
from src.data.datasets.image_loader import load_image
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.callbacks import TimeLoggingCallback
from src.training.memory_utils import log_memory_status, clear_memory
from src.utils.reward.accuracy_rm import compute_total_reward
from src.models.visual_primitive_parser import PrimitiveParser
from src.utils.conversation_builder import ConversationBuilder


def _build_quality_fn(use_api: bool, logger):
    """Return a Quality RM scorer ``f(pred_text, gt_text, task_type) -> float``.

    The Quality RM (paper Sec 2.5.2) is used here to select the best rollout as
    the SFT target during difficulty grading, so reward-hacking or redundant
    rollouts are not distilled into the Unified model.

    Default is the rule-based approximation (``quality_reward_text``). When
    ``use_api`` is set, use the LLM-based GRM (``quality_reward_api``) with a
    single pre-built client, mirroring the ``--use_quality_rm_api`` switch of
    Stages 4a/4b. The API path falls back to rule-based scoring on any failure.
    """
    if not use_api:
        from src.utils.reward.quality_rm import quality_reward_text
        logger.info("Quality RM (rollout selection): rule-based approximation")
        return quality_reward_text

    from src.utils.quality_rm_api import quality_reward_api, _load_api_config
    cfg = _load_api_config()
    client = None
    if cfg:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        except Exception:
            client = None
    logger.info(
        "Quality RM (rollout selection): LLM GRM via API (use_quality_rm_api=True)"
    )

    def _quality_fn(pred_text, gt_text, task_type):
        return quality_reward_api(pred_text, gt_text, task_type, client=client)

    return _quality_fn


def _expert_for_task(task_type: str) -> str:
    """Map a sample's task type to the expert that should generate for it."""
    return "box" if task_type == "box" else "point"


def generate_with_expert(expert_model, processor, sample, num_rollouts, max_new_tokens):
    """Expert generates N rollouts for a given sample."""
    if max_new_tokens is None:
        raise ValueError(
            "max_new_tokens must be set for expert generation. "
            "Add `max_new_tokens: 512` to configs/stage5_rft_unified.yaml "
            "or pass `--max_new_tokens 512`."
        )

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
        # max_length=None prevents transformers from falling back to a default
        # max_length when max_new_tokens is configured; the generation length is
        # then fully controlled by max_new_tokens.
        outputs = expert_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
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


def difficulty_grading(rollouts, gt_text, task_type, maze_grid, iou_threshold, dist_threshold, quality_fn=None):
    """Grade rollouts as Easy/Normal/Hard based on the number of correct rollouts.

    Matches the paper's Specialized RL / Unified RFT data selection (Sec 2.5.2
    and 2.5.3): a rollout is considered correct only when its final answer is
    correct AND its output satisfies basic syntax constraints. We then classify
    the prompt by the count of correct rollouts among the N generated samples.

    ``quality_fn`` (optional Quality RM, paper Sec 2.5.2) is folded into the
    best-rollout selection so that reward-hacking or redundant rollouts are not
    picked as the SFT target.
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

    # Pick the best rollout for SFT data. Score = accuracy/total reward plus an
    # optional Quality RM score (paper Sec 2.5.2), added with weight 1.0 to match
    # the equal weighting of the Quality reward in Stage 4a/4b GRPO.
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
            score = total["total_reward"]
            if quality_fn is not None:
                try:
                    score += quality_fn(rollout, gt_text, task_type)
                except Exception:
                    pass
            scores.append(score)
        best_idx = scores.index(max(scores))
    except Exception:
        best_idx = 0

    return difficulty, rollouts[best_idx], avg_score


def train(runner: StageRunner) -> None:
    args, logger = runner.args, runner.logger

    # 1. Determine resume checkpoint (explicit flag or latest auto checkpoint)
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt and not os.path.isdir(resume_ckpt):
        logger.error(f"Requested checkpoint not found: {resume_ckpt}")
        sys.exit(1)
    if not resume_ckpt:
        latest = runner.latest_checkpoint(args.output_dir)
        if latest:
            logger.info(f"Auto-resuming from latest checkpoint: {latest}")
            resume_ckpt = latest
        else:
            logger.info(f"No checkpoint found; starting fresh from {args.model_path}")

    # 2. Load Unified model from checkpoint when resuming, otherwise from merged Stage 2 base
    load_path = resume_ckpt if resume_ckpt else args.model_path
    logger.info(f"Loading Unified model from: {load_path}")
    unified_model, processor = load_qlora_model(
        model_name=load_path,
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

        path_prompts = generate_path_tracing_dataset(
            n=args.num_path_prompts,
            seed=46,
            cache_dir=os.path.join(args.output_dir, "path_tracing_prompt_cache"),
        )
        # task_type is already "path" from generator
        prompts.extend(path_prompts)
        logger.info(f"  Path tracing prompts: {len(path_prompts)}")

        random.shuffle(prompts)
        logger.info(f"Total prompts: {len(prompts)}")
        return prompts

    prompt_cache_key = (
        f"{args.num_box_prompts}|{args.num_counting_prompts}|{args.num_clevr_prompts}|"
        f"{args.num_point_prompts}|{args.num_maze_prompts}|{args.num_path_prompts}|"
        f"{args.coco_image_dir}|{args.coco_ann_file}"
    )
    prompt_cache_hash = hashlib.md5(prompt_cache_key.encode()).hexdigest()[:8]
    prompt_cache_path = os.path.join(
        args.output_dir, f"prompts_cache_{prompt_cache_hash}.pkl"
    )

    # Pre-compute filtered cache path so we can decide whether prompts are needed.
    filtered_cache_key = (
        f"{args.num_box_prompts}|{args.num_counting_prompts}|{args.num_clevr_prompts}|"
        f"{args.num_point_prompts}|{args.num_maze_prompts}|{args.num_path_prompts}|"
        f"{args.box_expert_path}|{args.point_expert_path}|"
        f"{args.num_rollouts}|{args.max_new_tokens}|"
        f"{args.iou_threshold}|{args.point_dist_threshold}|"
        f"{args.use_quality_rm_api}"
    )
    filtered_cache_hash = hashlib.md5(filtered_cache_key.encode()).hexdigest()[:8]
    filtered_cache_path = os.path.join(
        args.output_dir, f"filtered_data_cache_{filtered_cache_hash}.pkl"
    )

    if not args.skip_expert_generation:
        if args.regenerate_data and os.path.exists(prompt_cache_path):
            logger.info(f"--regenerate_data set; removing old cache {prompt_cache_path}")
            os.remove(prompt_cache_path)

        all_prompts = runner.cached_data(prompt_cache_path, _generate_prompts)

    # 3. Generate or load cached filtered data (expert generation + difficulty grading)

    def _run_expert_generation():
        # Quality RM used to select the best rollout as the SFT target
        # (paper Sec 2.5.2). Built once here; skipped on a cache hit since this
        # closure only runs when filtered data is regenerated.
        quality_fn = _build_quality_fn(args.use_quality_rm_api, logger)

        # Generate rollouts for each task type, loading only the relevant expert
        # into GPU memory at a time. Stage 4 experts are full QLoRA models; two
        # of them plus the Unified model do not fit on a single 24 GB card.
        filtered_data = []
        easy_samples = []
        hard_count = 0

        task_type_order = ["box", "point"]
        for task_type_to_process in task_type_order:
            expert_prompts = [
                s for s in all_prompts
                if _expert_for_task(s.get("task_type", "box")) == task_type_to_process
            ]
            if not expert_prompts:
                continue

            logger.info(
                f"Loading {task_type_to_process.capitalize()} Expert for "
                f"{len(expert_prompts)} prompts..."
            )
            expert_path = (
                args.box_expert_path if task_type_to_process == "box"
                else args.point_expert_path
            )
            expert, _ = load_qlora_model(
                model_name=expert_path,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
            )
            log_memory_status(f"{task_type_to_process.capitalize()} Expert loaded:")

            for i, sample in enumerate(expert_prompts):
                rollouts, gt_text, _, maze_grid = generate_with_expert(
                    expert, processor, sample,
                    args.num_rollouts, args.max_new_tokens,
                )

                difficulty, best_rollout, avg_score = difficulty_grading(
                    rollouts, gt_text, sample.get("task_type", "box"),
                    sample.get("maze_grid"),
                    args.iou_threshold, args.point_dist_threshold,
                    quality_fn=quality_fn,
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
                        f"  {task_type_to_process} {i + 1}/{len(expert_prompts)}: "
                        f"{len(filtered_data)} normal kept, {len(easy_samples)} easy, {hard_count} hard"
                    )

            # Release this expert before loading the next one
            logger.info(f"Releasing {task_type_to_process.capitalize()} Expert...")
            expert_path_to_release = expert_path
            expert = None
            del expert
            gc.collect()
            clear_memory()
            log_memory_status(f"{task_type_to_process.capitalize()} Expert released:")

        # Retain all Normal + 5% Easy to mitigate catastrophic forgetting
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

        return filtered_data

    if args.skip_expert_generation:
        if args.train_data_path:
            logger.info(f"--skip_expert_generation set; loading train data from {args.train_data_path}")
            with open(args.train_data_path, "rb") as f:
                filtered_data = pickle.load(f)
        elif os.path.exists(filtered_cache_path):
            logger.info(f"--skip_expert_generation set; loading cached filtered data from {filtered_cache_path}")
            with open(filtered_cache_path, "rb") as f:
                filtered_data = pickle.load(f)
        else:
            logger.warning(
                "--skip_expert_generation is set but no training data is available "
                f"(--train_data_path not given and {filtered_cache_path} does not exist). "
                "Skipping SFT. To generate the filtered data, set "
                "skip_expert_generation: false in the config or pass --no-skip_expert_generation."
            )
            return
    else:
        if args.regenerate_data and os.path.exists(filtered_cache_path):
            logger.info(f"--regenerate_data set; removing old cache {filtered_cache_path}")
            os.remove(filtered_cache_path)

        filtered_data = runner.cached_data(filtered_cache_path, _run_expert_generation)

    if len(filtered_data) < args.min_normal_samples:
        logger.warning(
            f"Too few Normal samples ({len(filtered_data)} < "
            f"min_normal_samples={args.min_normal_samples}) — consider "
            "adjusting thresholds or increasing prompt counts"
        )
        return

    # 6. SFT Unified model on Normal-difficulty expert data
    logger.info("Training Unified model on Normal-difficulty expert data...")

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
        additional_callbacks=[TimeLoggingCallback()],
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
    runner.add_arg("--num_path_prompts", type=int, default=None)
    runner.add_arg("--num_rollouts", type=int, default=None)
    runner.add_arg("--max_new_tokens", type=int, default=512)
    runner.add_arg("--iou_threshold", type=float, default=None)
    runner.add_arg("--point_dist_threshold", type=float, default=None)
    runner.add_arg("--min_normal_samples", type=int, default=None,
                   help="Minimum number of Normal + retained Easy samples required "
                        "to start Unified SFT. Default is 10 in fast mode.")
    runner.add_arg(
        "--use_quality_rm_api",
        action="store_true",
        help="Use OpenAI-compatible API (LLM GRM) for the Quality RM used in "
             "best-rollout selection (requires .env key). Off by default.",
    )
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
    runner.add_arg("--regenerate_data", action="store_true",
                   help="Force regeneration of prompts and filtered data, ignoring existing caches.")
    runner.add_arg("--skip_expert_generation", action="store_true",
                   help="Skip the expert rollout generation / difficulty grading step. "
                        "Use this when you already have prepared training data or want to "
                        "go straight to SFT training.")
    runner.add_arg("--train_data_path", type=str, default=None,
                   help="Path to a pickle file containing pre-filtered training records. "
                        "Used only when --skip_expert_generation is set.")
    runner.run(train)
