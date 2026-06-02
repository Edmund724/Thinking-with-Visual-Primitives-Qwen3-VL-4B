#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Stage 3: RFT (Rejection Sampling Fine-Tuning).

GRPO output model → generate 5 rollouts per sample → select best by
process_reward → SFT on filtered high-quality data.
"""

import argparse
import logging
import yaml

import torch

from src.data.datasets.sft_dataset import SFTDataset
from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.data.generators.synthetic_maze import generate_maze_dataset
from src.data.generators.synthetic_path import generate_path_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.training.trainers.sft_trainer import create_sft_trainer
from src.training.memory_utils import log_memory_status
from src.utils.logging_utils import setup_logging
from src.utils.metrics import process_reward, split_generated_text

logger = setup_logging(log_file="logs/stage3_rft.log")


def rejection_sample(model, processor, sample, config, max_new_tokens):
    """Generate 5 rollouts, return best by process reward."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful visual reasoning assistant. Think step by step.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        },
    ]
    gt_text = sample["reasoning"] + f"\n</think>\n\nThe answer is {sample.get('answer', '')}."

    best_score = -float("inf")
    best_reasoning = None
    best_answer = None

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[sample["image"]],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    for _ in range(config.get("num_rollouts", 5)):
        with torch.inference_mode():
            outputs = model.generate(
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

        r = process_reward(
            pred_text=pred,
            gt_text=gt_text,
            task_type=sample.get("task_type", "box"),
            iou_threshold=config.get("rejection_iou_threshold", 0.7),
            point_dist_threshold=config.get("rejection_point_dist_threshold", 10.0),
            maze_grid=sample.get("maze_grid"),
        )

        score = 0.0
        if r["answer_correct"]:
            score += 1.0
            if r["syntax_valid"]:
                score += 0.2
            if sample.get("task_type") == "box":
                score += r.get("box_avg_iou", 0.0)
            elif sample.get("task_type") in ("point", "maze"):
                dist = r.get("point_avg_dist", float("inf"))
                score += max(0, 1.0 - min(dist, 100.0) / 100.0)
            if sample.get("task_type") == "maze":
                if r.get("wall_collision_count", 0) == 0:
                    score += 0.3

        if score > best_score:
            best_score = score
            reasoning, answer = split_generated_text(pred)
            best_reasoning = reasoning
            best_answer = answer

    return best_reasoning, best_answer, best_score


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Stage 3: RFT (Rejection Sampling Fine-Tuning)")
    logger.info("=" * 60)

    torch.cuda.empty_cache()

    # 1. Load GRPO model
    model_path = args.model_path or config["grpo_checkpoint"]
    logger.info(f"Loading model from {model_path}...")
    model, processor = load_qlora_model(
        model_name=model_path,
        lora_r=config.get("lora_r", 64),
        lora_alpha=config.get("lora_alpha", 128),
    )
    log_memory_status("Model loaded:")

    # 2. Generate data for rejection sampling
    logger.info("Generating data for rejection sampling...")
    all_data = []

    box_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_coco or config.get("num_coco_samples", 5000),
    )
    for d in box_data:
        d["task_type"] = "box"
    all_data.extend(box_data)
    logger.info(f"  Box: {len(box_data)}")

    maze_data = generate_maze_dataset(
        n=args.num_maze or config.get("num_maze_samples", 3000),
        seed=45,
    )
    for d in maze_data:
        d["task_type"] = "maze"
    all_data.extend(maze_data)
    logger.info(f"  Maze: {len(maze_data)}")

    path_data = generate_path_dataset(
        n=args.num_path or config.get("num_path_samples", 2000),
        seed=45,
    )
    for d in path_data:
        d["task_type"] = "point"
    all_data.extend(path_data)
    logger.info(f"  Path: {len(path_data)}")

    logger.info(f"Total candidates: {len(all_data)}")

    # 3. Rejection sampling
    logger.info("Running rejection sampling...")
    filtered_data = []
    rejected = 0

    for i, sample in enumerate(all_data):
        best_reasoning, best_answer, best_score = rejection_sample(
            model, processor, sample, config, args.max_new_tokens,
        )

        if best_score >= args.accept_threshold:
            filtered_data.append({
                "image": sample["image"],
                "prompt": sample["prompt"],
                "reasoning": best_reasoning or sample["reasoning"],
                "answer": best_answer or sample["answer"],
                "task_type": sample.get("task_type", "box"),
            })
        else:
            rejected += 1

        if (i + 1) % 50 == 0:
            logger.info(
                f"  {i + 1}/{len(all_data)}: "
                f"{len(filtered_data)} accepted, {rejected} rejected"
            )

    accept_rate = len(filtered_data) / max(len(all_data), 1) * 100
    logger.info(
        f"Rejection sampling done: {len(filtered_data)} accepted, "
        f"{rejected} rejected ({accept_rate:.1f}%)"
    )

    if len(filtered_data) < 100:
        logger.warning("Too few accepted samples — consider lowering accept_threshold")
        return

    # 4. SFT on filtered data
    logger.info("Training SFT on filtered data...")
    trainer = create_sft_trainer(
        model=model,
        processor=processor,
        train_data=filtered_data,
        output_dir=config["output_dir"],
        num_epochs=config.get("num_train_epochs", 1),
        learning_rate=config.get("learning_rate", 1e-5),
        per_device_batch_size=config.get("per_device_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        max_seq_length=config.get("max_seq_length", 2048),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        warmup_steps=config.get("warmup_steps", 100),
        use_wandb=config.get("report_to") == "wandb",
    )

    trainer.train()
    final_dir = f"{config['output_dir']}/final_model"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    logger.info(f"Stage 3 complete. Final model saved to {final_dir}")
    log_memory_status("Stage 3 complete:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: RFT")
    parser.add_argument("--config", type=str, default="configs/stage3_rft.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default="data/coco/train2017")
    parser.add_argument("--coco_ann_file", type=str,
                        default="data/coco/annotations/instances_train2017.json")
    parser.add_argument("--num_coco", type=int, default=None)
    parser.add_argument("--num_maze", type=int, default=None)
    parser.add_argument("--num_path", type=int, default=None)
    parser.add_argument("--accept_threshold", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    main(parser.parse_args())
