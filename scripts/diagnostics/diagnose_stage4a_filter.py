#!/usr/bin/env python3
"""Diagnose why Stage 4a difficulty filtering marks everything as Hard."""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.qwen_vl_loader import load_qlora_model
from src.utils.batch_inference import generate_single_completion
from src.utils.difficulty import is_rollout_correct
from src.utils.reward.accuracy_rm import compute_total_reward, process_reward
from src.utils.reward.format_rm import format_reward


def main():
    model_path = "outputs/stage3a_sft_box"
    cache_path = "outputs/stage4a_grpo_box/train_data_cache.pkl"

    print(f"Loading model from {model_path}...")
    model, processor = load_qlora_model(model_path, lora_r=256, lora_alpha=512)
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = True

    print(f"Loading data cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} samples")

    num_samples = 5
    num_generations = 2
    max_completion_length = 384

    for idx, sample in enumerate(data[:num_samples]):
        gt_text = (
            sample.get("reasoning", "")
            + f"\n</think>\n\nThe answer is {sample.get('answer', '')}."
        )
        print(f"\n{'='*60}")
        print(f"Sample {idx} | task_type={sample.get('task_type')}")
        print(f"Prompt: {sample['prompt'][:200]}")
        print(f"GT answer: {sample.get('answer')}")
        print(f"GT reasoning preview: {sample.get('reasoning','')[:200]}")

        correct = 0
        for gen_idx in range(num_generations):
            outputs, input_len = generate_single_completion(
                model=model,
                processor=processor,
                sample=sample,
                max_completion_length=max_completion_length,
                temperature=0.7,
            )
            pred = processor.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=False
            )
            fmt = format_reward(pred)
            proc = process_reward(pred, gt_text, task_type="box")
            total = compute_total_reward(pred, gt_text, task_type="box")
            ok = is_rollout_correct(pred, gt_text, task_type="box")
            if ok:
                correct += 1

            print(f"\n-- Generation {gen_idx} --")
            print(f"Pred (first 400 chars): {pred[:400]}")
            print(f"  tokens_paired={fmt.get('tokens_paired')}, "
                  f"coords_in_range={fmt.get('coords_in_range')}, "
                  f"no_nested={fmt.get('no_nested_tokens')}, "
                  f"has_think={fmt.get('has_think_tags')}")
            print(f"  answer_correct={proc.get('answer_correct')}, "
                  f"pred_boxes={proc.get('box_num_pred')}, "
                  f"gt_boxes={proc.get('box_num_gt')}")
            print(f"  total_reward={total['total_reward']}, is_rollout_correct={ok}")

        print(f"\n=> correct rollouts: {correct}/{num_generations}")


if __name__ == "__main__":
    main()
