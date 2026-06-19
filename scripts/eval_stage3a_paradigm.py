#!/usr/bin/env python3
"""Quick paradigm check for stage3a box expert.

Loads the stage3a adapter and generates on a small held-out slice of box
localization, coarse-grained counting and CLEVR spatial/VQA samples.
Reports whether the model emits the "language interleaved with coordinates"
reasoning style (Intent / Grounding with <|box|> / Summarization).
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src.data.generators.coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
)
from src.data.generators.clevr_spatial import generate_clevr_spatial_dataset
from src.models.qwen_vl_loader import load_qlora_model
from src.utils.constants import BOX_OPEN, BOX_CLOSE
from src.utils.conversation_builder import ConversationBuilder


def build_eval_set(args):
    """Generate a tiny held-out eval set with fixed seeds."""
    data = []

    box_data = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_samples,
        seed=args.seed,
        max_objects_per_image=3,
    )
    for d in box_data:
        d["task_type"] = "box"
        d["subtype"] = "localization"
    data.extend(box_data)

    count_data = generate_coco_counting_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_samples,
        seed=args.seed + 1,
    )
    for d in count_data:
        d["task_type"] = "box"
        d["subtype"] = "counting"
    data.extend(count_data)

    clevr_data = generate_clevr_spatial_dataset(
        n=args.num_samples,
        seed=args.seed + 2,
        cache_dir=os.path.join(args.output_dir, "eval_clevr_cache"),
    )
    for d in clevr_data:
        d["task_type"] = "box"
        d["subtype"] = "clevr"
    data.extend(clevr_data)

    return data


def normalize_answer(text: str, relaxed: bool = False) -> str:
    text = text.strip().lower()
    # Remove special tokens and boilerplate the model may emit
    text = re.sub(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>", "", text)
    text = text.replace("the answer is", "").replace("\\boxed{", "").replace("}", "")
    if relaxed:
        text = text.strip().rstrip(".,;:!?")
    # Keep the last numeric or boolean word
    words = text.strip().split()
    if words:
        last = words[-1]
        if re.fullmatch(r"\d+|true|false", last):
            return last
    # Otherwise extract last number
    nums = re.findall(r"\d+", text)
    if nums:
        return nums[-1]
    return text.strip()


def parse_output(text: str):
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer_text = text[think_match.end():].strip()
    else:
        reasoning = text.strip()
        answer_text = ""

    return reasoning, answer_text


def coords_in_range(coords):
    return all(0 <= c <= 999 for c in coords)


def parse_boxes_from_text(text: str):
    boxes = []
    pattern = re.compile(re.escape(BOX_OPEN) + r"(.*?)" + re.escape(BOX_CLOSE))
    for m in pattern.finditer(text):
        inner = m.group(1).strip()
        try:
            nums = [int(float(x.strip())) for x in inner.strip("[]").split(",")]
        except Exception:
            continue
        if len(nums) == 4:
            boxes.append(tuple(nums))
        elif len(nums) > 4 and len(nums) % 4 == 0:
            for i in range(0, len(nums), 4):
                boxes.append(tuple(nums[i:i + 4]))
    return boxes


def evaluate_sample(model, processor, sample, device):
    image = sample.get("image")
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    messages = ConversationBuilder("sft").build_prompt(sample["prompt"], image)

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    # Strip the prompt prefix if present
    if generated.startswith(prompt_text):
        generated = generated[len(prompt_text):].strip()

    reasoning, answer_text = parse_output(generated)
    pred_answer = normalize_answer(answer_text) if answer_text else normalize_answer(generated)
    gt_answer = normalize_answer(str(sample.get("answer", "")))

    boxes = parse_boxes_from_text(reasoning)
    all_coords = [c for b in boxes for c in b]

    # Lenient box count: count any `[[x,y,...]]` coordinate array in reasoning
    raw_arrays = re.findall(r"\[\[([^\]]+)\]\]", reasoning)
    lenient_box_count = 0
    for arr in raw_arrays:
        try:
            nums = [int(float(x.strip())) for x in arr.split(",") if x.strip()]
            if len(nums) == 4:
                lenient_box_count += 1
            elif len(nums) > 4 and len(nums) % 4 == 0:
                lenient_box_count += len(nums) // 4
        except Exception:
            continue

    # Syntax OK = paired tags and at least one parseable box when a tag is present
    has_any_box_tag = BOX_OPEN in reasoning or BOX_CLOSE in reasoning
    syntax_ok = has_any_box_tag and len(boxes) > 0

    pred_answer_relaxed = normalize_answer(answer_text if answer_text else generated, relaxed=True)

    metrics = {
        "has_think": "<think>" in generated and "</think>" in generated,
        "has_intent": "intent analysis" in reasoning.lower(),
        "has_grounding": "grounding" in reasoning.lower(),
        "has_summarization": "summarization" in reasoning.lower(),
        "has_box_tag": BOX_OPEN in reasoning and BOX_CLOSE in reasoning,
        "interleaved": (
            ("intent analysis" in reasoning.lower() or "grounding" in reasoning.lower())
            and (BOX_OPEN in reasoning or BOX_CLOSE in reasoning)
        ),
        "syntax_ok": syntax_ok,
        "box_count": len(boxes),
        "lenient_box_count": lenient_box_count,
        "coords_valid": coords_in_range(all_coords) if all_coords else None,
        "answer_em": pred_answer == gt_answer,
        "answer_em_relaxed": pred_answer_relaxed == gt_answer,
        "pred_answer": pred_answer,
        "pred_answer_relaxed": pred_answer_relaxed,
        "gt_answer": gt_answer,
        "reasoning": reasoning,
        "generated": generated,
    }
    return metrics


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading adapter from {args.model_path}")
    model, processor = load_qlora_model(
        model_name=args.model_path,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model.eval()
    # Speed up generation: gradient checkpointing is unnecessary for eval and
    # forces use_cache=False, which makes autoregressive decode much slower.
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = True
    device = next(model.parameters()).device

    print("Building eval set...")
    eval_data = build_eval_set(args)
    print(f"Evaluating on {len(eval_data)} samples")

    results = []
    for idx, sample in enumerate(eval_data):
        metrics = evaluate_sample(model, processor, sample, device)
        metrics["idx"] = idx
        metrics["subtype"] = sample.get("subtype", "unknown")
        results.append(metrics)
        if (idx + 1) % 5 == 0:
            print(f"  evaluated {idx + 1}/{len(eval_data)}", flush=True)

        if idx < args.show_examples:
            print(f"\n--- Sample {idx} ({metrics['subtype']}) ---")
            print(f"Prompt: {sample['prompt']}")
            print(f"GT answer: {metrics['gt_answer']}")
            print(f"Pred answer: {metrics['pred_answer']}  (EM={metrics['answer_em']})")
            print(f"Reasoning:\n{metrics['reasoning'][:1200]}")

    # Aggregate
    by_subtype = defaultdict(lambda: defaultdict(list))
    for r in results:
        for k in ["has_think", "has_intent", "has_grounding", "has_summarization",
                  "has_box_tag", "interleaved", "syntax_ok", "answer_em", "answer_em_relaxed"]:
            if r[k] is not None:
                by_subtype[r["subtype"]][k].append(float(r[k]))
        if r["coords_valid"] is not None:
            by_subtype[r["subtype"]]["coords_valid"].append(float(r["coords_valid"]))
        by_subtype[r["subtype"]]["box_count"].append(r["box_count"])
        by_subtype[r["subtype"]]["lenient_box_count"].append(r["lenient_box_count"])

    summary = {}
    for subtype, vals in by_subtype.items():
        summary[subtype] = {
            k: round(sum(v) / len(v), 3) if k != "box_count" else round(sum(v) / len(v), 1)
            for k, v in vals.items()
        }
        summary[subtype]["n"] = len(vals["answer_em"])

    print("\n========== PARADIGM CHECK SUMMARY ==========")
    print(json.dumps(summary, indent=2))

    overall = {
        k: round(
            sum(summary[st][k] * summary[st]["n"] for st in summary)
            / sum(summary[st]["n"] for st in summary),
            3,
        )
        for k in ["has_think", "has_intent", "has_grounding", "has_summarization",
                  "has_box_tag", "interleaved", "syntax_ok", "answer_em", "answer_em_relaxed"]
    }
    overall["n"] = sum(summary[st]["n"] for st in summary)
    print("\nOverall:")
    print(json.dumps(overall, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"summary": summary, "overall": overall, "results": results}, f, indent=2)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default=None)
    parser.add_argument("--coco_ann_file", type=str,
                        default=None)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples per subtype")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args()
    main(args)
