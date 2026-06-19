#!/usr/bin/env python3
"""Evaluate Stage 2 merged model on visual primitive format compliance.

Checks:
- Output contains expected primitive tags (<|box|>, <|point|>).
- No non-Latin characters (Chinese, etc.).
- Tags are properly paired.
- Coordinates are parseable and in [0, 999].
- Answer number is consistent with number of primitives (lenient ±1).
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.data.generators.coco_box_generator import generate_coco_box_samples
from src.data.generators.coco_box_generator import generate_coco_point_samples
from src.utils.constants import BOX_CLOSE, BOX_OPEN, POINT_CLOSE, POINT_OPEN
from src.utils.conversation_builder import ConversationBuilder
from src.models.visual_primitive_parser import PrimitiveParser


def contains_non_latin(text: str) -> bool:
    """Return True if text contains characters outside Basic Latin + common punctuation."""
    for ch in text:
        cat = ord(ch)
        # Basic Latin
        if cat <= 0x007F:
            continue
        # Latin-1 Supplement letters are OK, but CJK/etc are not
        if 0x0080 <= cat <= 0x00FF:
            continue
        # General Punctuation and currency symbols are OK
        if 0x2000 <= cat <= 0x206F:
            continue
        if 0x20A0 <= cat <= 0x20CF:
            continue
        return True
    return False


def tags_paired(text: str, task_type: str) -> bool:
    """Check that primitive tags are properly paired."""
    if task_type in ("box",):
        return text.count(BOX_OPEN) == text.count(BOX_CLOSE) and text.count(BOX_OPEN) > 0
    if task_type in ("point",):
        return text.count(POINT_OPEN) == text.count(POINT_CLOSE) and text.count(POINT_OPEN) > 0
    return True


def evaluate(model, processor, samples, task_type, max_new_tokens=256):
    """Run model on samples and collect metrics."""
    stats = {
        "total": 0,
        "has_primitive_tags": 0,
        "tags_paired": 0,
        "no_non_latin": 0,
        "coords_in_range": 0,
        "answer_consistent": 0,
    }

    device = next(model.parameters()).device
    for sample in tqdm(samples, desc=f"Eval {task_type}"):
        stats["total"] += 1
        image = Image.open(sample["image"]).convert("RGB")
        messages = ConversationBuilder("sft").build_prompt(sample["prompt"], image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        output = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Extract assistant reasoning/content.
        assistant_part = output.split("<|im_start|>assistant")[-1]

        # 1. Primitive tags present
        if task_type == "box":
            has_tags = BOX_OPEN in assistant_part and BOX_CLOSE in assistant_part
        elif task_type == "point":
            has_tags = POINT_OPEN in assistant_part and POINT_CLOSE in assistant_part
        else:
            has_tags = False
        if has_tags:
            stats["has_primitive_tags"] += 1

        # 2. Tags paired
        if tags_paired(assistant_part, task_type):
            stats["tags_paired"] += 1

        # 3. No non-Latin
        if not contains_non_latin(assistant_part):
            stats["no_non_latin"] += 1

        # 4. Coords in range
        coords_ok = True
        if task_type == "box":
            boxes = PrimitiveParser.lenient_extract_boxes(assistant_part)
            if not boxes or not all(0 <= c <= 999 for b in boxes for c in b):
                coords_ok = False
        elif task_type == "point":
            points = PrimitiveParser.extract_points(assistant_part)
            if not points or not all(0 <= c <= 999 for p in points for c in p):
                coords_ok = False
        if coords_ok:
            stats["coords_in_range"] += 1

        # 5. Answer consistency (lenient ±1)
        pred_answer = PrimitiveParser.extract_answer(assistant_part)
        gt_answer = str(sample.get("answer", ""))
        try:
            pred_num = int(re.findall(r"\d+", pred_answer)[-1]) if re.findall(r"\d+", pred_answer) else None
        except (ValueError, IndexError):
            pred_num = None
        try:
            gt_num = int(re.findall(r"\d+", gt_answer)[-1]) if re.findall(r"\d+", gt_answer) else None
        except (ValueError, IndexError):
            gt_num = None

        if pred_num is not None and gt_num is not None and abs(pred_num - gt_num) <= 1:
            stats["answer_consistent"] += 1

    return stats


def main(args):
    print(f"Loading model from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    # Box evaluation
    box_samples = generate_coco_box_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_box,
        use_thinking=False,
        seed=args.seed,
    )
    box_stats = evaluate(model, processor, box_samples, "box", args.max_new_tokens)

    # Point evaluation
    point_samples = generate_coco_point_samples(
        image_dir=args.coco_image_dir,
        ann_file=args.coco_ann_file,
        num_samples=args.num_point,
        use_thinking=False,
        seed=args.seed + 1,
    )
    point_stats = evaluate(model, processor, point_samples, "point", args.max_new_tokens)

    def report(name, stats):
        print(f"\n=== {name} ===")
        total = stats["total"]
        for k, v in stats.items():
            if k == "total":
                continue
            print(f"  {k}: {v}/{total} ({100 * v / total:.1f}%)")

    report("Box Task", box_stats)
    report("Point Task", point_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--coco_image_dir", type=str, default=None)
    parser.add_argument("--coco_ann_file", type=str, default=None)
    parser.add_argument("--num_box", type=int, default=None)
    parser.add_argument("--num_point", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
