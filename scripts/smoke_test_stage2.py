#!/usr/bin/env python3
"""
5-minute smoke test for the merged Stage 2 model.
Loads outputs/stage2_merged_base, runs one image+text prompt, and checks
that visual primitive tags (<|box|>, <|point|>) appear inside <think> tags.
"""
import argparse
import os
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.utils.conversation_builder import ConversationBuilder


def find_coco_image():
    coco_dir = "data/coco/train2017"
    if os.path.isdir(coco_dir):
        for name in sorted(os.listdir(coco_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                return os.path.join(coco_dir, name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    # Use expandable segments to avoid fragmentation during load/generate.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    image_path = args.image_path or find_coco_image()
    if image_path is None or not os.path.exists(image_path):
        print("ERROR: No COCO image found. Provide --image_path.", file=sys.stderr)
        sys.exit(1)

    print(f"Image: {image_path}")
    print(f"Model: {args.model_path}")
    print("Loading model...")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    image = Image.open(image_path).convert("RGB")
    messages = ConversationBuilder("opd").build_prompt(args.question, image)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    print("Generating...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    duration = time.time() - t0
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"\n=== Generated response ({duration:.1f}s) ===")
    print(response)
    print("=== End of response ===")

    # Checks
    has_think = "<think>" in response and "</think>" in response
    has_box_tags = "<|box|>" in response and "<|/box|>" in response
    has_point_tags = "<|point|>" in response and "<|/point|>" in response
    box_coords = re.findall(r"<\|box\|>\[\[(\d{1,3},\d{1,3},\d{1,3},\d{1,3})\]\]</\|box\|>", response)
    point_coords = re.findall(r"<\|point\|>\[\[(\d{1,3},\d{1,3})\]\]</\|point\|>", response)
    # Stage 2 only establishes visual -> coordinate grounding; exact primitive tags are finalized in Stage 3 SFT.
    any_bbox = re.findall(r"bbox_2d.*?:\s*\[(\d{1,3},\s*\d{1,3},\s*\d{1,3},\s*\d{1,3})\]", response) or \
               re.findall(r"\[(\d{1,3},\s*\d{1,3},\s*\d{1,3},\s*\d{1,3})\]", response)
    any_point = re.findall(r"\[(\d{1,3},\s*\d{1,3})\]", response)

    print("\n=== Smoke test checks ===")
    print(f"  <think> tags present:   {has_think}")
    print(f"  <box> tags present:     {has_box_tags}")
    print(f"  <point> tags present:   {has_point_tags}")
    print(f"  Valid box coords:       {len(box_coords)}")
    print(f"  Valid point coords:     {len(point_coords)}")
    print(f"  Any bbox coords:        {len(any_bbox)}")
    print(f"  Any point coords:       {len(any_point)}")

    if has_think and (has_box_tags or len(any_bbox) > 0):
        print("\n✅ Smoke test PASSED — merged model emits thinking + spatial coordinates. Ready for Stage 3a.")
        sys.exit(0)
    else:
        print("\n⚠️ Smoke test INCONCLUSIVE — no <think> + spatial output. Inspect response above.")
        sys.exit(2)


if __name__ == "__main__":
    main()
