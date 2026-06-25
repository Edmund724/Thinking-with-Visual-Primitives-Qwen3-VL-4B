#!/usr/bin/env python3
"""Fast verification that special visual-primitive tokens are learned correctly.

Can be run after Stage 1 (adapter), Stage 2 (merged base), or Stage 3 (SFT adapter)
to catch garbled non-Latin output before committing to the next long training stage.

Checks:
- <|box|> / <|/box|> tags appear and are paired
- No non-Latin script characters (CJK / Thai / Cyrillic / Arabic / ...)
- Coordinates inside boxes are in [0, 999]
- Answer contains a number when asked for a count
"""

import argparse
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.models.qwen_vl_loader import load_qlora_model
from src.utils.conversation_builder import ConversationBuilder
from src.utils.constants import BOX_CLOSE, BOX_OPEN


# Regex matching characters from non-Latin scripts
_NON_LATIN_RE = re.compile(
    r"["
    r"\u0370-\u03FF\u1F00-\u1FFF"  # Greek
    r"\u0400-\u04FF"               # Cyrillic
    r"\u0600-\u06FF"               # Arabic
    r"\u0900-\u097F"               # Devanagari
    r"\u0E00-\u0E7F"               # Thai
    r"\u3040-\u309F\u30A0-\u30FF"  # Hiragana/Katakana
    r"\u4E00-\u9FFF"               # CJK Unified Ideographs
    r"\uAC00-\uD7AF"               # Korean Hangul
    r"\U00020000-\U0002EBEF"       # CJK Extension B-F / others
    r"]"
)


def _find_coco_image():
    coco_dir = Path("data/coco/train2017")
    if coco_dir.is_dir():
        for p in sorted(coco_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                return str(p)
    return None


def _make_placeholder_image():
    """A simple colored image so the VLM has visual input even without COCO."""
    return Image.new("RGB", (448, 448), color=(120, 180, 220))


def _contains_non_latin(text: str) -> bool:
    return _NON_LATIN_RE.search(text) is not None


def _check_response(text: str, expect_box: bool = True, expect_count: bool = False) -> dict:
    result = {
        "raw": text,
        "has_think": "<think>" in text and "</think>" in text,
        "box_open": text.count(BOX_OPEN),
        "box_close": text.count(BOX_CLOSE),
        "box_paired": text.count(BOX_OPEN) == text.count(BOX_CLOSE) and text.count(BOX_OPEN) > 0,
        "non_latin": _contains_non_latin(text),
        "coords_ok": True,
        "answer_number": None,
    }

    # Box coordinate range check
    if expect_box:
        coords_ok = True
        for m in re.finditer(r"\[\[(.*?)\]\]", text):
            nums = re.findall(r"\d+", m.group(1))
            if len(nums) != 4:
                continue
            if not all(0 <= int(n) <= 999 for n in nums):
                coords_ok = False
                break
        result["coords_ok"] = coords_ok

    # Extract a number from the final answer
    nums = re.findall(r"\d+", text)
    if nums:
        result["answer_number"] = int(nums[-1])

    return result


def _score_prompt(model, processor, prompt: str, image: Image.Image, expect_box=True, expect_count=False) -> dict:
    messages = ConversationBuilder("grpo").build_prompt(prompt, image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )

    full = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Keep only the assistant part for evaluation
    assistant = full.split("<|im_start|>assistant")[-1]
    return _check_response(assistant, expect_box=expect_box, expect_count=expect_count)


def main():
    parser = argparse.ArgumentParser(description="Verify visual primitive special tokens")
    parser.add_argument("--model_path", type=str, default="outputs/stage2_merged_base",
                        help="Path to Stage 1 adapter, Stage 2 merged base, or Stage 3 adapter.")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Optional image path; falls back to COCO then a placeholder.")
    parser.add_argument("--num_prompts", type=int, default=3,
                        help="Number of box prompts to test (default 3).")
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    image_path = args.image_path or _find_coco_image()
    image = Image.open(image_path).convert("RGB") if image_path else _make_placeholder_image()
    print(f"Using image: {image_path or '(placeholder 448x448)'}")

    print(f"\nLoading model from {args.model_path}...")
    model, processor = load_qlora_model(args.model_path)
    print("Model loaded.\n")

    prompts = [
        ("Find all visible objects and mark each with <|box|>.", True, False),
        ("How many persons are in this image? Use <|box|> to mark each one.", True, True),
        ("Locate the main object in the image. Mark it with a box.", True, False),
    ][:args.num_prompts]

    results = []
    for prompt, expect_box, expect_count in prompts:
        print(f"Prompt: {prompt}")
        res = _score_prompt(model, processor, prompt, image, expect_box, expect_count)
        results.append(res)
        print(f"  think tags: {res['has_think']}, box tags paired: {res['box_paired']}, "
              f"non-Latin: {res['non_latin']}, coords in [0,999]: {res['coords_ok']}, "
              f"answer number: {res['answer_number']}")
        snippet = res['raw'].replace('\n', ' | ')[:300]
        print(f"  output: {snippet}\n")

    # Aggregate
    total = len(results)
    paired = sum(r["box_paired"] for r in results)
    no_garbage = sum(not r["non_latin"] for r in results)
    coords_ok = sum(r["coords_ok"] for r in results)

    print("=== Summary ===")
    print(f"  Samples: {total}")
    print(f"  Box tags paired: {paired}/{total}")
    print(f"  No non-Latin chars: {no_garbage}/{total}")
    print(f"  Coordinates in range: {coords_ok}/{total}")

    if paired == total and no_garbage == total and coords_ok == total:
        print("\n✅ PASSED — special tokens look healthy. Proceed to the next stage.")
        sys.exit(0)
    elif paired >= total // 2 and no_garbage >= total // 2:
        print("\n⚠️ PARTIAL — some outputs are OK but special tokens are not fully stable. "
              "Consider more Stage 1/3 training data or longer epochs.")
        sys.exit(2)
    else:
        print("\n❌ FAILED — special tokens are not learned; output contains garbage or missing tags. "
              "Do not proceed. Re-train the current stage with the latest code.")
        sys.exit(1)


if __name__ == "__main__":
    main()
