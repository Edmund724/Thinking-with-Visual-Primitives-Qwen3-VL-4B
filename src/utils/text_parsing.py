"""Text parsing utilities for visual primitive tags and answers."""

import re
from typing import List, Tuple

from .constants import BOX_PATTERN, POINT_PATTERN


def extract_answer(text: str) -> str | None:
    """Extract answer from generated or full text.

    Handles:
        - \\boxed{...}
        - <answer>...</answer>
        - Plain text after </think> (for COCO counting, path tracing, etc.)
    """
    # Try boxed first
    match = re.search(r"\\boxed\{(.*?)\}", text)
    if match:
        return match.group(1).strip()

    # Fallback to <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: extract text after </think>, stripping special tokens
    think_close = text.find("</think>")
    if think_close != -1:
        after = text[think_close + len("</think>"):].strip().lstrip("\n")
        # Remove Qwen special tokens like <|im_end|>, <|endoftext|>, etc.
        after = re.sub(r"<\|[^>]+\>", "", after)
        after = after.strip()
        if after:
            return after

    # Last resort: clean and return non-empty text
    cleaned = re.sub(r"<\|[^>]+\>", "", text).strip()
    return cleaned if cleaned else None


def split_generated_text(text: str) -> tuple[str | None, str | None]:
    """Split model-generated text into (reasoning, answer).

    Returns:
        (reasoning_text, answer_text) where either may be None.
    """
    reasoning = extract_reasoning(text)
    answer = extract_answer(text)
    return reasoning, answer


def extract_reasoning(text: str) -> str | None:
    """Extract reasoning content from Qwen3 thinking format.

    Generated text (model output after prompt) looks like:
        "I see cats. <|box|>...<|/box|>\n</think>\n\nThe answer is 2."
    Full chat-template text looks like:
        "...<think>\nI see cats. ...\n</think>\n\nThe answer is 2.<|im_end|>"
    """
    # Case 1: Full text with <think> tags (training data)
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Case 2: Generated text without opening <think> (model output)
    # Look for </think> to find where reasoning ends
    think_close = text.find("</think>")
    if think_close != -1:
        return text[:think_close].strip()

    # Fallback: treat everything before answer as reasoning
    return text.strip()


def parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """Parse all bounding boxes from text."""
    boxes = []
    for match in BOX_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            # Handle single box or list of boxes
            if "],[" in coords_str:
                parts = coords_str.split("],[")
            else:
                parts = [coords_str]
            for part in parts:
                part = part.strip("[]")
                nums = [int(float(x.strip())) for x in part.split(",")]
                if len(nums) == 4:
                    boxes.append((nums[0], nums[1], nums[2], nums[3]))
        except (ValueError, IndexError):
            continue
    return boxes


def parse_points(text: str) -> List[Tuple[int, int]]:
    """Parse all points from text."""
    points = []
    for match in POINT_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            if "],[" in coords_str:
                parts = coords_str.split("],[")
            else:
                parts = [coords_str]
            for part in parts:
                part = part.strip("[]")
                nums = [int(float(x.strip())) for x in part.split(",")]
                if len(nums) == 2:
                    points.append((nums[0], nums[1]))
        except (ValueError, IndexError):
            continue
    return points


_LENIENT_BOX_RE = re.compile(r"\[\[(.*?)\]\]")


def lenient_parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """Parse bounding boxes from text even when tags are missing/wrong.

    Extracts every ``[[x1,y1,x2,y2]]`` coordinate array. This is intentionally
    more forgiving than ``parse_boxes`` and is used for difficulty grading
    after SFT, where the model may have learned coordinates but not the exact
    tag order.
    """
    boxes = []
    for match in _LENIENT_BOX_RE.finditer(text):
        coords_str = match.group(1)
        try:
            if "],[" in coords_str:
                parts = coords_str.split("],[")
            else:
                parts = [coords_str]
            for part in parts:
                part = part.strip("[]")
                nums = [int(float(x.strip())) for x in part.split(",") if x.strip()]
                if len(nums) == 4:
                    boxes.append((nums[0], nums[1], nums[2], nums[3]))
        except (ValueError, IndexError):
            continue
    return boxes


def _normalize_answer_text(text: str | None) -> str | None:
    """Normalize an extracted answer for robust comparison.

    Handles:
      - \\boxed{...}
      - <answer>...</answer>
      - Plain text after </think>
      - Numeric extraction fallback
      - Boolean normalization
    """
    if text is None:
        return None

    normalized = text.strip()
    # Unwrap boxed / answer tags
    match = re.search(r"\\boxed\{(.*?)\}", normalized)
    if match:
        normalized = match.group(1)
    match = re.search(r"<answer>(.*?)</answer>", normalized, re.DOTALL)
    if match:
        normalized = match.group(1)

    normalized = re.sub(r"<\|[^>]+\>", "", normalized)
    normalized = re.sub(r"<think>|</think>|<answer>|</answer>", "", normalized)
    normalized = normalized.strip().rstrip(".,;:!?")
    lowered = normalized.lower()

    if lowered in ("true", "false"):
        return lowered

    nums = re.findall(r"\d+", normalized)
    if nums:
        return nums[-1]

    return lowered if lowered else None


def syntax_valid(text: str) -> bool:
    """Check if primitive tags are syntactically valid (paired)."""
    from .constants import BOX_OPEN, BOX_CLOSE, POINT_OPEN, POINT_CLOSE

    box_open_count = text.count(BOX_OPEN)
    box_close_count = text.count(BOX_CLOSE)
    point_open_count = text.count(POINT_OPEN)
    point_close_count = text.count(POINT_CLOSE)

    return (
        box_open_count == box_close_count
        and point_open_count == point_close_count
    )
