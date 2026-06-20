"""Strict verifier for 'thinking with visual primitives' cold-start data.

Ensures that the generated thinking chain (reasoning) is faithful to the
underlying annotations before it is used for SFT / GRPO / RFT training.
"""

import re
from typing import Dict, List, Tuple

from ..models.visual_primitive_parser import PrimitiveParser
from .constants import BOX_OPEN, BOX_CLOSE, POINT_OPEN, POINT_CLOSE


def _parse_int(text: str) -> int | None:
    """Extract the last integer from text, if any."""
    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


def _parse_boxes(text: str) -> List[Tuple[int, int, int, int]]:
    """Parse all boxes from text as (x1, y1, x2, y2)."""
    return PrimitiveParser.extract_boxes(text)


def _parse_points(text: str) -> List[Tuple[int, int]]:
    """Parse all points from text as (x, y)."""
    return PrimitiveParser.extract_points(text)


def _tags_paired(text: str) -> bool:
    """Check that primitive tags are properly paired."""
    return PrimitiveParser.validate_syntax(text)


def _coords_in_range(coords: List[int]) -> bool:
    """Check that all coordinates are within [0, 999]."""
    return all(0 <= c <= 999 for c in coords)


def _refs_meaningful(text: str) -> bool:
    """Check that box references are not empty or pure numbers."""
    refs = re.findall(r"<\|ref\|>(.*?)<\|/ref\|>", text)
    if not refs:
        return True
    for ref in refs:
        ref_clean = ref.strip()
        if not ref_clean or ref_clean.isdigit():
            return False
    return True


def verify_thinking_chain(sample: Dict, tolerate_count_error: int = 1) -> Tuple[bool, str]:
    """Verify a single cold-start sample's thinking chain.

    Checks:
      1. Tags are paired.
      2. All coordinates are in [0, 999].
      3. References are meaningful (for box tasks).
      4. For counting-style answers, the final answer number is consistent
         with the number of visual primitives in the reasoning.
      5. For maze tasks, solvable/unsolvable claims are not blatantly wrong.

    Args:
        sample: Dict with keys {reasoning, answer, task_type, ...}.
        tolerate_count_error: Maximum allowed difference between answer count
            and number of visual primitives (default 1 for partial occlusion).

    Returns:
        (is_valid, reason) tuple.
    """
    reasoning = sample.get("reasoning", "")
    answer = sample.get("answer", "")
    task_type = sample.get("task_type", "box")

    if not reasoning:
        return False, "empty reasoning"

    # 1. Tag pairing
    if not _tags_paired(reasoning):
        return False, "unpaired primitive tags"

    # 2. Coordinate range
    boxes = _parse_boxes(reasoning)
    points = _parse_points(reasoning)
    all_coords = []
    for b in boxes:
        all_coords.extend(b)
    for p in points:
        all_coords.extend(p)
    if all_coords and not _coords_in_range(all_coords):
        return False, "coordinates out of [0, 999]"

    # 3. Meaningful references for box tasks
    if task_type == "box" and not _refs_meaningful(reasoning):
        return False, "empty or numeric box reference"

    # 4. Count consistency for counting-style answers
    answer_count = _parse_int(answer)
    if answer_count is not None:
        if task_type == "box":
            num_primitives = len(boxes) if task_type == "box" else len(points)
            # Some tasks ask for total count, others ask for category count.
            # We allow the reasoning to contain at least as many primitives as
            # the answer, within tolerance.
            if num_primitives > 0 and abs(answer_count - num_primitives) > tolerate_count_error:
                return False, f"answer count {answer_count} inconsistent with {num_primitives} primitives"
        elif task_type == "maze":
            # For maze, answer should be True/False; numeric answer is suspicious
            pass

    # 5. Maze contradiction
    if task_type == "maze":
        answer_lower = str(answer).lower()
        claims_solvable = "true" in answer_lower or "yes" in answer_lower
        has_path = len(points) >= 2
        if claims_solvable and not has_path:
            return False, "maze claims solvable but provides no path"

    return True, "ok"


def filter_verified_samples(samples: List[Dict], logger=None) -> List[Dict]:
    """Filter a list of cold-start samples, keeping only verified thinking chains."""
    verified = []
    reasons = {}
    for sample in samples:
        is_valid, reason = verify_thinking_chain(sample)
        if is_valid:
            verified.append(sample)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1

    if logger is not None:
        logger.info(
            f"Thinking-chain verification: {len(verified)}/{len(samples)} passed"
        )
        if reasons:
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  - {reason}: {count} rejected")

    return verified
