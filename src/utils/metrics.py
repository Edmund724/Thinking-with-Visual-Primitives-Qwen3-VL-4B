"""Process-level reward metrics for visual primitives."""

import re
from typing import List, Tuple

import numpy as np

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


def box_iou(box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_boxes(
    pred_boxes: List[Tuple[int, ...]],
    gt_boxes: List[Tuple[int, ...]],
    iou_threshold: float = 0.5,
) -> Tuple[float, int, int]:
    """Match predicted boxes to GT boxes via greedy IoU matching.

    Returns:
        (average_iou, num_matches, num_gt)
    """
    if not gt_boxes:
        return 0.0, 0, 0
    if not pred_boxes:
        return 0.0, 0, len(gt_boxes)

    gt_matched = [False] * len(gt_boxes)
    ious = []

    for pb in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for idx, gb in enumerate(gt_boxes):
            if gt_matched[idx]:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            gt_matched[best_idx] = True
            ious.append(best_iou)

    avg_iou = float(np.mean(ious)) if ious else 0.0
    return avg_iou, len(ious), len(gt_boxes)


def point_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """L2 distance between two points."""
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def match_points(
    pred_points: List[Tuple[int, int]],
    gt_points: List[Tuple[int, int]],
    dist_threshold: float = 10.0,
) -> Tuple[float, int, int]:
    """Match predicted points to GT points via greedy L2 matching.

    Returns:
        (average_distance, num_matches, num_gt)
    """
    if not gt_points:
        return 0.0, 0, 0
    if not pred_points:
        return 0.0, 0, len(gt_points)

    gt_matched = [False] * len(gt_points)
    dists = []

    for pp in pred_points:
        best_dist = float("inf")
        best_idx = -1
        for idx, gp in enumerate(gt_points):
            if gt_matched[idx]:
                continue
            d = point_distance(pp, gp)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        if best_idx >= 0 and best_dist <= dist_threshold:
            gt_matched[best_idx] = True
            dists.append(best_dist)

    avg_dist = float(np.mean(dists)) if dists else float("inf")
    return avg_dist, len(dists), len(gt_points)


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


def check_wall_collision(
    points: List[Tuple[int, int]],
    maze_grid: np.ndarray,
) -> List[int]:
    """Check if line segments between consecutive points cross walls.

    Args:
        points: List of (x, y) points in normalized 0-999 coords.
        maze_grid: Binary grid where 0=wall, 1=path.

    Returns:
        List of step indices where collision occurs.
    """
    if len(points) < 2 or maze_grid is None:
        return []

    h, w = maze_grid.shape
    collisions = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        # Convert normalized coords to grid coords
        gx1, gy1 = int(x1 / 999 * (w - 1)), int(y1 / 999 * (h - 1))
        gx2, gy2 = int(x2 / 999 * (w - 1)), int(y2 / 999 * (h - 1))

        # Bresenham-like sampling along the line
        steps = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
        for t in range(steps + 1):
            gx = int(gx1 + (gx2 - gx1) * t / steps)
            gy = int(gy1 + (gy2 - gy1) * t / steps)
            gx = np.clip(gx, 0, w - 1)
            gy = np.clip(gy, 0, h - 1)
            if maze_grid[gy, gx] == 0:
                collisions.append(i)
                break

    return collisions


def check_backtracking_missing(
    pred_text: str,
    gt_text: str,
) -> bool:
    """Check if model failed to backtrack when GT required it.

    Heuristic: if GT contains 'backtrack' / 'dead end' but prediction doesn't.
    """
    gt_lower = gt_text.lower()
    pred_lower = pred_text.lower()

    gt_has_backtrack = any(w in gt_lower for w in ["backtrack", "dead end", "retreat", "go back"])
    pred_has_backtrack = any(w in pred_lower for w in ["backtrack", "dead end", "retreat", "go back"])

    return gt_has_backtrack and not pred_has_backtrack


def process_reward(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    maze_grid: np.ndarray | None = None,
) -> dict:
    """Compute comprehensive process-level reward.

    Args:
        pred_text: Model's full output.
        gt_text: Ground truth full output.
        task_type: "box", "point", or "maze".
        maze_grid: Optional maze grid for wall collision detection.

    Returns:
        Dict with metrics.
    """
    pred_answer = extract_answer(pred_text)
    gt_answer = extract_answer(gt_text)
    answer_correct = pred_answer is not None and pred_answer == gt_answer

    pred_reasoning = extract_reasoning(pred_text)
    gt_reasoning = extract_reasoning(gt_text)

    valid = syntax_valid(pred_text)

    result = {
        "answer_correct": answer_correct,
        "syntax_valid": valid,
        "pred_has_answer": pred_answer is not None,
    }

    if task_type in ("box", "point"):
        pred_boxes = parse_boxes(pred_reasoning)
        gt_boxes = parse_boxes(gt_reasoning)
        avg_iou, num_match, num_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
        result.update({
            "box_avg_iou": avg_iou,
            "box_precision": len(pred_boxes),
            "box_recall": num_match / num_gt if num_gt > 0 else 0.0,
            "box_f1": (
                2 * num_match / (len(pred_boxes) + num_gt)
                if (len(pred_boxes) + num_gt) > 0 else 0.0
            ),
            "box_num_pred": len(pred_boxes),
            "box_num_gt": num_gt,
        })

    if task_type in ("point", "maze"):
        pred_points = parse_points(pred_reasoning)
        gt_points = parse_points(gt_reasoning)
        avg_dist, num_match, num_gt = match_points(
            pred_points, gt_points, point_dist_threshold
        )
        result.update({
            "point_avg_dist": avg_dist,
            "point_precision": len(pred_points),
            "point_recall": num_match / num_gt if num_gt > 0 else 0.0,
            "point_f1": (
                2 * num_match / (len(pred_points) + num_gt)
                if (len(pred_points) + num_gt) > 0 else 0.0
            ),
            "point_num_pred": len(pred_points),
            "point_num_gt": num_gt,
        })

        # Maze-specific checks
        if maze_grid is not None:
            wall_collisions = check_wall_collision(pred_points, maze_grid)
            result["wall_collision_count"] = len(wall_collisions)
            result["wall_collision_steps"] = wall_collisions
            result["backtracking_missing"] = check_backtracking_missing(pred_text, gt_text)

    return result
