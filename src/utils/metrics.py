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


def maze_causal_exploration_progress(
    pred_points: List[Tuple[int, int]],
    maze_grid: np.ndarray,
    start: Tuple[int, int] | None = None,
    end: Tuple[int, int] | None = None,
) -> float:
    """Causal exploration progress for solvable mazes.

    After the FIRST wall violation, all subsequent exploration is invalid
    (causal chain broken). Measures distance from the last causally-valid
    point to the end point.

    Returns:
        Score in [0, 1]: 1 = reached end, 0 = no valid exploration.
    """
    if maze_grid is None or len(pred_points) < 2:
        return 0.0

    h, w = maze_grid.shape

    # Find first wall violation
    first_violation = -1
    for i in range(len(pred_points) - 1):
        x1, y1 = pred_points[i]
        x2, y2 = pred_points[i + 1]
        gx1, gy1 = int(x1 / 999 * (w - 1)), int(y1 / 999 * (h - 1))
        gx2, gy2 = int(x2 / 999 * (w - 1)), int(y2 / 999 * (h - 1))
        steps = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
        violated = False
        for t in range(steps + 1):
            gx = int(gx1 + (gx2 - gx1) * t / steps)
            gy = int(gy1 + (gy2 - gy1) * t / steps)
            gx = np.clip(gx, 0, w - 1)
            gy = np.clip(gy, 0, h - 1)
            if maze_grid[gy, gx] == 0:
                first_violation = i
                violated = True
                break
        if violated:
            break

    if first_violation == -1:
        # No violations — full exploration is valid
        valid_points = pred_points
    else:
        # Truncate at first violation (exclusive)
        valid_points = pred_points[:first_violation + 1]

    if not valid_points:
        return 0.0

    # Score: how close did valid exploration get to the end?
    if end is None:
        return 0.0

    ex, ey = end
    # Normalize end to 0-999
    nex, ney = int(ex / (w - 1) * 999) if w > 1 else 0, int(ey / (h - 1) * 999) if h > 1 else 0

    last_point = valid_points[-1]
    dist_to_end = point_distance(last_point, (nex, ney))
    max_dist = np.sqrt(999 ** 2 + 999 ** 2)  # ~1413

    return max(0.0, 1.0 - dist_to_end / max_dist)


def maze_exploration_completeness(
    pred_points: List[Tuple[int, int]],
    maze_grid: np.ndarray,
) -> float:
    """Exploration completeness for unsolvable mazes.

    Measures what fraction of all reachable cells were explored by the
    exhaustive search.

    Returns:
        Score in [0, 1]: fraction of reachable cells visited.
    """
    if maze_grid is None or not pred_points:
        return 0.0

    h, w = maze_grid.shape

    # Find all reachable cells from start via BFS
    # Start is typically the first point's grid position
    if not pred_points:
        return 0.0
    sx, sy = pred_points[0]
    gsx, gsy = int(sx / 999 * (w - 1)), int(sy / 999 * (h - 1))
    gsx, gsy = np.clip(gsx, 0, w - 1).item(), np.clip(gsy, 0, h - 1).item()

    reachable = set()
    from collections import deque
    q = deque([(gsx, gsy)])
    while q:
        cx, cy = q.popleft()
        if (cx, cy) in reachable:
            continue
        reachable.add((cx, cy))
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and maze_grid[ny, nx] == 1:
                q.append((nx, ny))

    if not reachable:
        return 0.0

    # Count how many unique grid cells the prediction visited
    visited = set()
    for px, py in pred_points:
        gx, gy = int(px / 999 * (w - 1)), int(py / 999 * (h - 1))
        gx, gy = np.clip(gx, 0, w - 1).item(), np.clip(gy, 0, h - 1).item()
        visited.add((gx, gy))

    return min(1.0, len(visited) / len(reachable))


def maze_wall_violation_penalty(
    pred_points: List[Tuple[int, int]],
    maze_grid: np.ndarray,
) -> float:
    """Global wall violation penalty.

    Measures ratio of steps that cross walls, independent of causal
    truncation logic.

    Returns:
        Penalty in [0, 1]: 0 = no violations, 1 = all steps cross walls.
    """
    if maze_grid is None or len(pred_points) < 2:
        return 0.0

    collisions = check_wall_collision(pred_points, maze_grid)
    total_steps = len(pred_points) - 1
    violation_ratio = len(collisions) / total_steps if total_steps > 0 else 0.0

    # Return penalty (1 - violation_ratio), but scaled
    # 0 violations = score 1, all violations = score 0
    return 1.0 - violation_ratio


def maze_path_validity(
    pred_points: List[Tuple[int, int]],
    maze_grid: np.ndarray,
) -> float:
    """Final path validity for solvable mazes.

    Checks if consecutive points are adjacent (Manhattan distance ≤ 1)
    in grid space, AND not crossing walls.

    Returns:
        Score in [0, 1]: fraction of consecutive pairs that are valid.
    """
    if maze_grid is None or len(pred_points) < 2:
        return 0.0

    h, w = maze_grid.shape
    valid_pairs = 0
    total_pairs = len(pred_points) - 1

    for i in range(total_pairs):
        x1, y1 = pred_points[i]
        x2, y2 = pred_points[i + 1]
        gx1, gy1 = int(x1 / 999 * (w - 1)), int(y1 / 999 * (h - 1))
        gx2, gy2 = int(x2 / 999 * (w - 1)), int(y2 / 999 * (h - 1))

        # Check Manhattan adjacency
        manhattan = abs(gx2 - gx1) + abs(gy2 - gy1)
        if manhattan == 0:
            continue  # Same cell, skip
        if manhattan > 1:
            continue  # Non-adjacent

        # Check no wall crossing
        if maze_grid[gy1, gx1] == 1 and maze_grid[gy2, gx2] == 1:
            valid_pairs += 1

    return valid_pairs / total_pairs if total_pairs > 0 else 0.0


def maze_answer_correctness(pred_text: str, gt_text: str) -> float:
    """Binary check: did the model correctly determine solvability?

    Extracts \\boxed{True} or \\boxed{False} and compares.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    pred_answer = extract_answer(pred_text)
    gt_answer = extract_answer(gt_text)
    return 1.0 if (pred_answer is not None and pred_answer == gt_answer) else 0.0


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

    if task_type in ("box", "point", "path"):
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

    if task_type in ("point", "maze", "path"):
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


def format_reward(text: str) -> dict:
    """Format Reward Model (Format RM).

    Checks:
        1. <think>...</think> tags exist and are paired
        2. Special token syntax is valid (paired open/close)
        3. Coordinates are within [0, 999]
        4. No duplicate special tokens (e.g., nested boxes)

    Returns:
        Dict with individual check results and total score (0.0~0.8).
    """
    from .constants import BOX_OPEN, BOX_CLOSE, POINT_OPEN, POINT_CLOSE

    score = 0.0
    details = {}

    # 1. Check <think> tags
    think_open = text.count("<think>")
    think_close = text.count("</think>")
    has_think = think_open == 1 and think_close == 1
    details["has_think_tags"] = has_think
    if has_think:
        score += 0.2

    # 2. Check special token pairing
    box_open = text.count(BOX_OPEN)
    box_close = text.count(BOX_CLOSE)
    point_open = text.count(POINT_OPEN)
    point_close = text.count(POINT_CLOSE)
    tokens_paired = (
        box_open == box_close and point_open == point_close
        and box_open >= 0 and point_open >= 0
    )
    details["tokens_paired"] = tokens_paired
    if tokens_paired:
        score += 0.2

    # 3. Check coordinate range [0, 999]
    coords_valid = True
    for match in BOX_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            nums = [int(x) for x in re.findall(r"\d+", coords_str)]
            for n in nums:
                if n < 0 or n > 999:
                    coords_valid = False
                    break
        except (ValueError, IndexError):
            coords_valid = False
        if not coords_valid:
            break

    for match in POINT_PATTERN.finditer(text):
        coords_str = match.group(1)
        try:
            nums = [int(x) for x in re.findall(r"\d+", coords_str)]
            for n in nums:
                if n < 0 or n > 999:
                    coords_valid = False
                    break
        except (ValueError, IndexError):
            coords_valid = False
        if not coords_valid:
            break

    details["coords_in_range"] = coords_valid
    if coords_valid:
        score += 0.2

    # 4. Check no duplicate/nested special tokens
    # Simple heuristic: each <|box|> should have exactly one matching <|/box|>
    # and no nested boxes
    no_duplicates = True
    # Check for patterns like <|box|>...<|box|>...<|/box|>...<|/box|>
    # which indicates nesting
    box_segments = re.findall(
        re.escape(BOX_OPEN) + r"(.*?)" + re.escape(BOX_CLOSE), text, re.DOTALL
    )
    for seg in box_segments:
        if BOX_OPEN in seg or BOX_CLOSE in seg:
            no_duplicates = False
            break

    point_segments = re.findall(
        re.escape(POINT_OPEN) + r"(.*?)" + re.escape(POINT_CLOSE), text, re.DOTALL
    )
    for seg in point_segments:
        if POINT_OPEN in seg or POINT_CLOSE in seg:
            no_duplicates = False
            break

    details["no_nested_tokens"] = no_duplicates
    if no_duplicates:
        score += 0.2

    details["total_format_score"] = round(score, 2)
    return details


def counting_reward(pred_count: int, gt_count: int) -> float:
    """Counting accuracy reward with smooth exponential decay.

    Formula from paper: R = 0.7 * exp(-3 * |y_hat - y| / (|y| + 1))

    Args:
        pred_count: Predicted count.
        gt_count: Ground truth count.

    Returns:
        Reward score in [0, 0.7].
    """
    if gt_count == 0:
        # Avoid division by zero; if GT is 0, reward is 0.7 only if pred is also 0
        return 0.7 if pred_count == 0 else 0.0

    abs_error = abs(pred_count - gt_count)
    reward = 0.7 * np.exp(-3 * abs_error / (gt_count + 1))
    return float(reward)


def compute_total_reward(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    maze_grid: np.ndarray | None = None,
) -> dict:
    """Compute total reward = Format RM + Accuracy RM for GRPO training.

    Returns dict with:
        - total_reward: scalar for GRPO
        - format_reward: Format RM score (0~0.8)
        - accuracy_reward: Accuracy RM score (0~1.0+)
        - difficulty: "easy" / "normal" / "hard" (for difficulty grading)
    """
    # Format RM
    fmt = format_reward(pred_text)
    format_score = fmt["total_format_score"]

    # Process reward (Accuracy RM base)
    proc = process_reward(
        pred_text, gt_text, task_type,
        iou_threshold, point_dist_threshold, maze_grid,
    )

    accuracy_score = 0.0

    if task_type == "box":
        # Box: IoU + counting support
        avg_iou = proc.get("box_avg_iou", 0.0)
        num_pred = proc.get("box_num_pred", 0)
        num_gt = proc.get("box_num_gt", 0)

        # Try to get counts for counting reward
        pred_answer = extract_answer(pred_text)
        gt_answer = extract_answer(gt_text)
        if pred_answer is not None and gt_answer is not None:
            try:
                pred_count = int(pred_answer)
                gt_count = int(gt_answer)
                count_r = counting_reward(pred_count, gt_count)
            except ValueError:
                count_r = 0.0
        else:
            count_r = 0.0

        # If counts match perfectly, use IoU; otherwise use counting reward
        if count_r >= 0.7 * 0.99:  # Almost perfect count match
            accuracy_score = avg_iou + 0.3  # Bonus for correct count
        else:
            accuracy_score = count_r + avg_iou * 0.3  # Weighted mix

    elif task_type == "point":
        # Point: distance-based
        avg_dist = proc.get("point_avg_dist", float("inf"))
        if avg_dist != float("inf"):
            accuracy_score = max(0, 1.0 - min(avg_dist, 100.0) / 100.0)

    elif task_type == "maze":
        # 5-component Maze Accuracy RM (paper-aligned):
        # 1. Causal exploration progress (solvable) — how far before first wall hit
        # 2. Exploration completeness (unsolvable) — fraction of reachable cells
        # 3. Wall violation penalty — global ratio (both solvable/unsolvable)
        # 4. Path validity (solvable) — adjacency + no-wall for final path
        # 5. Answer correctness — binary solvability judgment
        pred_points = parse_points(extract_reasoning(pred_text))
        gt_points = parse_points(extract_reasoning(gt_text))
        maze_grid_data = maze_grid

        answer_correct = maze_answer_correctness(pred_text, gt_text)

        if answer_correct > 0.5:
            # Solvable maze — use components 1, 3, 4, 5
            causal_progress = maze_causal_exploration_progress(
                pred_points, maze_grid_data)
            wall_penalty = maze_wall_violation_penalty(
                pred_points, maze_grid_data)
            path_valid = maze_path_validity(
                pred_points, maze_grid_data)
            accuracy_score = (
                0.25 * causal_progress +
                0.15 * wall_penalty +
                0.20 * path_valid +
                0.40 * answer_correct
            )
        else:
            # Unsolvable maze — use components 2, 3, 5
            exploration_comp = maze_exploration_completeness(
                pred_points, maze_grid_data)
            wall_penalty = maze_wall_violation_penalty(
                pred_points, maze_grid_data)
            accuracy_score = (
                0.30 * exploration_comp +
                0.20 * wall_penalty +
                0.50 * answer_correct
            )

    # Total reward
    total = format_score + accuracy_score

    # Difficulty grading
    # Easy: total >= 1.5 (both format and accuracy are good)
    # Normal: 0.5 <= total < 1.5 (partially correct)
    # Hard: total < 0.5 (mostly wrong)
    if total >= 1.5:
        difficulty = "easy"
    elif total >= 0.5:
        difficulty = "normal"
    else:
        difficulty = "hard"

    return {
        "total_reward": round(total, 4),
        "format_reward": round(format_score, 4),
        "accuracy_reward": round(accuracy_score, 4),
        "difficulty": difficulty,
        "process_metrics": proc,
        "format_details": fmt,
    }
