"""Geometry utilities for boxes, points, and maze paths."""

from collections import deque
from typing import List, Tuple

import numpy as np


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
    from .text_parsing import extract_answer, _normalize_answer_text

    pred_answer = extract_answer(pred_text)
    gt_answer = extract_answer(gt_text)
    return 1.0 if (
        pred_answer is not None
        and _normalize_answer_text(pred_answer) == _normalize_answer_text(gt_answer)
    ) else 0.0


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


def _has_duplicate_coords(coords: List[Tuple[int, ...]], tolerance: int = 3) -> bool:
    """Check if any two coordinates are nearly identical."""
    if len(coords) < 2:
        return False
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if all(abs(a - b) <= tolerance for a, b in zip(coords[i], coords[j])):
                return True
    return False


def _count_repeated_coordinates(coords: List[Tuple[int, ...]], tolerance: int = 3) -> int:
    """Count coordinate clusters that appear multiple times."""
    if len(coords) < 2:
        return 0
    duplicates = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if all(abs(a - b) <= tolerance for a, b in zip(coords[i], coords[j])):
                duplicates += 1
    return duplicates


# ---------------------------------------------------------------------------
# Path Tracing Accuracy RM (paper Sec 2.5.2 — Accuracy RM for Path Tracing)
# ---------------------------------------------------------------------------


def point_to_segment_distance(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Minimum distance from point *p* to line segment *a*–*b*."""
    px, py = float(p[0]), float(p[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx ** 2 + aby ** 2
    if ab_len_sq == 0:
        return float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return float(np.hypot(px - proj_x, py - proj_y))


def point_to_polyline_distance(
    point: Tuple[float, float],
    polyline: List[Tuple[float, float]],
) -> float:
    """Minimum distance from *point* to any segment of *polyline*.

    Args:
        point: Query point (x, y).
        polyline: Ordered list of (x, y) waypoints forming the polyline.

    Returns:
        Minimum L2 distance to the polyline. Returns ``inf`` if the
        polyline has fewer than 2 points.
    """
    if len(polyline) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(polyline) - 1):
        d = point_to_segment_distance(point, polyline[i], polyline[i + 1])
        if d < min_dist:
            min_dist = d
    return min_dist


def path_forward_accuracy(
    pred_points: List[Tuple[int, int]],
    gt_curve: List[Tuple[int, int]],
    max_dist: float = 200.0,
) -> float:
    """Forward trajectory accuracy: how close predicted points are to GT curve.

    For each predicted point, compute its minimum distance to the GT polyline,
    then average. Normalise so that distance 0 → score 1, distance ≥ max_dist → 0.

    Args:
        pred_points: Model-predicted waypoints.
        gt_curve: Dense ground-truth curve waypoints (from Bézier sampling).
        max_dist: Distance at which score saturates to 0.

    Returns:
        Score in [0, 1]. Higher is better.
    """
    if not pred_points or len(gt_curve) < 2:
        return 0.0
    dists = [point_to_polyline_distance(p, gt_curve) for p in pred_points]
    avg = float(np.mean(dists))
    return max(0.0, 1.0 - avg / max_dist)


def path_reverse_accuracy(
    gt_curve: List[Tuple[int, int]],
    pred_points: List[Tuple[int, int]],
    max_dist: float = 200.0,
) -> float:
    """Reverse trajectory accuracy: how well GT curve is covered by predicted polyline.

    For each GT point, compute its minimum distance to the predicted polyline.
    Penalises incomplete coverage where the model skips portions of the curve.

    Args:
        gt_curve: Dense ground-truth curve waypoints.
        pred_points: Model-predicted waypoints forming the predicted polyline.
        max_dist: Distance at which score saturates to 0.

    Returns:
        Score in [0, 1]. Higher is better.
    """
    if not gt_curve or len(pred_points) < 2:
        return 0.0
    dists = [point_to_polyline_distance(p, pred_points) for p in gt_curve]
    avg = float(np.mean(dists))
    return max(0.0, 1.0 - avg / max_dist)


def path_endpoint_accuracy(
    pred_start: Tuple[int, int] | None,
    pred_end: Tuple[int, int] | None,
    gt_start: Tuple[int, int],
    gt_end: Tuple[int, int],
    tolerance: float = 50.0,
) -> float:
    """Endpoint accuracy: how well the model identifies start and end locations.

    Score decays linearly with distance, reaching 0 beyond *tolerance*.

    Args:
        pred_start: Model's predicted start coordinate (or None).
        pred_end: Model's predicted end coordinate (or None).
        gt_start: Ground-truth start coordinate.
        gt_end: Ground-truth end coordinate.
        tolerance: Distance beyond which score is 0.

    Returns:
        Score in [0, 1]. Average of start and end scores.
    """
    scores = []
    if pred_start is not None:
        d = point_distance(pred_start, gt_start)
        scores.append(max(0.0, 1.0 - d / tolerance))
    else:
        scores.append(0.0)
    if pred_end is not None:
        d = point_distance(pred_end, gt_end)
        scores.append(max(0.0, 1.0 - d / tolerance))
    else:
        scores.append(0.0)
    return float(np.mean(scores))


def path_continuity_penalty(
    pred_trajectory_last_point: Tuple[int, int] | None,
    pred_endpoint: Tuple[int, int] | None,
    threshold: float = 80.0,
    penalty: float = 0.1,
) -> float:
    """Trajectory continuity penalty: detect 'jump to guessed endpoint'.

    If the distance between the last point of the model's trajectory and its
    predicted endpoint exceeds *threshold*, apply a fixed negative penalty.
    This discourages outputting a partial trajectory and jumping to the end.

    Returns:
        0.0 if within threshold or points missing, -penalty otherwise.
    """
    if pred_trajectory_last_point is None or pred_endpoint is None:
        return 0.0
    d = point_distance(pred_trajectory_last_point, pred_endpoint)
    if d > threshold:
        return -penalty
    return 0.0
