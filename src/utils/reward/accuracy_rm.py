"""Accuracy Reward Model (Accuracy RM) for visual primitives."""

import re
from typing import Dict, List, Tuple

import numpy as np

from ...models.visual_primitive_parser import PrimitiveParser


def process_reward(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    maze_grid: np.ndarray | None = None,
) -> Dict:
    """Compute comprehensive process-level reward.

    Args:
        pred_text: Model's full output.
        gt_text: Ground truth full output.
        task_type: "box", "point", or "maze".
        maze_grid: Optional maze grid for wall collision detection.

    Returns:
        Dict with metrics.
    """
    pred_answer = PrimitiveParser.extract_answer(pred_text)
    gt_answer = PrimitiveParser.extract_answer(gt_text)
    answer_correct = (
        PrimitiveParser.normalize_answer_text(pred_answer)
        is not None
        and PrimitiveParser.normalize_answer_text(pred_answer) == PrimitiveParser.normalize_answer_text(gt_answer)
    )

    pred_reasoning = PrimitiveParser.extract_reasoning(pred_text)
    gt_reasoning = PrimitiveParser.extract_reasoning(gt_text)

    valid = PrimitiveParser.validate_syntax(pred_text)

    result = {
        "answer_correct": answer_correct,
        "syntax_valid": valid,
        "pred_has_answer": pred_answer is not None,
    }

    if task_type in ("box", "point", "path"):
        pred_boxes = PrimitiveParser.extract_boxes(pred_reasoning)
        gt_boxes = PrimitiveParser.extract_boxes(gt_reasoning)
        avg_iou, num_match, num_gt = PrimitiveParser.match_boxes(pred_boxes, gt_boxes, iou_threshold)
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
        pred_points = PrimitiveParser.extract_points(pred_reasoning)
        gt_points = PrimitiveParser.extract_points(gt_reasoning)
        avg_dist, num_match, num_gt = PrimitiveParser.match_points(
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
            wall_collisions = PrimitiveParser.check_wall_collision_points(pred_points, maze_grid)
            result["wall_collision_count"] = len(wall_collisions)
            result["wall_collision_steps"] = wall_collisions
            result["backtracking_missing"] = PrimitiveParser.check_backtracking_missing(pred_text, gt_text)

    return result


def box_count_answer_consistency_reward(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
) -> float:
    """Process reward: predicted box count should match the numeric answer.

    For positive box/counting samples, the answer is the number of objects,
    so the number of parseable boxes in the reasoning should equal that answer.
    For negative samples, the expected box count is 0.

    Returns:
        Positive reward if counts match, negative penalty otherwise.
        Zero if the task is not box-type or the GT answer is not a count/refusal.
    """
    if task_type != "box":
        return 0.0

    gt_answer = PrimitiveParser.extract_answer(gt_text)
    if gt_answer is None:
        return 0.0

    expected = _count_in_answer(gt_answer)
    # Explicit refusal/negation means "0 boxes expected".
    if expected is None and gt_answer.strip().lower() == "false":
        expected = 0

    if expected is None:
        return 0.0

    reasoning = PrimitiveParser.extract_reasoning(pred_text) or ""
    pred_boxes = PrimitiveParser.extract_boxes(reasoning)
    pred_count = len(pred_boxes)

    if pred_count == expected:
        return 0.2

    diff = abs(pred_count - expected)
    return -0.1 * min(diff, 3)


def _count_in_answer(answer_text: str | None) -> int | None:
    """Extract a count integer from answer text, if possible."""
    if not answer_text:
        return None
    nums = re.findall(r"\d+", answer_text)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


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


def length_reward(completion_length: int, target_length: int, max_penalty: float = 0.05) -> float:
    """Gentle length regularization reward.

    Encourages completions close to a target length. Penalizes only when the
    completion is longer than the target; penalty grows linearly up to
    `max_penalty` when length reaches 2x the target.

    Args:
        completion_length: Number of tokens in the completion.
        target_length: Desired completion length in tokens.
        max_penalty: Maximum negative reward (default 0.05).

    Returns:
        Continuous reward in [-max_penalty, 0.0].
    """
    if completion_length <= target_length:
        return 0.0
    over_ratio = (completion_length - target_length) / target_length
    return -max_penalty * min(over_ratio, 1.0)


def _count_repeated_ngrams(tokens: List[str], n: int = 4) -> int:
    """Count how many n-grams appear more than once."""
    if len(tokens) < n:
        return 0
    seen = set()
    repeats = set()
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram in seen:
            repeats.add(gram)
        else:
            seen.add(gram)
    return len(repeats)


def repeat_token_penalty(
    pred_text: str,
    max_penalty: float = 0.15,
    ngram: int = 4,
) -> float:
    """Penalty for repeated text patterns (looping / reward hacking).

    Checks:
      1. Repeated n-grams in the reasoning text.
      2. Repeated box/point coordinates.

    Returns:
        Negative penalty in [-max_penalty, 0.0].
    """
    reasoning = PrimitiveParser.extract_reasoning(pred_text) or ""
    tokens = reasoning.split()
    ngram_repeats = _count_repeated_ngrams(tokens, n=ngram)

    boxes = PrimitiveParser.extract_boxes(pred_text)
    points = PrimitiveParser.extract_points(pred_text)
    coord_repeats = PrimitiveParser.count_repeated_coordinates(boxes) + PrimitiveParser.count_repeated_coordinates(points)

    # Normalize: one duplicate n-gram or coordinate pair is minor; many are severe.
    total_repeats = ngram_repeats + coord_repeats
    if total_repeats == 0:
        return 0.0

    penalty = max_penalty * min(total_repeats / 5.0, 1.0)
    return -penalty


def compute_total_reward(
    pred_text: str,
    gt_text: str,
    task_type: str = "box",
    iou_threshold: float = 0.5,
    point_dist_threshold: float = 10.0,
    maze_grid: np.ndarray | None = None,
    gt_curve: list | None = None,
    question_type: str | None = None,
) -> Dict:
    """Compute total reward = Format RM + Accuracy RM for GRPO training.

    Returns dict with:
        - total_reward: scalar for GRPO
        - format_reward: Format RM score (0~0.8)
        - accuracy_reward: Accuracy RM score (0~1.0+)
        - difficulty: "easy" / "normal" / "hard" (for difficulty grading)
    """
    from .format_rm import format_reward, primitive_format_compliance_reward

    # Format RM
    fmt = format_reward(pred_text)
    format_score = fmt["total_format_score"]

    # Format compliance: paired and correctly ordered primitive tags
    fmt_compliance_score = primitive_format_compliance_reward(pred_text)

    # Process reward (Accuracy RM base)
    proc = process_reward(
        pred_text, gt_text, task_type,
        iou_threshold, point_dist_threshold, maze_grid,
    )

    # Box-count should match numeric answer for box tasks
    box_count_consistency = box_count_answer_consistency_reward(
        pred_text, gt_text, task_type
    )

    accuracy_score = 0.0

    if task_type == "box":
        # Box: IoU + counting support + non-count answer correctness
        avg_iou = proc.get("box_avg_iou", 0.0)
        answer_correct = proc.get("answer_correct", False)

        # CLEVR complex questions: use LLM judge for accuracy
        _SPATIAL_LLM_TYPES = {"multihop", "compare", "spatial_existence", "spatial_count"}
        if question_type in _SPATIAL_LLM_TYPES:
            from ..quality_rm_api import spatial_accuracy_rm_api
            spatial_score = spatial_accuracy_rm_api(
                pred_text, gt_text, question_type=question_type,
            )
            accuracy_score = spatial_score + avg_iou * 0.2
        else:
            # Standard box accuracy logic
            # Try to get counts for counting reward
            pred_answer = PrimitiveParser.extract_answer(pred_text)
            gt_answer = PrimitiveParser.extract_answer(gt_text)
            both_are_counts = False
            count_r = 0.0
            if pred_answer is not None and gt_answer is not None:
                try:
                    pred_count = int(pred_answer)
                    gt_count = int(gt_answer)
                    count_r = counting_reward(pred_count, gt_count)
                    both_are_counts = True
                except ValueError:
                    count_r = 0.0

            if both_are_counts:
                if count_r >= 0.7 * 0.99:
                    accuracy_score = avg_iou + 0.3
                else:
                    accuracy_score = count_r + avg_iou * 0.3
            else:
                answer_r = 1.0 if answer_correct else 0.0
                accuracy_score = answer_r + avg_iou * 0.3

    elif task_type == "point":
        # Point: distance-based
        avg_dist = proc.get("point_avg_dist", float("inf"))
        if avg_dist != float("inf"):
            accuracy_score = max(0, 1.0 - min(avg_dist, 100.0) / 100.0)

    elif task_type == "path":
        # Path tracing: 4-component Accuracy RM (paper Sec 2.5.2)
        pred_reasoning = PrimitiveParser.extract_reasoning(pred_text)
        pred_points = PrimitiveParser.extract_points(pred_reasoning) if pred_reasoning else []
        gt_reasoning = PrimitiveParser.extract_reasoning(gt_text)
        gt_points = PrimitiveParser.extract_points(gt_reasoning) if gt_reasoning else []

        # Use gt_curve (dense Bézier samples) if available, else fall back to gt_points
        curve = gt_curve if gt_curve and len(gt_curve) >= 2 else gt_points

        # 1. Forward accuracy (0.30): predicted points → GT curve
        forward = PrimitiveParser.path_forward_accuracy(pred_points, curve)

        # 2. Reverse accuracy (0.25): GT curve → predicted polyline
        reverse = PrimitiveParser.path_reverse_accuracy(curve, pred_points)

        # 3. Endpoint accuracy (0.20): start/end distance decay
        pred_start = pred_points[0] if pred_points else None
        pred_end = pred_points[-1] if pred_points else None
        gt_start = curve[0] if curve else (0, 0)
        gt_end = curve[-1] if curve else (0, 0)
        endpoint = PrimitiveParser.path_endpoint_accuracy(
            pred_start, pred_end, gt_start, gt_end
        )

        # 4. Answer correctness (0.25): binary
        answer_r = 1.0 if proc.get("answer_correct", False) else 0.0

        # 5. Continuity penalty: jump from last trajectory point to endpoint
        continuity = PrimitiveParser.path_continuity_penalty(
            pred_end if pred_points else None,
            pred_end if pred_points else None,
        )

        accuracy_score = (
            0.30 * forward
            + 0.25 * reverse
            + 0.20 * endpoint
            + 0.25 * answer_r
            + continuity
        )

    elif task_type == "maze":
        # 5-component Maze Accuracy RM (paper-aligned):
        # 1. Causal exploration progress (solvable) — how far before first wall hit
        # 2. Exploration completeness (unsolvable) — fraction of reachable cells
        # 3. Wall violation penalty — global ratio (both solvable/unsolvable)
        # 4. Path validity (solvable) — adjacency + no-wall for final path
        # 5. Answer correctness — binary solvability judgment
        pred_points = PrimitiveParser.extract_points(PrimitiveParser.extract_reasoning(pred_text))
        gt_points = PrimitiveParser.extract_points(PrimitiveParser.extract_reasoning(gt_text))
        maze_grid_data = maze_grid

        answer_correct = PrimitiveParser.maze_answer_correctness(pred_text, gt_text)

        if answer_correct > 0.5:
            # Solvable maze — use components 1, 3, 4, 5
            causal_progress = PrimitiveParser.maze_causal_exploration_progress(
                pred_points, maze_grid_data)
            wall_penalty = PrimitiveParser.maze_wall_violation_penalty(
                pred_points, maze_grid_data)
            path_valid = PrimitiveParser.maze_path_validity(
                pred_points, maze_grid_data)
            accuracy_score = (
                0.25 * causal_progress +
                0.15 * wall_penalty +
                0.20 * path_valid +
                0.40 * answer_correct
            )
        else:
            # Unsolvable maze — use components 2, 3, 5
            exploration_comp = PrimitiveParser.maze_exploration_completeness(
                pred_points, maze_grid_data)
            wall_penalty = PrimitiveParser.maze_wall_violation_penalty(
                pred_points, maze_grid_data)
            accuracy_score = (
                0.30 * exploration_comp +
                0.20 * wall_penalty +
                0.50 * answer_correct
            )

    # Repeat-token penalty (discourage looping / reward hacking)
    repeat_penalty = repeat_token_penalty(pred_text)

    # Total reward
    total = (
        format_score
        + fmt_compliance_score
        + accuracy_score
        + box_count_consistency
        + repeat_penalty
    )

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
        "format_compliance_reward": round(fmt_compliance_score, 4),
        "accuracy_reward": round(accuracy_score, 4),
        "box_count_consistency_reward": round(box_count_consistency, 4),
        "difficulty": difficulty,
        "process_metrics": proc,
        "format_details": fmt,
    }
