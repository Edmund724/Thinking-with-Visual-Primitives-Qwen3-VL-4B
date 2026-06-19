"""Visual Primitive Parser — convenience facade for visual primitive operations.

This module provides static methods for parsing, formatting, validating, and
scoring visual primitive tokens (box / point) in generated text.  It delegates
to the underlying specialised modules:

* ``src.utils.text_parsing`` — parsing and syntax validation
* ``src.utils.geometry`` — geometric operations (IoU, distance, maze scoring)
* ``src.data.formatters.primitive_formatter`` — coordinate formatting

Production code may also import directly from these modules when it needs only a
subset of operations.  ``PrimitiveParser`` exists as a discoverability aid
(one import gives you everything).

For backward compatibility, ``src.utils.metrics`` re-exports the same public API.
"""

import re
from collections import deque
from typing import Callable, List, Tuple

import numpy as np

from ..data.formatters.primitive_formatter import (
    clean_primitive_tags,
    denormalize_coordinate,
    format_box,
    format_point,
    format_ref,
    normalize_coordinate,
)
from ..utils.constants import (
    BOX_CLOSE,
    BOX_OPEN,
    POINT_CLOSE,
    POINT_OPEN,
)
from ..utils.geometry import (
    _count_repeated_coordinates,
    _has_duplicate_coords,
    box_iou,
    check_backtracking_missing,
    check_wall_collision,
    match_boxes,
    match_points,
    maze_answer_correctness,
    maze_causal_exploration_progress,
    maze_exploration_completeness,
    maze_path_validity,
    maze_wall_violation_penalty,
    path_continuity_penalty,
    path_endpoint_accuracy,
    path_forward_accuracy,
    path_reverse_accuracy,
    point_distance,
    point_to_polyline_distance,
)
from ..utils.text_parsing import (
    _normalize_answer_text,
    extract_answer,
    extract_reasoning,
    extract_refs,
    lenient_parse_boxes,
    parse_boxes,
    parse_points,
    split_generated_text,
    syntax_valid,
    validate_ref_box_pairing,
)


class PrimitiveParser:
    """Domain seam for all visual primitive (box / point) operations.

    Static methods cover three concerns:
    * **Parsing** — extract primitives from model-generated text.
    * **Validation** — check syntax, coordinate bounds, wall collisions.
    * **Formatting** — build primitive tags for training data.
    * **Geometry** — IoU, point distance, maze metrics.

    Usage::

        from src.models.visual_primitive_parser import PrimitiveParser

        boxes = PrimitiveParser.extract_boxes(text)
        tags  = PrimitiveParser.format_box([(10, 20, 100, 200)])
        iou   = PrimitiveParser.box_iou(pred_box, gt_box)
    """

    # ── Parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def extract_answer(text: str) -> str | None:
        """Extract the final answer from model-generated text."""
        return extract_answer(text)

    @staticmethod
    def extract_reasoning(text: str) -> str | None:
        """Extract the reasoning / thinking chain from model-generated text."""
        return extract_reasoning(text)

    @staticmethod
    def split_generated_text(text: str) -> tuple[str | None, str | None]:
        """Split model-generated text into (reasoning, answer)."""
        return split_generated_text(text)

    @staticmethod
    def normalize_answer_text(text: str | None) -> str | None:
        """Normalize an extracted answer for robust comparison."""
        return _normalize_answer_text(text)

    @staticmethod
    def extract_boxes(text: str) -> List[Tuple[int, int, int, int]]:
        """Extract all bounding boxes from text.

        Returns list of (x1, y1, x2, y2) tuples.
        """
        return parse_boxes(text)

    @staticmethod
    def extract_points(text: str) -> List[Tuple[int, int]]:
        """Extract all points from text.

        Returns list of (x, y) tuples.
        """
        return parse_points(text)

    @staticmethod
    def extract_refs(text: str) -> List[str]:
        """Extract all ``<|ref|>...<|/ref|>`` reference names from text."""
        return extract_refs(text)

    @staticmethod
    def validate_ref_box_pairing(text: str) -> dict:
        """Check whether every ``<|box|>`` tag is preceded by a ``<|ref|>`` tag."""
        return validate_ref_box_pairing(text)

    @staticmethod
    def lenient_extract_boxes(text: str) -> List[Tuple[int, int, int, int]]:
        """Parse bounding boxes even when tags are missing or malformed.

        More forgiving than ``extract_boxes`` — extracts every
        ``[[x1,y1,x2,y2]]`` coordinate array regardless of tag order.
        Useful for difficulty grading after SFT, where the model may have
        learned coordinates but not perfect tag syntax.
        """
        return lenient_parse_boxes(text)

    # ── Validation ──────────────────────────────────────────────────────

    @staticmethod
    def validate_syntax(text: str) -> bool:
        """Check if all primitive tags are properly paired."""
        return syntax_valid(text)

    @staticmethod
    def validate_coordinates(
        text: str, image_size: Tuple[int, int]
    ) -> List[str]:
        """Validate that all coordinates lie within image bounds.

        Args:
            text: Text containing primitive tags.
            image_size: (width, height) in pixels.

        Returns:
            List of error messages (empty if all valid).
        """
        errors = []
        w, h = image_size

        boxes = parse_boxes(text)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if any(v < 0 for v in box) or x1 >= w or y1 >= h or x2 > w or y2 > h:
                errors.append(
                    f"Box {i} {box} out of bounds for image size {image_size}"
                )

        points = parse_points(text)
        for i, point in enumerate(points):
            x, y = point
            if x < 0 or y < 0 or x >= w or y >= h:
                errors.append(
                    f"Point {i} {point} out of bounds for image size {image_size}"
                )

        return errors

    @staticmethod
    def check_wall_collision(
        text: str, maze_grid: np.ndarray
    ) -> List[int]:
        """Check if path segments between consecutive points cross walls."""
        points = parse_points(text)
        return check_wall_collision(points, maze_grid)

    @staticmethod
    def check_wall_collision_points(
        points: List[Tuple[int, int]], maze_grid: np.ndarray
    ) -> List[int]:
        """Check wall collisions for already-parsed point lists."""
        return check_wall_collision(points, maze_grid)

    @staticmethod
    def has_backtracking_keywords(text: str) -> bool:
        """Check if text contains backtracking-related keywords."""
        text_lower = text.lower()
        return any(w in text_lower for w in ["backtrack", "dead end", "retreat", "go back"])

    @staticmethod
    def count_tags(text: str) -> dict:
        """Count occurrences of each primitive tag."""
        return {
            "box_open": text.count(BOX_OPEN),
            "box_close": text.count(BOX_CLOSE),
            "point_open": text.count(POINT_OPEN),
            "point_close": text.count(POINT_CLOSE),
        }

    # ── Formatting ──────────────────────────────────────────────────────

    @staticmethod
    def format_box(coords: List[Tuple[int, int, int, int]]) -> str:
        """Format bounding box(es) as a visual primitive tag."""
        return format_box(coords)

    @staticmethod
    def format_point(coords: List[Tuple[int, int]]) -> str:
        """Format point(s) as a visual primitive tag."""
        return format_point(coords)

    @staticmethod
    def format_ref(name: str) -> str:
        """Format a reference tag: ``<|ref|>name<|/ref|>``."""
        return format_ref(name)

    @staticmethod
    def clean_primitive_tags(text: str, task_type: str = "box") -> str:
        """Clean and normalize visual primitive tags in generated text."""
        return clean_primitive_tags(text, task_type)

    @staticmethod
    def normalize_coordinate(val: float, max_val: float = 1000.0) -> int:
        """Normalize coordinate to 0-999 integer range."""
        return normalize_coordinate(val, max_val)

    @staticmethod
    def denormalize_coordinate(val: int, max_val: float = 1000.0) -> float:
        """Convert normalized 0-999 coordinate back to pixel value."""
        return denormalize_coordinate(val, max_val)

    # ── Geometry ────────────────────────────────────────────────────────

    @staticmethod
    def box_iou(box_a: Tuple[int, ...], box_b: Tuple[int, ...]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        return box_iou(box_a, box_b)

    @staticmethod
    def match_boxes(
        pred_boxes: List[Tuple[int, ...]],
        gt_boxes: List[Tuple[int, ...]],
        iou_threshold: float = 0.5,
    ) -> Tuple[float, int, int]:
        """Match predicted boxes to GT boxes via greedy IoU matching.

        Returns:
            (average_iou, num_matches, num_gt)
        """
        return match_boxes(pred_boxes, gt_boxes, iou_threshold)

    @staticmethod
    def point_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """L2 distance between two points."""
        return point_distance(p1, p2)

    @staticmethod
    def match_points(
        pred_points: List[Tuple[int, int]],
        gt_points: List[Tuple[int, int]],
        dist_threshold: float = 10.0,
    ) -> Tuple[float, int, int]:
        """Match predicted points to GT points via greedy L2 matching.

        Returns:
            (average_distance, num_matches, num_gt)
        """
        return match_points(pred_points, gt_points, dist_threshold)

    @staticmethod
    def maze_causal_exploration_progress(
        pred_points: List[Tuple[int, int]],
        maze_grid: np.ndarray,
        start: Tuple[int, int] | None = None,
        end: Tuple[int, int] | None = None,
    ) -> float:
        """Causal exploration progress for solvable mazes (score 0-1)."""
        return maze_causal_exploration_progress(pred_points, maze_grid, start, end)

    @staticmethod
    def maze_exploration_completeness(
        pred_points: List[Tuple[int, int]],
        maze_grid: np.ndarray,
    ) -> float:
        """Exploration completeness for unsolvable mazes (score 0-1)."""
        return maze_exploration_completeness(pred_points, maze_grid)

    @staticmethod
    def maze_wall_violation_penalty(
        pred_points: List[Tuple[int, int]],
        maze_grid: np.ndarray,
    ) -> float:
        """Global wall violation penalty (0-1)."""
        return maze_wall_violation_penalty(pred_points, maze_grid)

    @staticmethod
    def maze_path_validity(
        pred_points: List[Tuple[int, int]],
        maze_grid: np.ndarray,
    ) -> float:
        """Final path validity for solvable mazes (score 0-1)."""
        return maze_path_validity(pred_points, maze_grid)

    @staticmethod
    def maze_answer_correctness(pred_text: str, gt_text: str) -> float:
        """Binary check: did the model correctly determine solvability?"""
        return maze_answer_correctness(pred_text, gt_text)

    @staticmethod
    def check_backtracking_missing(pred_text: str, gt_text: str) -> bool:
        """Check if model failed to backtrack when GT required it."""
        return check_backtracking_missing(pred_text, gt_text)

    @staticmethod
    def has_duplicate_coords(
        coords: List[Tuple[int, ...]], tolerance: int = 3
    ) -> bool:
        """Check if any two coordinates are nearly identical."""
        return _has_duplicate_coords(coords, tolerance)

    @staticmethod
    def count_repeated_coordinates(
        coords: List[Tuple[int, ...]], tolerance: int = 3
    ) -> int:
        """Count coordinate clusters that appear multiple times."""
        return _count_repeated_coordinates(coords, tolerance)

    # ── Path Tracing RM ───────────────────────────────────────────────

    @staticmethod
    def point_to_polyline_distance(
        point: Tuple[float, float],
        polyline: List[Tuple[float, float]],
    ) -> float:
        """Minimum distance from *point* to any segment of *polyline*."""
        return point_to_polyline_distance(point, polyline)

    @staticmethod
    def path_forward_accuracy(
        pred_points: List[Tuple[int, int]],
        gt_curve: List[Tuple[int, int]],
        max_dist: float = 200.0,
    ) -> float:
        """Forward trajectory accuracy: predicted points → GT curve."""
        return path_forward_accuracy(pred_points, gt_curve, max_dist)

    @staticmethod
    def path_reverse_accuracy(
        gt_curve: List[Tuple[int, int]],
        pred_points: List[Tuple[int, int]],
        max_dist: float = 200.0,
    ) -> float:
        """Reverse trajectory accuracy: GT curve → predicted polyline."""
        return path_reverse_accuracy(gt_curve, pred_points, max_dist)

    @staticmethod
    def path_endpoint_accuracy(
        pred_start,
        pred_end,
        gt_start: Tuple[int, int],
        gt_end: Tuple[int, int],
        tolerance: float = 50.0,
    ) -> float:
        """Endpoint accuracy with linear distance decay."""
        return path_endpoint_accuracy(pred_start, pred_end, gt_start, gt_end, tolerance)

    @staticmethod
    def path_continuity_penalty(
        pred_trajectory_last_point,
        pred_endpoint,
        threshold: float = 80.0,
        penalty: float = 0.1,
    ) -> float:
        """Trajectory continuity penalty: detect endpoint jumping."""
        return path_continuity_penalty(
            pred_trajectory_last_point, pred_endpoint, threshold, penalty
        )
