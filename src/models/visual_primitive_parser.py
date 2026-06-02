"""Visual Primitive Parser — unified interface for parsing and validating
visual primitive tokens (box / point) in model-generated text.

This module wraps the lower-level regex and geometry utilities from
src.utils.metrics into a single PrimitiveParser class for convenience.
"""

import re
from typing import List, Tuple

import numpy as np

from ..utils.constants import (
    BOX_CLOSE,
    BOX_OPEN,
    POINT_CLOSE,
    POINT_OPEN,
)
from ..utils.metrics import (
    check_backtracking_missing,
    check_wall_collision,
    parse_boxes,
    parse_points,
    syntax_valid,
)


class PrimitiveParser:
    """Parser for visual primitive tags in generated text."""

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
        """Check if path segments between consecutive points cross walls.

        Args:
            text: Text containing point tags.
            maze_grid: Binary grid where 0=wall, 1=path.

        Returns:
            List of step indices where collision occurs.
        """
        points = parse_points(text)
        return check_wall_collision(points, maze_grid)

    @staticmethod
    def count_tags(text: str) -> dict:
        """Count occurrences of each primitive tag."""
        return {
            "box_open": text.count(BOX_OPEN),
            "box_close": text.count(BOX_CLOSE),
            "point_open": text.count(POINT_OPEN),
            "point_close": text.count(POINT_CLOSE),
        }

    @staticmethod
    def has_backtracking_keywords(text: str) -> bool:
        """Check if text contains backtracking-related keywords."""
        text_lower = text.lower()
        return any(w in text_lower for w in ["backtrack", "dead end", "retreat", "go back"])
