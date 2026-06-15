"""Format visual primitive tags for training data."""

from typing import List, Tuple

from ...utils.constants import BOX_CLOSE, BOX_OPEN, POINT_CLOSE, POINT_OPEN


def format_box(coords: List[Tuple[int, int, int, int]]) -> str:
    """Format bounding box(es) as visual primitive tag.

    Args:
        coords: List of (x1, y1, x2, y2) tuples.

    Returns:
        Formatted box tag string. Single box: ``<|box|>[[x1,y1,x2,y2]]<|/box|>``.
        Multiple boxes: ``<|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>``.
    """
    parts = [f"{c[0]},{c[1]},{c[2]},{c[3]}" for c in coords]
    inner = "[[" + "],[".join(parts) + "]]"
    return f"{BOX_OPEN}{inner}{BOX_CLOSE}"


def format_point(coords: List[Tuple[int, int]]) -> str:
    """Format point(s) as visual primitive tag.

    Args:
        coords: List of (x, y) tuples.

    Returns:
        Formatted point tag string. Single point: ``<|point|>[[x,y]]<|/point|>``.
        Multiple points: ``<|point|>[[x1,y1],[x2,y2]]<|/point|>``.
    """
    parts = [f"{c[0]},{c[1]}" for c in coords]
    inner = "[[" + "],[".join(parts) + "]]"
    return f"{POINT_OPEN}{inner}{POINT_CLOSE}"


def normalize_coordinate(val: float, max_val: float = 1000.0) -> int:
    """Normalize coordinate to 0-999 integer range."""
    return int(min(max(val / max_val * 999, 0), 999))


def denormalize_coordinate(val: int, max_val: float = 1000.0) -> float:
    """Convert normalized 0-999 coordinate back to pixel value."""
    return val / 999.0 * max_val
