"""Format visual primitive tags for training data."""

import re
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


# Known bad token variants emitted by unstable models.
_BAD_PRIMITIVE_TOKENS = {
    "<|box_start|>": BOX_OPEN,
    "<|box_end|>": BOX_CLOSE,
    "<|point_start|>": POINT_OPEN,
    "<|point_end|>": POINT_CLOSE,
}

# Match a coordinate array surrounded by mismatched / wrong-order tags of the
# same primitive family, e.g. <|/box|>[[...]]<|box|> or <|box|>[[...]]<|box|>.
_TAGGED_COORD_RE = re.compile(
    r"<\|/?([^>|/]+)\|>(\[\[.*?\]\])<\|/?\1\|>", re.DOTALL
)


def clean_primitive_tags(text: str, task_type: str = "box") -> str:
    """Clean and normalize visual primitive tags in generated text.

    Fixes common SFT-data corruption patterns:
      - Reversed open/close tags
      - Duplicate open or duplicate close tags
      - Bad token variants like ``<|box_start|>`` / ``<|box_end|>``

    Args:
        text: The text to clean (usually the ``reasoning`` field).
        task_type: "box" or "point" — determines which tag family to use.

    Returns:
        Cleaned text with properly paired primitive tags.
    """
    for bad, good in _BAD_PRIMITIVE_TOKENS.items():
        text = text.replace(bad, good)

    def _fix_tagged(match: re.Match) -> str:
        name = match.group(1).lower()
        coords = match.group(2)
        if "point" in name:
            return f"{POINT_OPEN}{coords}{POINT_CLOSE}"
        return f"{BOX_OPEN}{coords}{BOX_CLOSE}"

    text = _TAGGED_COORD_RE.sub(_fix_tagged, text)
    return text


def normalize_coordinate(val: float, max_val: float = 1000.0) -> int:
    """Normalize coordinate to 0-999 integer range."""
    return int(min(max(val / max_val * 999, 0), 999))


def denormalize_coordinate(val: int, max_val: float = 1000.0) -> float:
    """Convert normalized 0-999 coordinate back to pixel value."""
    return val / 999.0 * max_val
