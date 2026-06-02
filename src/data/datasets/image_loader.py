"""Lazy image loading helper for datasets.

Avoids loading all images into RAM at once by loading from disk
on each __getitem__ call."""

from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def load_image(image_ref: Any) -> Any:
    """Resolve image reference to a PIL Image.

    Supports:
        - PIL.Image.Image (passthrough)
        - str / Path (Image.open from disk)
        - None (passthrough)
    """
    if image_ref is None:
        return None
    if Image is not None and isinstance(image_ref, Image.Image):
        return image_ref
    if isinstance(image_ref, (str, Path)):
        return Image.open(image_ref).convert("RGB")
    return image_ref
