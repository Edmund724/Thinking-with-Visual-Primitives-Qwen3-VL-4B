"""Lightweight CLEVR-style spatial reasoning / VQA generator.

Generates synthetic 2D scenes with colored shapes and programmatically
answerable multi-hop questions. This is a simplified implementation of the
paper's CLEVR-based cold-start data for spatial reasoning (Sec 2.4.2).

Shapes: sphere (circle), cube (square), cylinder (vertical rectangle).
Colors: red, blue, green, yellow, purple, brown.
Question types: counting, spatial existence, spatial count, attribute query.
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ...utils.thinking_verifier import filter_verified_samples
from ..formatters.primitive_formatter import format_box, normalize_coordinate


COLORS = {
    "red": "#FF6B6B",
    "blue": "#4ECDC4",
    "green": "#95E1D3",
    "yellow": "#FFE66D",
    "purple": "#9B59B6",
    "brown": "#8D6E63",
}

SHAPES = ["sphere", "cube", "cylinder"]
SIZES = ["small", "large"]


class SceneObject:
    """One object in a synthetic scene."""

    def __init__(
        self,
        obj_id: int,
        shape: str,
        color: str,
        size: str,
        x: int,
        y: int,
        pixel_size: int,
    ):
        self.obj_id = obj_id
        self.shape = shape
        self.color = color
        self.size = size
        self.x = x
        self.y = y
        self.pixel_size = pixel_size

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bbox in pixel coords."""
        half = self.pixel_size // 2
        return (
            self.x - half,
            self.y - half,
            self.x + half,
            self.y + half,
        )

    def normalized_bbox(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        return (
            normalize_coordinate(x1, img_w),
            normalize_coordinate(y1, img_h),
            normalize_coordinate(x2, img_w),
            normalize_coordinate(y2, img_h),
        )


def _shapes_overlap(a: SceneObject, b: SceneObject, margin: int = 4) -> bool:
    """Check if two objects overlap (with small margin)."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    return not (ax2 + margin < bx1 or bx2 + margin < ax1 or
                ay2 + margin < by1 or by2 + margin < ay1)


def _random_object(obj_id: int, img_size: Tuple[int, int], rng: random.Random,
                   existing: List[SceneObject], max_attempts: int = 50) -> SceneObject | None:
    """Sample a non-overlapping object."""
    img_w, img_h = img_size
    for _ in range(max_attempts):
        shape = rng.choice(SHAPES)
        color = rng.choice(list(COLORS.keys()))
        size = rng.choice(SIZES)
        pixel_size = rng.randint(28, 40) if size == "large" else rng.randint(16, 24)
        half = pixel_size // 2
        x = rng.randint(half + 5, img_w - half - 5)
        y = rng.randint(half + 5, img_h - half - 5)
        obj = SceneObject(obj_id, shape, color, size, x, y, pixel_size)
        if not any(_shapes_overlap(obj, other) for other in existing):
            return obj
    return None


def generate_scene(
    img_size: Tuple[int, int] = (384, 384),
    num_objects: int = 6,
    seed: int = 0,
) -> Tuple[Image.Image, List[SceneObject]]:
    """Generate a synthetic scene image and its objects."""
    rng = random.Random(seed)
    np.random.seed(seed)

    img = Image.new("RGB", img_size, "#F5F5F5")
    draw = ImageDraw.Draw(img)

    objects: List[SceneObject] = []
    for obj_id in range(num_objects):
        obj = _random_object(obj_id, img_size, rng, objects)
        if obj is None:
            continue
        objects.append(obj)

        color = COLORS[obj.color]
        x1, y1, x2, y2 = obj.bbox

        if obj.shape == "sphere":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline="black", width=1)
        elif obj.shape == "cube":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=1)
        elif obj.shape == "cylinder":
            # Vertical rectangle with rounded top/bottom
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black", width=1)
            draw.ellipse([x1, y1 - 4, x2, y1 + 4], fill=color, outline="black", width=1)
            draw.ellipse([x1, y2 - 4, x2, y2 + 4], fill=color, outline="black", width=1)

    return img, objects


def _object_description(obj: SceneObject, with_color: bool = True, with_size: bool = False) -> str:
    parts = []
    if with_size:
        parts.append(obj.size)
    if with_color:
        parts.append(obj.color)
    parts.append(obj.shape)
    return " ".join(parts)


def _left_of(a: SceneObject, b: SceneObject) -> bool:
    return a.x < b.x


def _right_of(a: SceneObject, b: SceneObject) -> bool:
    return a.x > b.x


def _filter_objects(
    objects: List[SceneObject],
    color: str | None = None,
    shape: str | None = None,
    size: str | None = None,
) -> List[SceneObject]:
    result = objects[:]
    if color is not None:
        result = [o for o in result if o.color == color]
    if shape is not None:
        result = [o for o in result if o.shape == shape]
    if size is not None:
        result = [o for o in result if o.size == size]
    return result


def _build_thinking_3step(
    intent: str,
    grounding_parts: List[str],
    summarization: str,
) -> str:
    """Build 3-step thinking content (raw text, no <think> tags)."""
    grounding = " ".join(grounding_parts) if grounding_parts else "No matching objects."
    return (
        f"Intent Analysis: {intent}\n"
        f"Grounding: {grounding}\n"
        f"Summarization: {summarization}"
    )


def _generate_counting_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a counting question."""
    if not objects:
        return None

    # Choose a random attribute to count by
    by_color = rng.random() < 0.5
    if by_color:
        color = rng.choice(list(COLORS.keys()))
        filtered = _filter_objects(objects, color=color)
        if not filtered:
            return None
        prompt = f"How many {color} objects are in the image? Use <|box|> to mark each one."
        intent = f"I need to count all {color} objects in the image."
    else:
        shape = rng.choice(SHAPES)
        filtered = _filter_objects(objects, shape=shape)
        if not filtered:
            return None
        prompt = f"How many {shape}s are in the image? Use <|box|> to mark each one."
        intent = f"I need to count all {shape}s in the image."

    count = len(filtered)
    # Batch grounding for counting-style questions.
    bboxes = [obj.normalized_bbox(img_w, img_h) for obj in filtered]
    grounding_parts = [f"objects are at {format_box(bboxes)}"]
    summarization = f"There are {count} matching objects in total."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": str(count),
    }


def _generate_spatial_existence_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a True/False spatial relation question."""
    if len(objects) < 2:
        return None

    a = rng.choice(objects)
    b = rng.choice(objects)
    if a is b:
        return None

    # Sometimes ask about a real relation, sometimes negate
    relation = rng.choice(["left of", "right of"])
    answer_value = _left_of(a, b) if relation == "left of" else _right_of(a, b)

    # With 30% chance, flip the relation in the prompt and answer
    flip = rng.random() < 0.3
    if flip:
        relation = "right of" if relation == "left of" else "left of"
        answer_value = not answer_value

    desc_a = _object_description(a)
    desc_b = _object_description(b)
    prompt = (
        f"Is there a {desc_a} {relation} a {desc_b}? "
        r"Display \boxed{True} or \boxed{False}."
    )
    intent = f"I need to verify whether a {desc_a} is {relation} a {desc_b}."

    grounding_parts = [
        f"the {desc_a} is at {format_box([a.normalized_bbox(img_w, img_h)])}",
        f"the {desc_b} is at {format_box([b.normalized_bbox(img_w, img_h)])}",
    ]
    summarization = (
        f"Yes, the {desc_a} is {relation} the {desc_b}."
        if answer_value else
        f"No, the {desc_a} is not {relation} the {desc_b}."
    )
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": r"\boxed{True}" if answer_value else r"\boxed{False}",
    }


def _generate_spatial_count_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate 'How many X are to the left/right of Y?' question."""
    if len(objects) < 3:
        return None

    anchor = rng.choice(objects)
    anchor_desc = _object_description(anchor)
    direction = rng.choice(["left of", "right of"])

    if direction == "left of":
        filtered = [o for o in objects if o.obj_id != anchor.obj_id and o.x < anchor.x]
    else:
        filtered = [o for o in objects if o.obj_id != anchor.obj_id and o.x > anchor.x]

    if not filtered:
        return None

    # Optionally filter by color
    if rng.random() < 0.5:
        color = rng.choice(list(COLORS.keys()))
        filtered = [o for o in filtered if o.color == color]
        if not filtered:
            return None
        prompt = (
            f"How many {color} objects are {direction} the {anchor_desc}? "
            "Use <|box|> to mark each one."
        )
        intent = f"I need to count {color} objects {direction} the {anchor_desc}."
    else:
        prompt = (
            f"How many objects are {direction} the {anchor_desc}? "
            "Use <|box|> to mark each one."
        )
        intent = f"I need to count objects {direction} the {anchor_desc}."

    count = len(filtered)
    # Batch grounding for spatial-count questions.
    bboxes = [obj.normalized_bbox(img_w, img_h) for obj in filtered]
    grounding_parts = [f"objects are at {format_box(bboxes)}"]
    summarization = f"There are {count} objects {direction} the {anchor_desc}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": str(count),
    }


def _generate_attribute_query_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate 'What color is the leftmost cube?' style question."""
    if not objects:
        return None

    shape = rng.choice(SHAPES)
    filtered = _filter_objects(objects, shape=shape)
    if not filtered:
        return None

    direction = rng.choice(["leftmost", "rightmost"])
    if direction == "leftmost":
        target = min(filtered, key=lambda o: o.x)
    else:
        target = max(filtered, key=lambda o: o.x)

    prompt = f"What color is the {direction} {shape}? Put the color in \\boxed{{}}."
    intent = f"I need to identify the color of the {direction} {shape}."
    grounding_parts = [
        f"the {direction} {shape} is at {format_box([target.normalized_bbox(img_w, img_h)])}"
    ]
    summarization = f"The {direction} {shape} is {target.color}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": f"\\boxed{{{target.color}}}",
    }


def generate_clevr_spatial_dataset(
    n: int = 10000,
    seed: int = 42,
    cache_dir: str = "data/cache/clevr_spatial",
    img_size: Tuple[int, int] = (384, 384),
    min_objects: int = 4,
    max_objects: int = 8,
) -> List[Dict]:
    """Generate CLEVR-style spatial reasoning / VQA dataset.

    Returns list of dicts with:
        image: str (saved image path)
        prompt: str
        reasoning: str (3-step thinking with boxes)
        answer: str
        task_type: str = "box"
    """
    os.makedirs(cache_dir, exist_ok=True)
    rng = random.Random(seed)

    data = []
    for idx in range(n):
        num_objects = rng.randint(min_objects, max_objects)
        img, objects = generate_scene(
            img_size=img_size,
            num_objects=num_objects,
            seed=seed + idx,
        )
        if len(objects) < min_objects:
            continue

        img_path = os.path.join(cache_dir, f"clevr_{seed}_{idx:06d}.png")
        img.save(img_path)

        img_w, img_h = img_size
        generators = [
            _generate_counting_question,
            _generate_spatial_existence_question,
            _generate_spatial_count_question,
            _generate_attribute_query_question,
        ]
        rng.shuffle(generators)

        sample = None
        for gen in generators:
            sample = gen(objects, img_w, img_h, rng)
            if sample is not None:
                break

        if sample is None:
            continue

        data.append({
            "image": img_path,
            "prompt": sample["prompt"],
            "reasoning": sample["reasoning"],
            "answer": sample["answer"],
            "task_type": "box",
        })

    data = filter_verified_samples(data)
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Generated {len(data)} verified CLEVR spatial samples")
    return data
