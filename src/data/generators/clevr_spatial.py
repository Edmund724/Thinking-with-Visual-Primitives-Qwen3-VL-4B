"""Lightweight CLEVR-style spatial reasoning / VQA generator.

Generates synthetic 2D scenes with colored shapes and programmatically
answerable multi-hop questions. This is a simplified implementation of the
paper's CLEVR-based cold-start data for spatial reasoning (Sec 2.4.2).

Shapes: sphere (circle), cube (square), cylinder (vertical rectangle).
Colors: red, blue, green, yellow, purple, brown.
Sizes: small, large.
Materials: metal, rubber, matte (visualized with simple cues).
Question types: existence, counting, spatial existence, spatial count,
attribute query (color/shape/size/material), compare-integer, multi-hop.
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ...utils.thinking_verifier import filter_verified_samples
from ...models.visual_primitive_parser import PrimitiveParser


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
MATERIALS = ["metal", "rubber", "matte"]


class SceneObject:
    """One object in a synthetic scene."""

    def __init__(
        self,
        obj_id: int,
        shape: str,
        color: str,
        size: str,
        material: str,
        x: int,
        y: int,
        pixel_size: int,
    ):
        self.obj_id = obj_id
        self.shape = shape
        self.color = color
        self.size = size
        self.material = material
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
            PrimitiveParser.normalize_coordinate(x1, img_w),
            PrimitiveParser.normalize_coordinate(y1, img_h),
            PrimitiveParser.normalize_coordinate(x2, img_w),
            PrimitiveParser.normalize_coordinate(y2, img_h),
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
        material = rng.choice(MATERIALS)
        pixel_size = rng.randint(28, 40) if size == "large" else rng.randint(16, 24)
        half = pixel_size // 2
        x = rng.randint(half + 5, img_w - half - 5)
        y = rng.randint(half + 5, img_h - half - 5)
        obj = SceneObject(obj_id, shape, color, size, material, x, y, pixel_size)
        if not any(_shapes_overlap(obj, other) for other in existing):
            return obj
    return None


def _augment_image(
    img: Image.Image,
    rng: random.Random,
    color_jitter: bool = True,
    occlusion: bool = True,
) -> Image.Image:
    """Apply mild offline augmentations that do not change object semantics.

    - Color jitter: brightness/contrast only (hue is preserved so color names
      in the questions remain valid).
    - Occlusion: a few small grey/white rectangles simulating partial occlusion.
    """
    if color_jitter:
        from PIL import ImageEnhance
        brightness_factor = rng.uniform(0.85, 1.15)
        contrast_factor = rng.uniform(0.85, 1.15)
        img = ImageEnhance.Brightness(img).enhance(brightness_factor)
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    if occlusion:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        num_occlusions = rng.randint(0, 2)
        for _ in range(num_occlusions):
            ow = rng.randint(w // 12, w // 6)
            oh = rng.randint(h // 12, h // 6)
            ox = rng.randint(0, max(0, w - ow))
            oy = rng.randint(0, max(0, h - oh))
            color = rng.choice(["#DDDDDD", "#EEEEEE", "#CCCCCC"])
            draw.rectangle([ox, oy, ox + ow, oy + oh], fill=color)

    return img


def generate_scene(
    img_size: Tuple[int, int] = (384, 384),
    num_objects: int = 6,
    seed: int = 0,
    augment: bool = False,
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

        # Material cue: metal gets a white highlight, matte gets a thicker border,
        # rubber is left plain.
        if obj.material == "metal":
            hl_size = max(4, (x2 - x1) // 5)
            hx1 = x1 + hl_size
            hy1 = y1 + hl_size // 2
            hx2 = min(x2 - 2, hx1 + hl_size)
            hy2 = min(y2 - 2, hy1 + hl_size // 2)
            if obj.shape == "sphere":
                draw.ellipse([hx1, hy1, hx2, hy2], fill="white")
            else:
                draw.ellipse([hx1, hy1, hx2, hy2], fill="white")
        elif obj.material == "matte":
            draw.rectangle([x1, y1, x2, y2], outline="#555555", width=2)

    if augment:
        img = _augment_image(img, rng, color_jitter=True, occlusion=True)

    return img, objects


def _object_description(
    obj: SceneObject,
    with_color: bool = True,
    with_size: bool = False,
    with_material: bool = False,
) -> str:
    parts = []
    if with_size:
        parts.append(obj.size)
    if with_material:
        parts.append(obj.material)
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
    material: str | None = None,
) -> List[SceneObject]:
    result = objects[:]
    if color is not None:
        result = [o for o in result if o.color == color]
    if shape is not None:
        result = [o for o in result if o.shape == shape]
    if size is not None:
        result = [o for o in result if o.size == size]
    if material is not None:
        result = [o for o in result if o.material == material]
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
    grounding_parts = [f"{PrimitiveParser.format_ref(f'{shape}s')}{PrimitiveParser.format_box(bboxes)}"]
    summarization = f"There are {count} matching objects in total."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": str(count),
        "question_type": "counting",
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
        f"{PrimitiveParser.format_ref(desc_a)}{PrimitiveParser.format_box([a.normalized_bbox(img_w, img_h)])}",
        f"{PrimitiveParser.format_ref(desc_b)}{PrimitiveParser.format_box([b.normalized_bbox(img_w, img_h)])}",
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
        "question_type": "spatial_existence",
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
    grounding_parts = [f"{PrimitiveParser.format_ref('objects')}{PrimitiveParser.format_box(bboxes)}"]
    summarization = f"There are {count} objects {direction} the {anchor_desc}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": str(count),
        "question_type": "spatial_count",
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
        f"{PrimitiveParser.format_ref(f'{direction} {shape}')}{PrimitiveParser.format_box([target.normalized_bbox(img_w, img_h)])}"
    ]
    summarization = f"The {direction} {shape} is {target.color}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": f"\\boxed{{{target.color}}}",
        "question_type": "attribute_query",
    }


def _generate_existence_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a simple True/False existence question.

    Examples: 'Is there a red cube in the image?' or 'Is there a small sphere?'
    """
    if not objects:
        return None

    # Decide which attributes to query.
    query_color = rng.random() < 0.6
    query_shape = rng.random() < 0.6
    query_size = rng.random() < 0.3
    query_material = rng.random() < 0.2

    color = rng.choice(list(COLORS.keys())) if query_color else None
    shape = rng.choice(SHAPES) if query_shape else None
    size = rng.choice(SIZES) if query_size else None
    material = rng.choice(MATERIALS) if query_material else None

    filtered = _filter_objects(objects, color=color, shape=shape, size=size, material=material)
    exists = len(filtered) > 0

    # Build a natural phrase for the queried object.
    attrs = []
    if size:
        attrs.append(size)
    if material:
        attrs.append(material)
    if color:
        attrs.append(color)
    if shape:
        attrs.append(shape)
    if not attrs:
        return None
    desc = " ".join(attrs)

    prompt = f"Is there a {desc} in the image? Display \\boxed{{True}} or \\boxed{{False}}."
    intent = f"I need to verify whether a {desc} exists in the image."

    if exists:
        target = rng.choice(filtered)
        grounding_parts = [
            f"{PrimitiveParser.format_ref(desc)}{PrimitiveParser.format_box([target.normalized_bbox(img_w, img_h)])}"
        ]
        summarization = f"Yes, there is a {desc} in the image."
    else:
        grounding_parts = ["I scanned the image and found no matching object."]
        summarization = f"No, there is no {desc} in the image."

    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)
    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": r"\boxed{True}" if exists else r"\boxed{False}",
        "question_type": "existence",
    }


def _generate_compare_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a compare-integer question.

    Example: 'Are there more red cubes than blue spheres?'
    """
    if len(objects) < 4:
        return None

    # Sample two attribute combos.
    attr_pool = []
    for _ in range(20):
        color1 = rng.choice(list(COLORS.keys()))
        shape1 = rng.choice(SHAPES)
        color2 = rng.choice(list(COLORS.keys()))
        shape2 = rng.choice(SHAPES)
        if (color1, shape1) != (color2, shape2):
            attr_pool.append(((color1, shape1), (color2, shape2)))
    if not attr_pool:
        return None

    (color1, shape1), (color2, shape2) = rng.choice(attr_pool)
    count1 = len(_filter_objects(objects, color=color1, shape=shape1))
    count2 = len(_filter_objects(objects, color=color2, shape=shape2))

    # Avoid trivial equal cases where possible.
    if count1 == count2:
        return None

    answer_value = count1 > count2
    prompt = (
        f"Are there more {color1} {shape1}s than {color2} {shape2}s? "
        r"Display \boxed{True} or \boxed{False}."
    )
    intent = (
        f"I need to compare the number of {color1} {shape1}s "
        f"with the number of {color2} {shape2}s."
    )

    boxes1 = [o.normalized_bbox(img_w, img_h) for o in _filter_objects(objects, color=color1, shape=shape1)]
    boxes2 = [o.normalized_bbox(img_w, img_h) for o in _filter_objects(objects, color=color2, shape=shape2)]
    grounding_parts = []
    if boxes1:
        grounding_parts.append(f"{PrimitiveParser.format_ref(f'{color1} {shape1}s')}{PrimitiveParser.format_box(boxes1)}")
    if boxes2:
        grounding_parts.append(f"{PrimitiveParser.format_ref(f'{color2} {shape2}s')}{PrimitiveParser.format_box(boxes2)}")
    if not grounding_parts:
        return None

    summarization = (
        f"There are {count1} {color1} {shape1}s and {count2} {color2} {shape2}s, "
        f"so the answer is {'True' if answer_value else 'False'}."
    )
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)
    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": r"\boxed{True}" if answer_value else r"\boxed{False}",
        "question_type": "compare",
    }


def _generate_query_material_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate 'What material is the leftmost red cube?' style question."""
    if not objects:
        return None

    # Filter by color/shape to make the target unambiguous.
    color = rng.choice(list(COLORS.keys()))
    shape = rng.choice(SHAPES)
    filtered = _filter_objects(objects, color=color, shape=shape)
    if not filtered:
        return None

    direction = rng.choice(["leftmost", "rightmost"])
    if direction == "leftmost":
        target = min(filtered, key=lambda o: o.x)
    else:
        target = max(filtered, key=lambda o: o.x)

    prompt = (
        f"What material is the {direction} {color} {shape}? "
        r"Put the material in \boxed{}."
    )
    intent = f"I need to identify the material of the {direction} {color} {shape}."
    grounding_parts = [
        f"{PrimitiveParser.format_ref(f'{direction} {color} {shape}')}"
        f"{PrimitiveParser.format_box([target.normalized_bbox(img_w, img_h)])}"
    ]
    summarization = f"The {direction} {color} {shape} is {target.material}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": f"\\boxed{{{target.material}}}",
        "question_type": "query_material",
    }


def _generate_multihop_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a 2-hop question.

    Example: 'What color is the cube to the left of the red sphere?'
    """
    if len(objects) < 3:
        return None

    # First hop: pick an anchor by color+shape.
    anchor_color = rng.choice(list(COLORS.keys()))
    anchor_shape = rng.choice(SHAPES)
    anchor_candidates = _filter_objects(objects, color=anchor_color, shape=anchor_shape)
    if not anchor_candidates:
        return None
    anchor = rng.choice(anchor_candidates)

    # Second hop: find objects to the left/right of the anchor.
    direction = rng.choice(["left of", "right of"])
    if direction == "left of":
        second_hop = [o for o in objects if o.obj_id != anchor.obj_id and o.x < anchor.x]
    else:
        second_hop = [o for o in objects if o.obj_id != anchor.obj_id and o.x > anchor.x]
    if not second_hop:
        return None

    # Filter second hop by shape for a clean attribute query.
    target_shape = rng.choice(SHAPES)
    targets = [o for o in second_hop if o.shape == target_shape]
    if not targets:
        return None

    # Pick an extreme to make answer unique.
    if direction == "left of":
        target = max(targets, key=lambda o: o.x)
    else:
        target = min(targets, key=lambda o: o.x)

    prompt = (
        f"What color is the {target_shape} {direction} the {anchor_color} {anchor_shape}? "
        r"Put the color in \boxed{}."
    )
    intent = (
        f"I need to find the {target_shape} {direction} the {anchor_color} {anchor_shape} "
        f"and report its color."
    )
    grounding_parts = [
        f"{PrimitiveParser.format_ref(f'{anchor_color} {anchor_shape}')}{PrimitiveParser.format_box([anchor.normalized_bbox(img_w, img_h)])}",
        f"{PrimitiveParser.format_ref(f'{target_shape} {direction} of anchor')}{PrimitiveParser.format_box([target.normalized_bbox(img_w, img_h)])}",
    ]
    summarization = f"The {target_shape} {direction} the {anchor_color} {anchor_shape} is {target.color}."
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": f"\\boxed{{{target.color}}}",
        "question_type": "multihop",
    }


def _generate_negative_existence_question(
    objects: List[SceneObject],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> Dict | None:
    """Generate a faithful-refusal negative sample.

    Asks about a color+shape combination that does NOT appear in the scene.
    The model must answer \boxed{False} and output no boxes, mirroring the
    paper's negative sample augmentation (Sec 2.4.2).
    """
    if not objects:
        return None

    present_combos = {(o.color, o.shape) for o in objects}
    absent_combos = [
        (c, s) for c in COLORS.keys() for s in SHAPES
        if (c, s) not in present_combos
    ]
    if not absent_combos:
        return None

    color, shape = rng.choice(absent_combos)
    prompt = (
        f"Is there a {color} {shape} in the image? "
        r"Display \boxed{True} if it exists, otherwise \boxed{False}."
    )
    intent = f"I need to verify whether a {color} {shape} exists in the image."

    # Grounding: enumerate what is present to justify the refusal.
    seen_combos = sorted({(o.color, o.shape) for o in objects})
    grounding_parts = [
        f"I see {PrimitiveParser.format_box([o.normalized_bbox(img_w, img_h)])} for a {o.color} {o.shape}."
        for o in objects[:4]  # keep refusal concise
    ]
    summarization = (
        f"No {color} {shape} appears in the image; only "
        f"{', '.join(f'{c} {s}' for c, s in seen_combos)} are present."
    )
    reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

    return {
        "prompt": prompt,
        "reasoning": reasoning,
        "answer": r"\boxed{False}",
        "question_type": "existence",
    }


def generate_clevr_spatial_dataset(
    n: int = 10000,
    seed: int = 42,
    cache_dir: str = "data/cache/clevr_spatial",
    img_size: Tuple[int, int] = (384, 384),
    min_objects: int = 4,
    max_objects: int = 8,
    negative_ratio: float = 0.25,
    augment: bool = True,
) -> List[Dict]:
    """Generate CLEVR-style spatial reasoning / VQA dataset.

    Args:
        negative_ratio: Fraction of samples that are faithful-refusal negatives
                        (querying a non-existent color+shape combo). Paper Sec 2.4.2.
        augment: If True, apply mild color/contrast/occlusion augmentation to
            each generated scene.

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
    negative_target = int(n * negative_ratio)
    positive_target = n - negative_target
    positive_count = 0
    negative_count = 0
    idx = 0

    while positive_count < positive_target or negative_count < negative_target:
        num_objects = rng.randint(min_objects, max_objects)
        img, objects = generate_scene(
            img_size=img_size,
            num_objects=num_objects,
            seed=seed + idx,
            augment=augment,
        )
        idx += 1
        if len(objects) < min_objects:
            continue

        img_path = os.path.join(cache_dir, f"clevr_{seed}_{idx:06d}.png")
        img.save(img_path)

        img_w, img_h = img_size

        # Decide whether to generate a negative or positive sample.
        need_negative = negative_count < negative_target
        need_positive = positive_count < positive_target
        if need_negative and (not need_positive or rng.random() < negative_ratio):
            sample = _generate_negative_existence_question(objects, img_w, img_h, rng)
            if sample is not None:
                negative_count += 1
        else:
            generators = [
                _generate_counting_question,
                _generate_existence_question,
                _generate_spatial_existence_question,
                _generate_spatial_count_question,
                _generate_attribute_query_question,
                _generate_query_material_question,
                _generate_compare_question,
                _generate_multihop_question,
            ]
            # Weight multi-hop lower because it needs richer scenes.
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.8, 0.5]
            gen_pairs = list(zip(generators, weights))
            rng.shuffle(gen_pairs)
            sample = None
            for gen, _ in gen_pairs:
                sample = gen(objects, img_w, img_h, rng)
                if sample is not None:
                    break
            if sample is not None:
                positive_count += 1

        if sample is None:
            continue

        data.append({
            "image": img_path,
            "prompt": sample["prompt"],
            "reasoning": sample["reasoning"],
            "answer": sample["answer"],
            "task_type": "box",
            "question_type": sample.get("question_type", "unknown"),
        })

    data = filter_verified_samples(data)
    logger = __import__("logging").getLogger(__name__)
    logger.info(
        f"Generated {len(data)} verified CLEVR spatial samples "
        f"(positive={positive_count}, negative={negative_count})"
    )
    return data
