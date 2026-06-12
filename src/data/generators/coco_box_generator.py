"""COCO-based box and point grounding dataset generator with 3-step thinking protocol."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..formatters.primitive_formatter import format_box, format_point, normalize_coordinate

logger = logging.getLogger(__name__)


def load_coco_annotations(ann_file: str) -> Dict:
    """Load COCO annotations."""
    with open(ann_file, "r") as f:
        return json.load(f)


def build_category_map(coco_data: Dict) -> Dict[int, str]:
    """Build id-to-name category map."""
    return {cat["id"]: cat["name"] for cat in coco_data["categories"]}


def build_image_annotations(coco_data: Dict) -> Dict[int, List[Dict]]:
    """Group annotations by image_id."""
    image_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_anns:
            image_anns[img_id] = []
        image_anns[img_id].append(ann)
    return image_anns


def bbox_to_normalized(bbox: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert COCO bbox [x, y, w, h] to normalized [x1, y1, x2, y2] in 0-999."""
    x1 = normalize_coordinate(bbox[0], img_w)
    y1 = normalize_coordinate(bbox[1], img_h)
    x2 = normalize_coordinate(bbox[0] + bbox[2], img_w)
    y2 = normalize_coordinate(bbox[1] + bbox[3], img_h)
    return (x1, y1, x2, y2)


def _build_thinking_3step(
    intent: str,
    grounding_parts: List[str],
    summarization: str,
) -> str:
    """Build 3-step thinking content (raw text, no <think> tags).

    The chat template will wrap this in <think>...</think> automatically.
    Format:
        Intent Analysis: {intent}
        Grounding: {grounding}
        Summarization: {summary}
    """
    grounding = " ".join(grounding_parts) if grounding_parts else "No objects detected."
    return (
        f"Intent Analysis: {intent}\n"
        f"Grounding: {grounding}\n"
        f"Summarization: {summarization}"
    )


def generate_coco_box_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 40000,
    seed: int = 42,
    max_objects_per_image: int = 3,
    use_thinking: bool = True,
) -> List[Dict]:
    """Generate box grounding samples from COCO.

    Args:
        use_thinking: If True, wraps reasoning in 3-step thinking protocol
                     (Intent→Grounding→Summarization). Set False for Stage 0.5
                     visual pretrain which needs simple prompt→response mapping.

    Returns list of dicts with:
        image: str (image path)
        prompt: str
        reasoning: str
        answer: str
        task_type: str = "box"
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ann_file):
        logger.warning(f"COCO annotation file not found: {ann_file}. Returning empty list.")
        return []

    coco_data = load_coco_annotations(ann_file)
    cat_map = build_category_map(coco_data)
    image_anns = build_image_annotations(coco_data)
    images = {img["id"]: img for img in coco_data["images"]}

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        logger.warning("No valid images with annotations found.")
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        img_id = random.choice(valid_img_ids)
        img_info = images[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        anns = image_anns[img_id]
        if not anns:
            continue

        # Group by category
        cat_groups = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_groups:
                cat_groups[cat_id] = []
            cat_groups[cat_id].append(ann)

        selected_cats = random.sample(
            list(cat_groups.keys()),
            k=min(random.randint(1, max_objects_per_image), len(cat_groups)),
        )

        grounding_parts = []
        total_count = 0
        all_boxes = []

        for cat_id in selected_cats:
            cat_name = cat_map.get(cat_id, "object")
            cat_anns = cat_groups[cat_id][:max_objects_per_image]

            boxes = []
            for ann in cat_anns:
                box = bbox_to_normalized(ann["bbox"], img_w, img_h)
                boxes.append(box)
                all_boxes.append(box)
                total_count += 1

            if boxes:
                grounding_parts.append(
                    f"the {cat_name} is at {format_box(boxes)}"
                )

        if total_count == 0:
            continue

        # Randomly choose task type
        task_roll = random.random()
        if task_roll < 0.5:
            # Localization task
            cat_name = cat_map.get(selected_cats[0], "object")
            prompt = f"Locate all {cat_name}s in the image and mark each with <|box|>."
            intent = f"I need to locate all {cat_name}s in the image."
            summarization = f"I found {len([b for b in all_boxes])} {cat_name}(s) in the image."
            answer = str(len([b for b in all_boxes]))
        elif task_roll < 0.8:
            # Counting task
            prompt = f"How many objects are there in total? Use <|box|> to mark each one."
            intent = "I need to count all objects visible in the image."
            summarization = f"There are {total_count} objects in total."
            answer = str(total_count)
        else:
            # Spatial reasoning / all objects
            prompt = f"Find all visible objects and describe their locations with <|box|>."
            intent = "I need to find all visible objects in the image."
            summarization = f"I found {total_count} objects across {len(selected_cats)} categories."
            answer = str(total_count)

        if use_thinking:
            reasoning = _build_thinking_3step(intent, grounding_parts, summarization)
        else:
            reasoning = f"<|ref|>{cat_name}<|/ref|>{format_box(all_boxes)}"

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "box",
        })

    logger.info(f"Generated {len(data)} COCO box samples from {len(valid_img_ids)} images")
    return data


def generate_coco_point_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 10000,
    seed: int = 42,
    use_thinking: bool = True,
) -> List[Dict]:
    """Generate point grounding samples from COCO object centers with 3-step thinking.

    Uses COCO bounding box annotations to compute center points.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ann_file):
        logger.warning(f"COCO annotation file not found: {ann_file}. Returning empty list.")
        return []

    coco_data = load_coco_annotations(ann_file)
    cat_map = build_category_map(coco_data)
    image_anns = build_image_annotations(coco_data)
    images = {img["id"]: img for img in coco_data["images"]}

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        logger.warning("No valid images with annotations found.")
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        img_id = random.choice(valid_img_ids)
        img_info = images[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        anns = image_anns[img_id]
        if not anns:
            continue

        # Pick one random annotation
        ann = random.choice(anns)
        cat_name = cat_map.get(ann["category_id"], "object")

        # Compute center point from bbox
        x, y, w, h = ann["bbox"]
        cx = normalize_coordinate(x + w / 2, img_w)
        cy = normalize_coordinate(y + h / 2, img_h)

        prompt = f"Point to the {cat_name} in this image."
        intent = f"I need to locate the {cat_name} in the image."
        grounding = f"the {cat_name} is centered at {format_point([(cx, cy)])}"
        summarization = f"The {cat_name} is located at coordinates ({cx}, {cy})."
        answer = f"({cx}, {cy})"

        if use_thinking:
            reasoning = _build_thinking_3step(intent, [grounding], summarization)
        else:
            reasoning = f"<|point|>{format_point([(cx, cy)])}<|/point|>"

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "point",
        })

    logger.info(f"Generated {len(data)} COCO point samples from {len(valid_img_ids)} images")
    return data


def generate_synthetic_dense_counting(
    n: int = 10000,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic dense counting scenes with 3-step thinking protocol."""
    random.seed(seed)
    np.random.seed(seed)

    data = []
    for _ in range(n):
        img_size = (512, 512)
        img = Image.new("RGB", img_size, "#87CEEB")
        draw = ImageDraw.Draw(img)

        num_objects = random.randint(15, 60)
        boxes = []

        for _ in range(num_objects):
            x = random.randint(20, img_size[0] - 40)
            y = random.randint(20, img_size[1] - 40)
            size = random.randint(8, 16)
            color = random.choice(["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3"])
            draw.ellipse([x, y, x + size, y + size], fill=color)
            boxes.append((
                normalize_coordinate(x, img_size[0]),
                normalize_coordinate(y, img_size[1]),
                normalize_coordinate(x + size, img_size[0]),
                normalize_coordinate(y + size, img_size[1]),
            ))

        intent = "I need to count all small objects in the image."
        grounding_parts = [f"object at {format_box([box])}" for box in boxes]
        # Limit grounding parts to avoid too long thinking
        if len(grounding_parts) > 10:
            grounding_parts = grounding_parts[:5] + [f"... and {len(grounding_parts) - 5} more objects"]
        summarization = f"There are {num_objects} small objects in total."
        reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

        data.append({
            "image": img,
            "prompt": "How many small objects are in this image? Use <|box|> to mark each one.",
            "reasoning": reasoning,
            "answer": str(num_objects),
            "task_type": "box",
        })

    return data
