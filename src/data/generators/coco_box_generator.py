"""COCO-based box grounding dataset generator for Stage 1/2."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..formatters.primitive_formatter import format_box, normalize_coordinate

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


def generate_coco_box_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 40000,
    seed: int = 42,
    max_objects_per_image: int = 3,
) -> List[Dict]:
    """Generate box grounding samples from COCO.

    Returns list of dicts with:
        image: PIL.Image
        prompt: str
        thinking: str
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

    # Filter images with annotations
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

        # Lazy loading: store image_path, load PIL Image on demand
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

        # Pick 1-3 categories
        selected_cats = random.sample(
            list(cat_groups.keys()),
            k=min(random.randint(1, max_objects_per_image), len(cat_groups)),
        )

        thinking_parts = []
        total_count = 0

        for cat_id in selected_cats:
            cat_name = cat_map.get(cat_id, "object")
            cat_anns = cat_groups[cat_id][:max_objects_per_image]

            boxes = []
            for ann in cat_anns:
                box = bbox_to_normalized(ann["bbox"], img_w, img_h)
                boxes.append(box)
                total_count += 1

            if boxes:
                thinking_parts.append(
                    f"Found {len(boxes)} {cat_name}(s): {format_box(boxes)}"
                )

        if total_count == 0:
            continue

        thinking = "\n".join(thinking_parts)
        answer = str(total_count)

        # Randomly choose task type
        task_roll = random.random()
        if task_roll < 0.5:
            # Localization task
            cat_name = cat_map.get(selected_cats[0], "object")
            prompt = f"Locate all {cat_name}s in the image and mark each with <|box|>."
        elif task_roll < 0.8:
            # Counting task
            prompt = f"How many objects are there in total? Use <|box|> to mark each one."
        else:
            # Spatial reasoning
            prompt = f"Find all visible objects and describe their locations with <|box|>."

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": thinking,
            "answer": answer,
            "task_type": "box",
        })

    logger.info(f"Generated {len(data)} COCO box samples from {len(valid_img_ids)} images")
    return data


def generate_synthetic_dense_counting(
    n: int = 10000,
    seed: int = 42,
) -> List[Dict]:
    """Generate synthetic dense counting scenes (fish schools, crowds, etc.)."""
    random.seed(seed)
    np.random.seed(seed)

    data = []
    for _ in range(n):
        img_size = (512, 512)
        img = Image.new("RGB", img_size, "#87CEEB")  # Sky blue background
        draw = ImageDraw.Draw(img)

        # Random number of small objects
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

        thinking = f"I can see {num_objects} small objects scattered across the image.\n"
        thinking += f"All objects: {format_box(boxes)}"

        data.append({
            "image": img,
            "prompt": "How many small objects are in this image? Use <|box|> to mark each one.",
            "reasoning": thinking,
            "answer": str(num_objects),
            "task_type": "box",
        })

    return data
