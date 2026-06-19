"""COCO-based box and point grounding dataset generator with 3-step thinking protocol."""

import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ...models.visual_primitive_parser import PrimitiveParser
from ...utils.thinking_verifier import filter_verified_samples

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
    x1 = PrimitiveParser.normalize_coordinate(bbox[0], img_w)
    y1 = PrimitiveParser.normalize_coordinate(bbox[1], img_h)
    x2 = PrimitiveParser.normalize_coordinate(bbox[0] + bbox[2], img_w)
    y2 = PrimitiveParser.normalize_coordinate(bbox[1] + bbox[3], img_h)
    return (x1, y1, x2, y2)


def _filter_annotations_by_geometry(
    anns: List[Dict],
    img_w: int,
    img_h: int,
    max_area_ratio: float = 0.9,
    min_area_ratio: float = 0.0001,
    min_edge_pixels: int = 2,
) -> List[Dict]:
    """Filter out geometrically bad COCO annotations.

    Rules (light-weight, suitable for small-scale data):
      - Mega boxes: area > 90% of image area (often classification data
        forced into detection format).
      - Tiny boxes: area < 0.01% of image area (not visually meaningful).
      - Degenerate boxes: width/height < 2 px.
      - Strongly edge-touching boxes: box lies almost entirely on the image
        border (touches two or more edges and spans >80% of that edge).
    """
    img_area = img_w * img_h
    if img_area <= 0:
        return []

    filtered = []
    for ann in anns:
        x, y, w, h = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if w <= 0 or h <= 0:
            continue

        area = w * h
        area_ratio = area / img_area
        if area_ratio > max_area_ratio:
            continue
        if area_ratio < min_area_ratio:
            continue
        if w < min_edge_pixels or h < min_edge_pixels:
            continue

        # Count how many edges the box touches
        touches_left = x <= 0
        touches_top = y <= 0
        touches_right = (x + w) >= img_w
        touches_bottom = (y + h) >= img_h
        touch_count = sum([touches_left, touches_top, touches_right, touches_bottom])

        # If it touches two or more edges and spans most of the image, drop it
        if touch_count >= 2 and (w / img_w > 0.8 or h / img_h > 0.8):
            continue

        filtered.append(ann)

    return filtered


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

        # Geometric quality filtering
        anns = _filter_annotations_by_geometry(anns, img_w, img_h)
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
                    f"the {cat_name} is at {PrimitiveParser.format_box(boxes)}"
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
            # Stage 2 visual pretrain: use a short natural-language thinking
            # sentence that naturally introduces the primitive tags, matching
            # the unified format from the paper.
            if task_roll < 0.5:
                nl_prefix = f"I need to locate all {cat_name}s. "
            elif task_roll < 0.8:
                nl_prefix = "I need to count all visible objects. "
            else:
                nl_prefix = "I need to find all visible objects. "
            grounding_sentence = " ".join(grounding_parts) if grounding_parts else ""
            reasoning = (
                f"{nl_prefix}"
                f"{grounding_sentence}"
            ).strip()

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "box",
        })

    data = filter_verified_samples(data, logger=logger)
    logger.info(f"Generated {len(data)} verified COCO box samples from {len(valid_img_ids)} images")
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

        # Geometric quality filtering
        anns = _filter_annotations_by_geometry(anns, img_w, img_h)
        if not anns:
            continue

        # Pick one random annotation
        ann = random.choice(anns)
        cat_name = cat_map.get(ann["category_id"], "object")

        # Compute center point from bbox
        x, y, w, h = ann["bbox"]
        cx = PrimitiveParser.normalize_coordinate(x + w / 2, img_w)
        cy = PrimitiveParser.normalize_coordinate(y + h / 2, img_h)

        prompt = f"Point to the {cat_name} in this image."
        intent = f"I need to locate the {cat_name} in the image."
        grounding = f"the {cat_name} is centered at {PrimitiveParser.format_point([(cx, cy)])}"
        summarization = f"The {cat_name} is located at coordinates ({cx}, {cy})."
        answer = f"({cx}, {cy})"

        if use_thinking:
            reasoning = _build_thinking_3step(intent, [grounding], summarization)
        else:
            # Stage 2 visual pretrain: use a short natural-language thinking
            # sentence that naturally introduces the primitive tags.
            reasoning = f"I need to locate the {cat_name}. {grounding}"

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "point",
        })

    data = filter_verified_samples(data, logger=logger)
    logger.info(f"Generated {len(data)} verified COCO point samples from {len(valid_img_ids)} images")
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
                PrimitiveParser.normalize_coordinate(x, img_size[0]),
                PrimitiveParser.normalize_coordinate(y, img_size[1]),
                PrimitiveParser.normalize_coordinate(x + size, img_size[0]),
                PrimitiveParser.normalize_coordinate(y + size, img_size[1]),
            ))

        intent = "I need to count all small objects in the image."
        # Batch grounding: all objects in one tag to match coarse-grained counting.
        grounding_parts = [f"objects are at {PrimitiveParser.format_box(boxes)}"]
        summarization = f"There are {num_objects} small objects in total."
        reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

        data.append({
            "image": img,
            "prompt": "How many small objects are in this image? Use <|box|> to mark each one.",
            "reasoning": reasoning,
            "answer": str(num_objects),
            "task_type": "box",
        })

    data = filter_verified_samples(data, logger=logger)
    return data



def _bbox_area(ann: Dict, img_w: int, img_h: int) -> float:
    """Return bbox area as a ratio of image area."""
    x, y, w, h = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / (img_w * img_h)


def _size_label(area_ratio: float) -> str:
    """Map area ratio to a coarse size label."""
    if area_ratio > 0.15:
        return "large"
    if area_ratio > 0.05:
        return "medium"
    return "small"


def _dominant_color_name(image_path: str, bbox: List[float]) -> str | None:
    """Extract a coarse dominant color name from a bbox region.

    Uses simple RGB centroid and maps to basic color names. Returns None if
    the image cannot be loaded.
    """
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        x, y, w, h = bbox
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img.crop((x1, y1, x2, y2))
        # Downsample for speed
        crop = crop.resize((32, 32))
        arr = np.array(crop).reshape(-1, 3)
        mean = arr.mean(axis=0)
        r, g, b = mean

        # Simple hue-ish mapping
        if r > g and r > b and abs(r - g) > 20 and abs(r - b) > 20:
            return "red"
        if g > r and g > b and abs(g - r) > 20:
            return "green"
        if b > r and b > g and abs(b - r) > 20:
            return "blue"
        if r > 180 and g > 180 and b < 120:
            return "yellow"
        if r > 120 and g > 80 and b > 120:
            return "purple"
        if r > 120 and g > 100 and b > 80 and abs(r - g) < 30 and abs(r - b) < 40:
            return "brown"
        if r > 180 and g > 180 and b > 180:
            return "white"
        if r < 60 and g < 60 and b < 60:
            return "black"
        return None
    except Exception:
        return None


def generate_coco_counting_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 20000,
    seed: int = 42,
    min_instances: int = 3,
    max_instances: int = 30,
    target_categories: List[str] | None = None,
    attribute_constraint_ratio: float = 0.0,
) -> List[Dict]:
    """Generate coarse-grained counting samples from COCO (paper Sec 2.4.1).

    For each sample:
      1. Pick an image and a target category that has >= min_instances boxes.
      2. Prompt: "How many {category}s are in this image?"
      3. Thinking: Intent Analysis → Batch Grounding (all boxes) → Summarization.
      4. Answer: integer count.

    Args:
        attribute_constraint_ratio: Fraction of counting samples that add a
            color or size attribute constraint (e.g. 'How many red cars?').

    This matches the paper's coarse-grained counting protocol: batch grounding
    all candidate objects simultaneously, then tally.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ann_file):
        logger.warning(f"COCO annotation file not found: {ann_file}. Returning empty list.")
        return []

    coco_data = load_coco_annotations(ann_file)
    cat_map = build_category_map(coco_data)
    name_to_cat = {name: cid for cid, name in cat_map.items()}
    image_anns = build_image_annotations(coco_data)
    images = {img["id"]: img for img in coco_data["images"]}

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        logger.warning("No valid images with annotations found.")
        return []

    if target_categories is None:
        # COCO categories commonly suitable for dense counting
        target_categories = [
            "person", "car", "chair", "book", "bottle", "cup",
            "orange", "apple", "bird", "cat", "dog",
        ]

    target_cat_ids = [
        name_to_cat[name] for name in target_categories if name in name_to_cat
    ]
    if not target_cat_ids:
        logger.warning("None of the target categories exist in COCO.")
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 10

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

        # Geometric quality filtering
        anns = _filter_annotations_by_geometry(anns, img_w, img_h)
        if not anns:
            continue

        # Group by category and keep only target categories with enough instances
        cat_groups = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in target_cat_ids:
                continue
            cat_groups.setdefault(cat_id, []).append(ann)

        dense_cats = [
            cid for cid, anns in cat_groups.items()
            if min_instances <= len(anns) <= max_instances
        ]
        if not dense_cats:
            continue

        cat_id = random.choice(dense_cats)
        cat_name = cat_map.get(cat_id, "object")
        cat_anns = cat_groups[cat_id]

        # Optional attribute constraint.
        attr_type = None
        attr_value = None
        if random.random() < attribute_constraint_ratio:
            attr_type = random.choice(["color", "size"])
            if attr_type == "color":
                # Filter to a dominant color present in at least 2 boxes.
                color_counts = {}
                color_boxes = {}
                for ann in cat_anns:
                    color = _dominant_color_name(img_path, ann["bbox"])
                    if color:
                        color_counts[color] = color_counts.get(color, 0) + 1
                        color_boxes.setdefault(color, []).append(ann)
                valid_colors = [c for c, n in color_counts.items() if n >= 2]
                if valid_colors:
                    attr_value = random.choice(valid_colors)
                    cat_anns = color_boxes[attr_value]
            else:
                # Filter to a size label present in at least 2 boxes.
                size_counts = {}
                size_boxes = {}
                for ann in cat_anns:
                    area_ratio = _bbox_area(ann, img_w, img_h)
                    size = _size_label(area_ratio)
                    size_counts[size] = size_counts.get(size, 0) + 1
                    size_boxes.setdefault(size, []).append(ann)
                valid_sizes = [s for s, n in size_counts.items() if n >= 2]
                if valid_sizes:
                    attr_value = random.choice(valid_sizes)
                    cat_anns = size_boxes[attr_value]

        boxes = [bbox_to_normalized(ann["bbox"], img_w, img_h) for ann in cat_anns]
        count = len(boxes)

        # Batch grounding: all target boxes in a single visual primitive tag.
        grounding_parts = [f"the {cat_name}s are at {PrimitiveParser.format_box(boxes)}"]

        if attr_type == "color":
            prompt = (
                f"How many {attr_value} {cat_name}s are in this image? "
                "Use <|box|> to mark each one."
            )
            intent = f"I need to count all {attr_value} {cat_name}s in the image."
            summarization = f"There are {count} {attr_value} {cat_name}(s) in total."
        elif attr_type == "size":
            prompt = (
                f"How many {attr_value} {cat_name}s are in this image? "
                "Use <|box|> to mark each one."
            )
            intent = f"I need to count all {attr_value} {cat_name}s in the image."
            summarization = f"There are {count} {attr_value} {cat_name}(s) in total."
        else:
            prompt = f"How many {cat_name}s are in this image? Use <|box|> to mark each one."
            intent = f"I need to count all {cat_name}s in the image."
            summarization = f"There are {count} {cat_name}(s) in total."

        reasoning = _build_thinking_3step(intent, grounding_parts, summarization)

        answer = str(count)

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "box",
        })

    data = filter_verified_samples(data, logger=logger)
    logger.info(
        f"Generated {len(data)} verified coarse-grained counting samples "
        f"from {len(valid_img_ids)} images"
    )
    return data


def generate_coco_negative_box_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 2000,
    seed: int = 45,
    candidate_categories: List[str] | None = None,
) -> List[Dict]:
    """Generate faithful-refusal negative box samples.

    For each sample, pick an image and a category that is NOT present in it,
    then ask the model to locate/count that category. The correct behavior is
    to answer \boxed{0} / \boxed{False} and output no boxes, matching the
    paper's negative sample augmentation (Sec 2.4.2).
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ann_file):
        logger.warning(f"COCO annotation file not found: {ann_file}. Returning empty list.")
        return []

    coco_data = load_coco_annotations(ann_file)
    cat_map = build_category_map(coco_data)
    name_to_cat = {name: cid for cid, name in cat_map.items()}
    image_anns = build_image_annotations(coco_data)
    images = {img["id"]: img for img in coco_data["images"]}

    if candidate_categories is None:
        candidate_categories = list(name_to_cat.keys())
    candidate_cat_ids = {name_to_cat[name] for name in candidate_categories if name in name_to_cat}

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        logger.warning("No valid images with annotations found.")
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        img_id = random.choice(valid_img_ids)
        img_info = images[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        present_cat_ids = {ann["category_id"] for ann in image_anns[img_id]}
        absent_cat_ids = candidate_cat_ids - present_cat_ids
        if not absent_cat_ids:
            continue

        cat_id = random.choice(list(absent_cat_ids))
        cat_name = cat_map.get(cat_id, "object")

        # Randomly choose localization or counting refusal.
        if random.random() < 0.5:
            prompt = f"Locate all {cat_name}s in the image and mark each with <|box|>."
            answer = r"\boxed{False}"
            summarization = f"There are no {cat_name}s visible in the image."
        else:
            prompt = f"How many {cat_name}s are in this image? Use <|box|> to mark each one."
            answer = r"\boxed{0}"
            summarization = f"There are 0 {cat_name}(s) in the image."

        intent = f"I need to check whether any {cat_name}s appear in the image."
        reasoning = _build_thinking_3step(
            intent,
            ["I scanned the image and found no matching objects."],
            summarization,
        )

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "box",
        })

    data = filter_verified_samples(data, logger=logger)
    logger.info(f"Generated {len(data)} verified COCO negative box samples")
    return data


def generate_coco_negative_point_samples(
    image_dir: str,
    ann_file: str,
    num_samples: int = 1000,
    seed: int = 47,
    candidate_categories: List[str] | None = None,
) -> List[Dict]:
    """Generate faithful-refusal negative point samples.

    Asks the model to point to a category that is NOT present in the image.
    Correct behavior: answer \boxed{False} / 'not found' and output no points.
    """
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(ann_file):
        logger.warning(f"COCO annotation file not found: {ann_file}. Returning empty list.")
        return []

    coco_data = load_coco_annotations(ann_file)
    cat_map = build_category_map(coco_data)
    name_to_cat = {name: cid for cid, name in cat_map.items()}
    image_anns = build_image_annotations(coco_data)
    images = {img["id"]: img for img in coco_data["images"]}

    if candidate_categories is None:
        candidate_categories = list(name_to_cat.keys())
    candidate_cat_ids = {name_to_cat[name] for name in candidate_categories if name in name_to_cat}

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        logger.warning("No valid images with annotations found.")
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        img_id = random.choice(valid_img_ids)
        img_info = images[img_id]

        img_path = os.path.join(image_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        present_cat_ids = {ann["category_id"] for ann in image_anns[img_id]}
        absent_cat_ids = candidate_cat_ids - present_cat_ids
        if not absent_cat_ids:
            continue

        cat_id = random.choice(list(absent_cat_ids))
        cat_name = cat_map.get(cat_id, "object")

        prompt = f"Point to the {cat_name} in this image."
        intent = f"I need to locate the {cat_name} in the image."
        reasoning = _build_thinking_3step(
            intent,
            ["I scanned the image and found no matching object."],
            f"There is no {cat_name} visible in the image.",
        )

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": r"\boxed{False}",
            "task_type": "point",
        })

    data = filter_verified_samples(data, logger=logger)
    logger.info(f"Generated {len(data)} verified COCO negative point samples")
    return data
