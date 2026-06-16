#!/usr/bin/env python3
"""Generate pure-text pretrain data for visual primitive token initialization.

No images. Programmatic 20K-30K samples teaching the model:
1. New special token format (<|box|>, <|point|>, etc.)
2. Coordinate normalization [0, 999]
3. Tag pairing (open/close must match)

Output: data/pretrain/pretrain_data.json (conversations format)
"""

import json
import os
import random
from typing import Dict, List, Tuple


def _load_coco_annotations(ann_file: str) -> Dict:
    """Load COCO annotations if available."""
    with open(ann_file, "r") as f:
        return json.load(f)


def _bbox_to_normalized(bbox: List[float], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert COCO bbox [x, y, w, h] to normalized [x1, y1, x2, y2] in 0-999."""
    x1 = max(0, min(999, int(bbox[0] / img_w * 999)))
    y1 = max(0, min(999, int(bbox[1] / img_h * 999)))
    x2 = max(0, min(999, int((bbox[0] + bbox[2]) / img_w * 999)))
    y2 = max(0, min(999, int((bbox[1] + bbox[3]) / img_h * 999)))
    return (x1, y1, x2, y2)


def _center_to_normalized(bbox: List[float], img_w: int, img_h: int) -> Tuple[int, int]:
    """Convert COCO bbox to normalized center point."""
    cx = max(0, min(999, int((bbox[0] + bbox[2] / 2) / img_w * 999)))
    cy = max(0, min(999, int((bbox[1] + bbox[3] / 2) / img_h * 999)))
    return (cx, cy)


def generate_coco_grounding_pretrain_samples(
    ann_file: str,
    num_samples: int = 5000,
    seed: int = 43,
) -> List[dict]:
    """Generate text-only pretrain samples using real COCO annotations.

    Stage 1 is embedding-only and does not process images, but using real
    category names + real bounding box coordinates helps the model learn
    realistic coordinate distributions and vocabulary, moving closer to the
    paper's large-scale grounding pretraining (Sec 2.3).
    """
    if not os.path.exists(ann_file):
        print(f"COCO annotations not found at {ann_file}; skipping COCO grounding mix.")
        return []

    random.seed(seed)
    coco_data = _load_coco_annotations(ann_file)
    cat_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    images = {img["id"]: img for img in coco_data["images"]}

    # Group annotations by image.
    image_anns: Dict[int, List[Dict]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        image_anns.setdefault(img_id, []).append(ann)

    valid_img_ids = [img_id for img_id in images if img_id in image_anns]
    if not valid_img_ids:
        return []

    data = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(data) < num_samples and attempts < max_attempts:
        attempts += 1
        img_id = random.choice(valid_img_ids)
        img_info = images[img_id]
        img_w, img_h = img_info["width"], img_info["height"]
        anns = image_anns[img_id]
        if not anns:
            continue

        # Build category groups for this image.
        cat_groups: Dict[int, List[Dict]] = {}
        for ann in anns:
            cat_groups.setdefault(ann["category_id"], []).append(ann)

        cat_id = random.choice(list(cat_groups.keys()))
        cat_name = cat_map.get(cat_id, "object")
        cat_anns = cat_groups[cat_id]

        task_type = random.random()
        if task_type < 0.5 and len(cat_anns) >= 1:
            # Single box.
            ann = random.choice(cat_anns)
            x1, y1, x2, y2 = _bbox_to_normalized(ann["bbox"], img_w, img_h)
            box_str = f"<|box|>[[{x1},{y1},{x2},{y2}]]<|/box|>"
            user = random.choice(BOX_USER_TEMPLATES).format(obj=cat_name)
            assistant = random.choice(BOX_ASSISTANT_TEMPLATES).format(obj=cat_name, box=box_str)
        elif len(cat_anns) >= 2:
            # Multiple boxes.
            selected = random.sample(cat_anns, k=min(random.randint(2, 5), len(cat_anns)))
            boxes = [_bbox_to_normalized(ann["bbox"], img_w, img_h) for ann in selected]
            inner = "],[".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in boxes)
            box_str = f"<|box|>[[{inner}]]<|/box|>"
            user = random.choice(BOX_MULTI_USER_TEMPLATES).format(obj=f"{cat_name}s")
            assistant = random.choice(BOX_MULTI_ASSISTANT_TEMPLATES).format(
                count=len(boxes), obj=f"{cat_name}s", boxes=box_str
            )
        else:
            # Single point (center of the only box).
            ann = cat_anns[0]
            cx, cy = _center_to_normalized(ann["bbox"], img_w, img_h)
            point_str = f"<|point|>[[{cx},{cy}]]<|/point|>"
            user = random.choice(POINT_USER_TEMPLATES).format(obj=cat_name)
            assistant = random.choice(POINT_ASSISTANT_TEMPLATES).format(obj=cat_name, point=point_str)

        if not _validate_tags(user + " " + assistant):
            continue
        if len(assistant.split()) > 80:
            continue

        data.append({
            "conversations": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })

    return data


def random_box() -> str:
    """Generate a random bounding box in normalized [0, 999] coordinates."""
    x1 = random.randint(0, 900)
    y1 = random.randint(0, 900)
    w = random.randint(20, 300)
    h = random.randint(20, 300)
    x2 = min(x1 + w, 999)
    y2 = min(y1 + h, 999)
    return f"<|box|>[[{x1},{y1},{x2},{y2}]]<|/box|>"


def random_multi_box(num: int) -> str:
    """Generate multiple boxes in one tag."""
    parts = []
    for _ in range(num):
        x1 = random.randint(0, 900)
        y1 = random.randint(0, 900)
        w = random.randint(20, 200)
        h = random.randint(20, 200)
        x2 = min(x1 + w, 999)
        y2 = min(y1 + h, 999)
        parts.append(f"{x1},{y1},{x2},{y2}")
    inner = "[[" + "],[".join(parts) + "]]"
    return f"<|box|>{inner}<|/box|>"


def random_point() -> str:
    """Generate a random point in [0, 999] coordinates."""
    x = random.randint(0, 999)
    y = random.randint(0, 999)
    return f"<|point|>[[{x},{y}]]<|/point|>"


def random_multi_point(num: int) -> str:
    """Generate multiple points in one tag (path/waypoints)."""
    parts = []
    for _ in range(num):
        x = random.randint(0, 999)
        y = random.randint(0, 999)
        parts.append(f"{x},{y}")
    inner = "[[" + "],[".join(parts) + "]]"
    return f"<|point|>{inner}<|/point|>"


# ---- Template pools ----
OBJECTS_SINGLE = [
    "cat", "dog", "car", "tree", "person", "bird", "chair", "table",
    "cup", "book", "phone", "laptop", "bicycle", "house", "flower",
    "ball", "shoe", "bag", "hat", "apple", "banana", "bottle", "clock",
    "lamp", "door", "window", "key", "pen", "mouse", "keyboard",
]

OBJECTS_MULTI = [
    "cats", "dogs", "cars", "people", "birds", "chairs", "trees",
    "flowers", "balls", "books",
]

BOX_USER_TEMPLATES = [
    "Mark the {obj} with a bounding box.",
    "Draw a box around the {obj}.",
    "Where is the {obj}? Use <|box|> to show the location.",
    "Highlight the {obj} using <|box|>.",
    "Please annotate the {obj} with a bounding box.",
    "Show me where the {obj} is with <|box|>.",
    "Can you locate the {obj} and return a box?",
    "Put a box around the {obj}.",
    "The {obj} needs a bounding box. Please add one.",
    "Use visual primitives to mark the {obj}.",
    "I want to know the position of the {obj}. Output a box.",
    "Detect the {obj} and provide coordinates in <|box|> format.",
    "Give me the bounding box coordinates for the {obj}.",
    "Localize the {obj} using <|box|> tags.",
    "Find the {obj} and enclose it with <|box|>.",
]

BOX_MULTI_USER_TEMPLATES = [
    "Mark all {obj} with bounding boxes.",
    "Draw boxes around every {obj}.",
    "Locate each {obj} using <|box|>.",
    "Annotate all {obj} with their bounding boxes.",
    "Show me where all the {obj} are. Use <|box|> for each.",
    "Put a box around every {obj} in the scene.",
    "Can you find all {obj} and give me their boxes?",
    "Detect all {obj} and return their coordinates.",
    "Localize every {obj} with <|box|> tags.",
    "I need boxes around all {obj}.",
]

POINT_USER_TEMPLATES = [
    "Mark the {obj} with a point.",
    "Where is the {obj}? Use <|point|>.",
    "Point to the {obj} using <|point|> coordinates.",
    "Give me the coordinates of the {obj} as a point.",
    "Show the location of the {obj} with <|point|>.",
    "Use <|point|> to indicate the {obj}'s position.",
    "I want a point coordinate for the {obj}.",
    "Locate the {obj} with a single point using <|point|>.",
    "Please mark the {obj}'s position with <|point|>.",
    "Provide the exact point coordinates of the {obj}.",
]

PATH_USER_TEMPLATES = [
    "Show the waypoints from {start} to {end} using <|point|>.",
    "Trace a path with points from {start} to {end}.",
    "Mark the route as a sequence of <|point|> coordinates.",
    "Give me waypoints from {start} to {end} in <|point|> format.",
    "I need a path with points connecting {start} and {end}.",
    "Provide the point sequence for navigating from {start} to {end}.",
    "Draw the path as <|point|> waypoints from {start} to {end}.",
    "Output the step-by-step points from {start} to {end}.",
]

BOX_ASSISTANT_TEMPLATES = [
    "I see the {obj} at {box}.",
    "The {obj} is located at {box}.",
    "I can identify the {obj}: {box}.",
    "Here is the {obj} location: {box}.",
    "Found the {obj}: {box}.",
    "Bounding box for {obj}: {box}.",
    "The {obj} is marked as {box}.",
    "{obj} position: {box}.",
    "I've located the {obj} at coordinates {box}.",
    "The bounding box for the {obj} is {box}.",
]

BOX_MULTI_ASSISTANT_TEMPLATES = [
    "I can see {count} {obj}. {boxes}.",
    "There are {count} {obj} in the scene: {boxes}.",
    "I found {count} {obj}: {boxes}.",
    "Detected {count} {obj}: {boxes}.",
    "Here are all {count} {obj}: {boxes}.",
]

POINT_ASSISTANT_TEMPLATES = [
    "The {obj} is at {point}.",
    "I can see the {obj} at coordinates {point}.",
    "Pointing to the {obj}: {point}.",
    "The {obj}'s location is {point}.",
    "Located the {obj} at {point}.",
    "Here: {point} marks the {obj}.",
]

PATH_ASSISTANT_TEMPLATES = [
    "Waypoints from {start} to {end}: {points}.",
    "Path: {points}.",
    "Here is the route: {points}.",
    "Navigating from {start} to {end}: {points}.",
    "The path goes through these points: {points}.",
    "Step-by-step from {start} to {end}: {points}.",
]


def generate_sample() -> dict | None:
    """Generate one conversation sample with visual primitive tags."""
    task_type = random.random()

    if task_type < 0.40:
        # Single box
        obj = random.choice(OBJECTS_SINGLE)
        box = random_box()
        user = random.choice(BOX_USER_TEMPLATES).format(obj=obj)
        assistant = random.choice(BOX_ASSISTANT_TEMPLATES).format(obj=obj, box=box)

    elif task_type < 0.55:
        # Multiple boxes (2-5 objects)
        obj = random.choice(OBJECTS_MULTI)
        count = random.randint(2, 5)
        boxes = [random_box() for _ in range(count)]
        # Multiple single tags or one multi tag
        if random.random() < 0.5:
            box_str = " ".join(boxes)
        else:
            box_str = random_multi_box(count)
        user = random.choice(BOX_MULTI_USER_TEMPLATES).format(obj=obj)
        assistant = random.choice(BOX_MULTI_ASSISTANT_TEMPLATES).format(
            count=count, obj=obj, boxes=box_str
        )

    elif task_type < 0.80:
        # Single point
        obj = random.choice(OBJECTS_SINGLE)
        point = random_point()
        user = random.choice(POINT_USER_TEMPLATES).format(obj=obj)
        assistant = random.choice(POINT_ASSISTANT_TEMPLATES).format(obj=obj, point=point)

    else:
        # Multi-point (path/waypoints, 3-8 points)
        start = random.choice(OBJECTS_SINGLE)
        end = random.choice([o for o in OBJECTS_SINGLE if o != start])
        num_points = random.randint(3, 8)
        points = random_multi_point(num_points)
        user = random.choice(PATH_USER_TEMPLATES).format(start=start, end=end)
        assistant = random.choice(PATH_ASSISTANT_TEMPLATES).format(
            start=start, end=end, points=points
        )

    # Validate syntax: tags must be paired
    if not _validate_tags(user + " " + assistant):
        return None

    # Rough length check (avoid > 256 tokens)
    if len(assistant.split()) > 80:
        return None

    return {
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _validate_tags(text: str) -> bool:
    """Ensure all primitive tags are properly paired."""
    box_open = text.count("<|box|>")
    box_close = text.count("<|/box|>")
    point_open = text.count("<|point|>")
    point_close = text.count("<|/point|>")
    return (
        box_open == box_close
        and point_open == point_close
        and box_open + box_close + point_open + point_close > 0
    )


def _sample_complexity(sample: dict) -> int:
    """Proxy complexity: number of tokens in the assistant message."""
    convs = sample.get("conversations", [])
    if not convs:
        return 0
    text = convs[-1].get("content", "")
    # Count visual primitive tags and coordinate numbers.
    tag_count = (
        text.count("<|box|>")
        + text.count("<|point|>")
        + text.count("<|/box|>")
        + text.count("<|/point|>")
    )
    num_count = len([c for c in text if c.isdigit()])
    return len(text.split()) + tag_count * 2 + num_count // 4


def generate_dataset(
    n: int = 25000,
    seed: int = 42,
    coco_ann_file: str | None = None,
    coco_grounding_ratio: float = 0.0,
    curriculum: bool = False,
) -> List[dict]:
    """Generate pretrain conversation dataset.

    Args:
        n: Target number of samples.
        seed: Random seed.
        coco_ann_file: Optional COCO annotation file. If provided, a fraction
            of the samples will use real COCO categories + coordinates.
        coco_grounding_ratio: Fraction of samples to draw from COCO (0.0~0.5).
        curriculum: If True, sort samples from short/simple to long/complex
            (paper-style Stage 1/2 curriculum).

    Returns:
        List of conversation dicts.
    """
    random.seed(seed)
    coco_count = int(n * max(0.0, min(coco_grounding_ratio, 0.5)))
    synthetic_count = n - coco_count

    # Synthetic samples.
    data = []
    attempts = 0
    max_attempts = synthetic_count * 3

    while len(data) < synthetic_count and attempts < max_attempts:
        attempts += 1
        sample = generate_sample()
        if sample is not None:
            data.append(sample)

    # COCO-derived samples (real categories + coordinates, still text-only).
    if coco_count > 0 and coco_ann_file:
        coco_data = generate_coco_grounding_pretrain_samples(
            ann_file=coco_ann_file,
            num_samples=coco_count,
            seed=seed + 1,
        )
        data.extend(coco_data)

    if curriculum:
        data.sort(key=_sample_complexity)
    else:
        random.shuffle(data)
    return data[:n]


def export_for_training(data: List[dict], output_path: str):
    """Export as a single JSON array of conversation dicts."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Generated {len(data)} pretrain samples → {output_path}")


if __name__ == "__main__":
    data = generate_dataset(n=25000, seed=42)
    export_for_training(data, "data/pretrain/pretrain_data.json")

    # Print stats
    single_box = sum(1 for d in data if d["conversations"][1]["content"].count("<|box|>") == 1 and "<|point|>" not in d["conversations"][1]["content"])
    multi_box = sum(1 for d in data if d["conversations"][1]["content"].count("<|box|>") > 1)
    point = sum(1 for d in data if "<|point|>" in d["conversations"][1]["content"] and d["conversations"][1]["content"].count("<|box|>") == 0)
    mixed = len(data) - single_box - multi_box - point
    print(f"  - Single box: {single_box}")
    print(f"  - Multi box: {multi_box}")
    print(f"  - Point/path: {point}")

    # Verify a few samples
    print("\n  Sample 1:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))
    print("\n  Sample 500:")
    print(json.dumps(data[500], ensure_ascii=False, indent=2))
