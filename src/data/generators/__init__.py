"""Visual primitive data generators — unified registry.

All generators return ``list[dict]`` with the standard keys:
    image, prompt, reasoning, answer, task_type
"""

from .coco_box_generator import (
    generate_coco_box_samples,
    generate_coco_counting_samples,
    generate_coco_negative_box_samples,
    generate_coco_negative_point_samples,
    generate_coco_point_samples,
    generate_synthetic_dense_counting,
)
from .clevr_spatial import generate_clevr_spatial_dataset
from .path_tracing import generate_path_tracing_dataset
from .synthetic_maze import generate_maze_dataset
from .synthetic_path import generate_path_dataset

GENERATORS: dict[str, object] = {
    "coco_box": generate_coco_box_samples,
    "coco_point": generate_coco_point_samples,
    "coco_counting": generate_coco_counting_samples,
    "coco_negative_box": generate_coco_negative_box_samples,
    "coco_negative_point": generate_coco_negative_point_samples,
    "synthetic_dense_counting": generate_synthetic_dense_counting,
    "clevr_spatial": generate_clevr_spatial_dataset,
    "maze": generate_maze_dataset,
    "path_tracing": generate_path_tracing_dataset,
    "synthetic_path": generate_path_dataset,
}

__all__ = [
    "GENERATORS",
    "generate_coco_box_samples",
    "generate_coco_counting_samples",
    "generate_coco_negative_box_samples",
    "generate_coco_negative_point_samples",
    "generate_coco_point_samples",
    "generate_synthetic_dense_counting",
    "generate_clevr_spatial_dataset",
    "generate_maze_dataset",
    "generate_path_tracing_dataset",
    "generate_path_dataset",
]
