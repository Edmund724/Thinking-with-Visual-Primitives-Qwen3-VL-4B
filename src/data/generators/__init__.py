"""Visual primitive data generators.

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
