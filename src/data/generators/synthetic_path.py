"""Uniform-Style path tracing dataset generator."""

import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..formatters.primitive_formatter import format_point, normalize_coordinate


def generate_smooth_curve(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int = 20,
    curvature: float = 0.3,
) -> List[Tuple[float, float]]:
    """Generate a smooth curve between start and end with some randomness."""
    points = [start]
    for i in range(1, num_points - 1):
        t = i / (num_points - 1)
        # Linear interpolation + perpendicular offset
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        # Add perpendicular noise
        offset = curvature * np.sin(t * np.pi) * random.uniform(-1, 1)
        px = -(end[1] - start[1])
        py = end[0] - start[0]
        norm = np.sqrt(px**2 + py**2) + 1e-8
        x += offset * px / norm
        y += offset * py / norm
        points.append((x, y))
    points.append(end)
    return points


def generate_path_tracing_image(
    image_size: Tuple[int, int] = (512, 512),
    n_curves: int = 3,
    line_width: int = 3,
) -> Tuple[Image.Image, List[Dict]]:
    """Generate uniform-style path tracing image.

    All curves share the same color and stroke width.
    Endpoints are distinguished by different shapes/labels.

    Returns:
        (image, list_of_curve_metadata)
    """
    img = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(img)

    # Uniform color for all curves (forces model to use geometric continuity, not color)
    curve_color = "#333333"

    curves = []
    margin = 80

    for i in range(n_curves):
        # Random start and end points
        sx = random.randint(margin, image_size[0] - margin)
        sy = random.randint(margin, image_size[1] - margin)
        ex = random.randint(margin, image_size[0] - margin)
        ey = random.randint(margin, image_size[1] - margin)

        curve_points = generate_smooth_curve((sx, sy), (ex, ey), num_points=25)

        # Draw curve
        for j in range(len(curve_points) - 1):
            draw.line(
                [curve_points[j], curve_points[j + 1]],
                fill=curve_color,
                width=line_width,
            )

        # Draw endpoint markers with different shapes
        shapes = ["circle", "square", "triangle"]
        start_shape = shapes[i % len(shapes)]
        end_shape = shapes[(i + 1) % len(shapes)]

        # Start marker
        if start_shape == "circle":
            draw.ellipse([sx-10, sy-10, sx+10, sy+10], fill="#00AA00", outline="black")
        elif start_shape == "square":
            draw.rectangle([sx-10, sy-10, sx+10, sy+10], fill="#0000AA", outline="black")
        else:  # triangle
            draw.polygon([(sx, sy-10), (sx-10, sy+10), (sx+10, sy+10)], fill="#AA0000", outline="black")

        # End marker
        if end_shape == "circle":
            draw.ellipse([ex-10, ey-10, ex+10, ey+10], fill="#00AA00", outline="black")
        elif end_shape == "square":
            draw.rectangle([ex-10, ey-10, ex+10, ey+10], fill="#0000AA", outline="black")
        else:
            draw.polygon([(ex, ey-10), (ex-10, ey+10), (ex+10, ey+10)], fill="#AA0000", outline="black")

        # Label markers
        label = chr(ord("A") + i)
        draw.text((sx - 4, sy - 20), f"{label}1", fill="black")
        draw.text((ex - 4, ey - 20), f"{label}2", fill="black")

        # Sample waypoints for GT (every 5th point)
        waypoints = curve_points[::5]
        norm_waypoints = [
            (normalize_coordinate(p[0], image_size[0]), normalize_coordinate(p[1], image_size[1]))
            for p in waypoints
        ]

        curves.append({
            "start_shape": start_shape,
            "end_shape": end_shape,
            "start_label": f"{label}1",
            "end_label": f"{label}2",
            "waypoints": norm_waypoints,
        })

    return img, curves


def generate_path_dataset(n: int = 30000, seed: int = 42) -> List[Dict]:
    """Generate uniform-style path tracing dataset.

    Returns list of dicts with:
        image: PIL.Image
        prompt: str
        thinking: str
        answer: str
        task_type: str = "path"
    """
    random.seed(seed)
    np.random.seed(seed)
    data = []

    for idx in range(n):
        img, curves = generate_path_tracing_image(n_curves=random.randint(2, 4))

        # Pick one curve as target
        target = random.choice(curves)
        thinking_parts = [
            f"I need to trace the curve from the {target['start_shape']} labeled {target['start_label']} "
            f"to the {target['end_shape']} labeled {target['end_label']}."
        ]

        for wp in target["waypoints"]:
            thinking_parts.append(f"Waypoint: {format_point([wp])}")
        thinking_parts.append("Path complete.")

        data.append({
            "image": img,
            "prompt": (
                f"Trace the curve from the {target['start_shape']} labeled {target['start_label']} "
                f"to the {target['end_shape']} labeled {target['end_label']}. "
                f"Mark key waypoints with <|point|>."
            ),
            "reasoning": "\n".join(thinking_parts),
            "answer": "Path traced successfully.",
            "task_type": "path",
        })

    return data
