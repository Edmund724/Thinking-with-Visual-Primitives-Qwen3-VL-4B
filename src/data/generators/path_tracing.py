"""Path tracing dataset generator with visual primitives (paper Sec 2.4.4).

The task asks the model to follow a specified curve through a tangle of
overlapping lines to identify the endpoint it reaches. Waypoints are sampled
along the target curve and interleaved into the thinking chain as <|point|>
visual primitives.
"""

import os
import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...utils.thinking_verifier import filter_verified_samples
from ...models.visual_primitive_parser import PrimitiveParser


LABELS = [
    "apple", "banana", "cherry", "grape", "lemon", "melon",
    "orange", "peach", "pear", "plum", "kiwi", "mango",
]

COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71", "#F1C40F",
    "#9B59B6", "#E67E22", "#1ABC9C", "#34495E",
]


def _bezier_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    """Quadratic Bezier point at parameter t."""
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return x, y


def _sample_bezier_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    num_points: int = 20,
) -> List[Tuple[int, int]]:
    """Sample points along a quadratic Bezier curve."""
    pts = []
    for i in range(num_points + 1):
        t = i / num_points
        x, y = _bezier_point(p0, p1, p2, t)
        pts.append((int(round(x)), int(round(y))))
    return pts


def _point_segment_distance(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """Distance from point p to line segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    ab_len_sq = abx ** 2 + aby ** 2
    if ab_len_sq == 0:
        return np.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    return np.hypot(px - proj_x, py - proj_y)


def _is_endpoint_clear(
    endpoint: Tuple[int, int],
    other_curves: List[List[Tuple[int, int]]],
    threshold: float = 12.0,
) -> bool:
    """Check that endpoint is not crossed or overlapped by other curves."""
    for curve in other_curves:
        for i in range(len(curve) - 1):
            dist = _point_segment_distance(endpoint, curve[i], curve[i + 1])
            if dist < threshold:
                return False
    return True


def _curves_intersect(
    curve_a: List[Tuple[int, int]],
    curve_b: List[Tuple[int, int]],
    min_distance: float = 8.0,
) -> bool:
    """Check if two curves come close to each other (excluding shared endpoints)."""
    for i in range(len(curve_a) - 1):
        a1, a2 = curve_a[i], curve_a[i + 1]
        for j in range(len(curve_b) - 1):
            b1, b2 = curve_b[j], curve_b[j + 1]
            # Sample a few points on segment a1-a2 and check distance to segment b1-b2
            for k in range(5):
                t = k / 4.0
                px = a1[0] + t * (a2[0] - a1[0])
                py = a1[1] + t * (a2[1] - a1[1])
                if _point_segment_distance((px, py), b1, b2) < min_distance:
                    return True
    return False


def _generate_random_control_point(
    start: Tuple[int, int],
    end: Tuple[int, int],
    img_size: Tuple[int, int],
    rng: random.Random,
) -> Tuple[int, int]:
    """Generate a control point that creates a reasonably curved line."""
    img_w, img_h = img_size
    # Midpoint
    mx = (start[0] + end[0]) / 2
    my = (start[1] + end[1]) / 2
    # Offset perpendicular to the line
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.hypot(dx, dy)
    if length < 1:
        offset_range = 80
    else:
        offset_range = min(length * 0.6, 120)

    # Perpendicular direction
    perp_x = -dy / length if length > 0 else 1
    perp_y = dx / length if length > 0 else 0
    offset = rng.uniform(-offset_range, offset_range)

    cx = int(mx + perp_x * offset)
    cy = int(my + perp_y * offset)

    # Keep inside image with margin
    cx = max(40, min(img_w - 40, cx))
    cy = max(40, min(img_h - 40, cy))
    return cx, cy


def _random_endpoint(
    img_size: Tuple[int, int],
    rng: random.Random,
    margin: int = 50,
) -> Tuple[int, int]:
    """Sample a random endpoint away from image borders."""
    img_w, img_h = img_size
    return (
        rng.randint(margin, img_w - margin),
        rng.randint(margin, img_h - margin),
    )


def _render_scene(
    curves: List[List[Tuple[int, int]]],
    labels: List[str],
    colors: List[str],
    img_size: Tuple[int, int],
    uniform_style: bool = False,
) -> Image.Image:
    """Render the path tracing scene."""
    img = Image.new("RGB", img_size, "#FAFAFA")
    draw = ImageDraw.Draw(img)

    # Draw all curves
    for curve, color in zip(curves, colors):
        render_color = "#555555" if uniform_style else color
        for i in range(len(curve) - 1):
            draw.line([curve[i], curve[i + 1]], fill=render_color, width=3)

    # Draw start/end markers with labels
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, (curve, label) in enumerate(zip(curves, labels)):
        start = curve[0]
        end = curve[-1]

        # Start marker: circle
        r = 10
        draw.ellipse([start[0] - r, start[1] - r, start[0] + r, start[1] + r],
                     fill="white", outline="black", width=2)
        draw.text((start[0] + r + 2, start[1] - 8), f"{label} start", fill="black", font=font)

        # End marker: square
        draw.rectangle([end[0] - r, end[1] - r, end[0] + r, end[1] + r],
                       fill="white", outline="black", width=2)
        draw.text((end[0] + r + 2, end[1] - 8), label, fill="black", font=font)

    return img


def _build_path_tracing_thinking(
    start_label: str,
    end_label: str,
    waypoints: List[Tuple[int, int]],
) -> str:
    """Build 3-step thinking for path tracing."""
    if not waypoints:
        return ""

    start_pt = waypoints[0]
    end_pt = waypoints[-1]
    intent = f"I need to trace the line from {start_label} to its destination."

    # Use a denser set of points for high-curvature regions; here we uniformly sample
    # but cap total to keep sequence length reasonable.
    if len(waypoints) > 25:
        indices = np.linspace(0, len(waypoints) - 1, 25, dtype=int)
        sampled = [waypoints[i] for i in indices]
    else:
        sampled = waypoints

    path_str = PrimitiveParser.format_point(sampled)
    grounding = f"Following the line from {start_label}: {path_str}"
    summarization = f"The line from {start_label} connects to {end_label}."

    return (
        f"Intent Analysis: {intent}\n"
        f"Grounding: {grounding}\n"
        f"Summarization: {summarization}"
    )


def generate_path_tracing_dataset(
    n: int = 10000,
    seed: int = 42,
    cache_dir: str = "data/cache/path_tracing",
    img_size: Tuple[int, int] = (512, 512),
    num_lines: int = 4,
    uniform_style_ratio: float = 0.3,
) -> List[Dict]:
    """Generate path tracing dataset.

    For each sample:
      - Generate N entangled Bezier curves, each connecting a unique start/end label.
      - Ensure endpoints are not crossed by unrelated curves and endpoints do not overlap.
      - Randomly render in uniform style (all lines same color) 30% of the time.
      - Prompt: "Where does the {start_label} connect to?"
      - Thinking: sequence of <|point|> waypoints along the target curve.
      - Answer: \\boxed{{end_label}}
    """
    os.makedirs(cache_dir, exist_ok=True)
    rng = random.Random(seed)

    data = []
    sample_idx = 0
    attempts = 0
    max_total_attempts = n * 20

    while len(data) < n and attempts < max_total_attempts:
        attempts += 1
        rng_inner = random.Random(seed + attempts)

        # Sample labels
        sample_labels = rng_inner.sample(LABELS, k=num_lines)

        curves: List[List[Tuple[int, int]]] = []
        colors: List[str] = []
        line_attempts = 0

        for i in range(num_lines):
            line_attempts_single = 0
            while line_attempts_single < 30:
                line_attempts_single += 1
                start = _random_endpoint(img_size, rng_inner)
                end = _random_endpoint(img_size, rng_inner)

                # Avoid endpoint overlap with existing endpoints
                existing_endpoints = []
                for c in curves:
                    existing_endpoints.extend([c[0], c[-1]])
                too_close = any(
                    np.hypot(start[0] - ep[0], start[1] - ep[1]) < 30
                    or np.hypot(end[0] - ep[0], end[1] - ep[1]) < 30
                    for ep in existing_endpoints
                )
                if too_close:
                    continue

                control = _generate_random_control_point(start, end, img_size, rng_inner)
                curve = _sample_bezier_points(start, control, end, num_points=30)

                # Ensure this curve crosses/intersects others enough to be entangled
                # but does not violate endpoint safety.
                endpoint_clear = (
                    _is_endpoint_clear(curve[0], curves, threshold=12)
                    and _is_endpoint_clear(curve[-1], curves, threshold=12)
                )
                if not endpoint_clear:
                    continue

                # Optional: require at least one crossing with previous curves
                has_crossing = any(_curves_intersect(curve, c, min_distance=10) for c in curves)
                if curves and not has_crossing:
                    # Allow some lines without crossings to add variety
                    if rng_inner.random() < 0.7:
                        continue

                curves.append(curve)
                colors.append(rng_inner.choice(COLORS))
                break
            else:
                break

        if len(curves) < num_lines:
            continue

        uniform_style = rng_inner.random() < uniform_style_ratio
        img = _render_scene(curves, sample_labels, colors, img_size, uniform_style)

        # Choose one curve as the target
        target_idx = rng_inner.randint(0, num_lines - 1)
        target_curve = curves[target_idx]
        start_label = f"{sample_labels[target_idx]} start"
        end_label = sample_labels[target_idx]

        img_path = os.path.join(cache_dir, f"path_{seed}_{sample_idx:06d}.png")
        img.save(img_path)
        sample_idx += 1

        # Normalize waypoints
        normalized_waypoints = [
            (PrimitiveParser.normalize_coordinate(x, img_size[0]), PrimitiveParser.normalize_coordinate(y, img_size[1]))
            for x, y in target_curve
        ]

        reasoning = _build_path_tracing_thinking(start_label, end_label, normalized_waypoints)
        prompt = f"Where does the {start_label} connect to? Put the destination label in \\boxed{{}}."
        answer = f"\\boxed{{{end_label}}}"

        data.append({
            "image": img_path,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "task_type": "point",
        })

    data = filter_verified_samples(data)
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Generated {len(data)} verified path tracing samples")
    return data
