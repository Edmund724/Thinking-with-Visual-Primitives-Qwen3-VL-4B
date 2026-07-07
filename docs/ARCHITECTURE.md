# Architecture

This document covers the key technical designs of the reproduction.

## Visual Primitive Format

Visual primitives are inline spatial markers embedded directly into the Chain-of-Thought:

```text
<|box|>[[x1, y1, x2, y2]]<|/box|>                       # single box
<|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>           # multiple boxes
<|point|>[[x, y]]<|/point|>                            # point / path coordinate
```

Coordinates are normalized to `[0, 999]`.

## Memory Optimization

| Technique | Effect |
|-----------|--------|
| 4-bit NF4 + Double Quantization | ~6GB per model instance |
| Gradient Checkpointing | trade VRAM for time |
| Paged AdamW 8-bit | compressed optimizer states |
| bf16 compute | speed + memory |

A single 24GB GPU can hold **Policy + Reference models** simultaneously. TRL's `GRPOTrainer` reuses the same 4-bit base weights for PEFT models by disabling adapters, giving a peak VRAM of ~14–18GB (KV cache is the dominant cost).

### VRAM Adaptation Guide

| GPU VRAM | batch_size | grad_accum | LoRA r | image_size | max_length | note |
|----------|-----------|-----------|--------|-----------|-----------|------|
| **24GB** (5090D / 4090) | 2 | 2 | 256 | 448 | 2048 | default |
| **16GB** (4080 / 4070 Ti Super) | 1 | 4 | 128 | 384 | 1536 | lower rank |
| **12GB** (4070 Ti / 3060 12G) | 1 | 8 | 64 | 336 | 1024 | aggressive compression; GRPO `num_generations=3` |
| **80GB** (A100 / H100) | 4 | 1 | 256 | 448 | 4096 | full-parameter or larger batch possible |

Tips:
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for 12GB cards.
- OPD keeps the student resident and loads one expert at a time; peak VRAM is ~1.3–1.5× single-model SFT.
- GRPO VRAM scales with `num_generations`: 24GB → `num_generations=5`; 12GB → `num_generations=3`.
- On Windows, high "Shared GPU memory" usage with low total utilization is usually a WDDM allocation issue, not a real OOM. The stage scripts already set `expandable_segments:True`.

## Process Reward Function

Inspired by the paper's three reward heads (**Format**, **Quality**, **Accuracy**):

- **Box tasks**: IoU matching, missed detections, format validity.
- **Point / Maze tasks**: L2 distance, wall collision (Bresenham sampling), backtracking detection.
- **General**: tag pairing (`syntax_valid`), non-Latin penalty, completion length penalty.

```python
from src.utils.metrics import process_reward

reward = process_reward(
    pred_text=pred,
    gt_text=gt,
    task_type="maze",
    iou_threshold=0.5,
    point_dist_threshold=10.0,
    maze_grid=grid,
)
```

## Configuration Management

All stage scripts use a three-level cascade:

```
argparse default (None) → YAML config value → CLI override
```

- `configs/*.yaml` are the single source of truth.
- argparse defaults are `None`; missing YAML keys raise early.
- CLI overrides win.

`StageRunner` in `src/training/stage_runner.py` provides shared argparse setup, YAML loading (`apply_yaml_defaults`), logging, and pickle-cache helpers.

## Domain Seams

### `PrimitiveParser`

`src/models/visual_primitive_parser.py` is the single public API for all visual primitive operations: parse, validate, format, and geometry.

```python
from src.models.visual_primitive_parser import PrimitiveParser

boxes = PrimitiveParser.extract_boxes(text)            # parse
tags  = PrimitiveParser.format_box([(10,20,100,200)])  # format
iou   = PrimitiveParser.box_iou(pred, gt)              # geometry
```

Lower-level modules (`text_parsing.py`, `geometry.py`, `primitive_formatter.py`) are internal.

### `ConversationBuilder`

`src/utils/conversation_builder.py` builds messages uniformly across SFT, GRPO, OPD, and pretrain stages.

### `StageRunner`

`src/training/stage_runner.py` centralizes argparse, YAML loading, logging, and data-cache boilerplate for every stage script.

## Data Generation & Quality Control

| Generator | Task | Description |
|-----------|------|-------------|
| `coco_box_generator.py` | Box / counting | COCO boxes + geometric filtering + coarse counting (3–30 instances) |
| `clevr_spatial.py` | Spatial VQA | 2D synthetic scenes (sphere/cube/cylinder); counting, existence, spatial count, attribute queries |
| `path_tracing.py` | Point | Winding Bézier curves; uniform-color mode forces reliance on curvature continuity |
| `synthetic_maze.py` | Point / Maze | Random maze + BFS path solving |

### `thinking_verifier.py`

All generated samples are validated before training:
- Tag pairing (`<|box|>`/`<|/box|>`, `<|point|>`/`<|/point|>`)
- Coordinate range `[0, 999]`
- Reference validity (thinking steps cite real primitives)
- Counting answer consistency
- Maze self-contradiction detection

Samples failing any check are dropped before SFT/GRPO cold-start training.
