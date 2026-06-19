"""Backward-compatible shim for process-level reward metrics.

This module previously contained all parsing, geometry, and reward logic.
It has been split into focused modules:

  - src/utils/text_parsing.py      : answer / reasoning / box / point parsing
  - src/utils/geometry.py          : IoU, point distance, maze geometry
  - src/utils/reward/format_rm.py  : Format RM
  - src/utils/reward/quality_rm.py : Quality RM
  - src/utils/reward/accuracy_rm.py: Accuracy RM (process_reward, compute_total_reward)
  - src/utils/difficulty.py        : Easy/Normal/Hard difficulty grading

For new code, import directly from those modules. This file re-exports the
public API so existing notebooks / README snippets keep working.
"""

# Text parsing
from .text_parsing import (
    extract_answer,
    extract_reasoning,
    split_generated_text,
    parse_boxes,
    parse_points,
    lenient_parse_boxes,
    _normalize_answer_text,
    syntax_valid,
)

# Geometry
from .geometry import (
    box_iou,
    match_boxes,
    point_distance,
    match_points,
    check_wall_collision,
    maze_causal_exploration_progress,
    maze_exploration_completeness,
    maze_wall_violation_penalty,
    maze_path_validity,
    maze_answer_correctness,
    check_backtracking_missing,
    _has_duplicate_coords,
    _count_repeated_coordinates,
)

# Format RM
from .reward.format_rm import (
    format_reward,
    primitive_format_compliance_reward,
    _NON_LATIN_SCRIPT_RE,
)

# Quality RM
from .reward.quality_rm import (
    quality_reward_text,
    make_quality_reward_fn,
    _looks_like_reward_hacking,
    _meaningful_references,
    _count_in_answer,
)

# Accuracy RM
from .reward.accuracy_rm import (
    process_reward,
    box_count_answer_consistency_reward,
    counting_reward,
    length_reward,
    repeat_token_penalty,
    compute_total_reward,
    _count_repeated_ngrams,
)

# Difficulty grading
from .difficulty import (
    is_rollout_correct,
    filter_normal_level_data,
)

# TRL completion extraction helper used by some reward factories.
from ..training.grpo_utils import extract_completion_text

# Domain seam — the canonical entry point for new code.
from ..models.visual_primitive_parser import PrimitiveParser

__all__ = [
    # text parsing
    "extract_answer",
    "extract_reasoning",
    "split_generated_text",
    "parse_boxes",
    "parse_points",
    "lenient_parse_boxes",
    "_normalize_answer_text",
    "syntax_valid",
    # geometry
    "box_iou",
    "match_boxes",
    "point_distance",
    "match_points",
    "check_wall_collision",
    "maze_causal_exploration_progress",
    "maze_exploration_completeness",
    "maze_wall_violation_penalty",
    "maze_path_validity",
    "maze_answer_correctness",
    "check_backtracking_missing",
    "_has_duplicate_coords",
    "_count_repeated_coordinates",
    # format rm
    "format_reward",
    "primitive_format_compliance_reward",
    "_NON_LATIN_SCRIPT_RE",
    # quality rm
    "quality_reward_text",
    "make_quality_reward_fn",
    "_looks_like_reward_hacking",
    "_meaningful_references",
    "_count_in_answer",
    # accuracy rm
    "process_reward",
    "box_count_answer_consistency_reward",
    "counting_reward",
    "length_reward",
    "repeat_token_penalty",
    "compute_total_reward",
    "_count_repeated_ngrams",
    # difficulty
    "is_rollout_correct",
    "filter_normal_level_data",
    # helper
    "extract_completion_text",
    # domain seam
    "PrimitiveParser",
]
