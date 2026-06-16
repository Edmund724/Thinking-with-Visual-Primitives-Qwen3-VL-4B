"""Constants for TVP-4B-5090D-IterDPO project."""

import re

# Model
BASE_MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"
BASE_VOCAB_SIZE = 151936  # Qwen3-VL-4B-Thinking original vocab size

# Visual Primitive Tags (revised to paper format)
BOX_OPEN = "<|box|>"
BOX_CLOSE = "<|/box|>"
POINT_OPEN = "<|point|>"
POINT_CLOSE = "<|/point|>"
REF_OPEN = "<|ref|>"
REF_CLOSE = "<|/ref|>"

SPECIAL_TOKENS = [
    BOX_OPEN, BOX_CLOSE,
    POINT_OPEN, POINT_CLOSE,
    REF_OPEN, REF_CLOSE,
]

# Regex patterns for parsing
BOX_PATTERN = re.compile(
    r"<\|box\|>\[\[(.*?)\]\]<\|/box\|>"
)
POINT_PATTERN = re.compile(
    r"<\|point\|>\[\[(.*?)\]\]<\|/point\|>"
)

# QLoRA defaults (aligned with reproduction hyperparameters)
DEFAULT_LORA_R = 256
DEFAULT_LORA_ALPHA = 512
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Memory constraints (GB)
MAX_GPU_MEMORY_GB = 22.0
GPU_MEMORY_WARNING_GB = 20.0

# Training defaults
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_IMAGE_SIZE = 448

# Hard Negative Mining thresholds per round (3-round GRPO schedule)
HN_BOX_IOU_THRESHOLDS = [0.3, 0.5, 0.7]
HN_POINT_DIST_THRESHOLD_PX = 10.0

# DPO
DEFAULT_DPO_BETA = 0.1

# OPD distillation temperature (paper Sec 2.5.4 uses 1.0~1.5)
DEFAULT_DISTILL_TEMPERATURE = 1.0

# Maze dataset ratio
MAZE_SOLVABLE_RATIO = 0.5
MAZE_UNSOLVABLE_RATIO = 0.5

# Qwen3 Thinking tags (chat template wraps reasoning_content with these)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Answer formats
ANSWER_TRUE = r"\boxed{True}"
ANSWER_FALSE = r"\boxed{False}"
