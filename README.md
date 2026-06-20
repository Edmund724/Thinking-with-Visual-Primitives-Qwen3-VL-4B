English | **[中文](README_zh.md)**

# Thinking with Visual Primitives — Qwen3-VL-4B Reproduction

> **Reproducing DeepSeek's "Thinking with Visual Primitives" on a single RTX 5090D (24GB).**

---

## 📄 Paper Background

**Thinking with Visual Primitives** (DeepSeek, 2026) introduces a novel reasoning framework:
> Instead of just "seeing clearer", the model learns to **"point while it reasons."** By **interleaving spatial markers** (points and bounding boxes) directly into the Chain-of-Thought (CoT) reasoning process as **Visual Primitives**, the model elevates spatial references to **"minimal units of thought"** — anchoring abstract linguistic concepts to concrete physical coordinates and closing the **Reference Gap** in complex visual reasoning.

The paper identifies two distinct challenges in multimodal reasoning:
- **Perception Gap**: the ability to see finer details (addressed by high-resolution cropping, dynamic patching, etc.)
- **Reference Gap**: the inability of natural language to precisely point to things in a complex scene

Unlike conventional approaches that treat grounding as a post-hoc verification step, the paper's core thesis is: **spatial markers are neither the result of reasoning nor auxiliary evidence — they are the primary medium of reasoning itself**, an intrinsic component of the cognitive trajectory.

---

## 🎯 Project Positioning

| Dimension | Paper (DeepSeek) | This Reproduction |
|-----------|-----------------|-------------------|
| Base Model | DeepSeek-V4-Flash (284B MoE) | **Qwen3-VL-4B-Thinking** (4B Dense) |
| Training Method | Large-scale GRPO + custom framework | **QLoRA + TRL GRPO + RFT** |
| Visual Primitives (Spatial Markers) | Custom tokens | `<\|box\|>` / `<\|point\|>` |
| VRAM Requirement | Multi-GPU A100/H100 | **Single RTX 5090D 24GB** |
| Data Scale | 460K+ mazes / 125K+ paths | 50K mazes / 10K paths (extensible) |

Since 24GB VRAM cannot accommodate online multi-rollout training for a 284B MoE model, this project adopts a **lightweight Separated Experts (Specialist) architecture + On-Policy Distillation (OPD)**, preserving the core idea while achieving single-GPU feasibility through **4-bit QLoRA (r=256) + Gradient Checkpointing + Paged AdamW 8-bit**.

> **⚠️ Pretrain Limitation**: The original paper's pretrain involves **trillion-scale multimodal pretraining** on 40M+ curated web grounding samples. Due to compute constraints, this project uses a **unified visual grounding pretrain** as the direct entry point — the model learns special token embeddings and visual→coordinate mapping simultaneously on COCO + CLEVR data via QLoRA, eliminating the split between text-only format learning and visual grounding. This is a closer approximation to the paper's approach than the previous two-stage text-then-visual split.

---

## 🖥️ Hardware & Software Requirements

### Hardware
- **GPU**: NVIDIA RTX 5090D (24GB VRAM)
- **VRAM headroom**: Keep usage under 22GB (reserve 2GB for CUDA context and VRAM fragmentation)

### Software (Blackwell Compatible)

| Package | Minimum Version |
|---------|----------------|
| PyTorch | 2.11.0 |
| transformers | 5.10.0 |
| flash-attn | 2.8.3 (auto fallback to eager) |
| bitsandbytes | 0.49.0 |
| accelerate | 1.13.0 |
| peft | 0.19.0 |
| trl | 1.6.0 |

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
# Create environment (recommended)
conda create -n tvp python=3.12 -y
conda activate tvp

# Install PyTorch (CUDA 13.0+)
pip install torch>=2.11.0 torchvision>=0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention 2 for acceleration
# Code auto-falls back to eager attention if compilation fails
pip install flash-attn --no-build-isolation
```

### 2. Download Base Model

```bash
# Auto-downloads from Hugging Face (cached on first run)
# Or download manually
huggingface-cli download Qwen/Qwen3-VL-4B-Thinking --local-dir models/Qwen3-VL-4B-Thinking
```

### 3. Prepare COCO Dataset (Optional, for Box tasks)

```bash
mkdir -p data/coco
wget http://images.cocodataset.org/zips/train2017.zip -P data/coco
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco
unzip data/coco/train2017.zip -d data/coco
unzip data/coco/annotations_trainval2017.zip -d data/coco
```

> Maze and path data are **generated programmatically at runtime** — no pre-download needed.

---

## 🚀 Training Pipeline (Separated Experts + On-Policy Distillation)

```
Stage 1:  Unified Visual Pretrain  COCO + CLEVR images, box/point grounding  ~?    ✅
Stage 2:  Merge LoRA               Merge visual pretrain LoRA into base      ~27s   ✅
Stage 3a: Box Expert SFT           Box-specific SFT with format-token weighting ~?  ✅
Stage 3b: Point Expert SFT         Point+Maze SFT                             ~?    ✅ (including resume)
Stage 4a: Box Expert GRPO          Box expert GRPO (3 rounds, default)        ~6h    (est.)
Stage 4b: Point Expert GRPO        Point expert GRPO (3 rounds, default)      ~6h    (est.)
Stage 5:  Unified RFT              Expert-generated rollouts → Unified learning ~5h  (est.)
Stage 6:  OPD                      On-Policy Distillation (D_KL(student||expert)) ~7h  (est.)
                                ──────────────────────────────────────────────
                                Total (measured):                           ~52h
```

**Core Design**:
- **Separated Experts (Specialists)**: Box Specialist and Point Specialist share the same 4-bit base model but each has an independent LoRA adapter
- **Frozen Specialists**: Both specialists are frozen after Stage 3 training, serving as fixed Teacher models
- **Expert Rollout Generation**: In Stage 5 RFT, specialists generate rollouts (generator) while the Unified model learns (learner)
- **Difficulty Grading**: Easy/Normal/Hard three-tier system; only Normal-level samples used for training
- **On-Policy Distillation (OPD)**: Uses D_KL(student || expert) to consolidate both specialists into a single Unified model
- **Three-Step Thinking (CoT)**: Intent Analysis → Grounding → Summarization

### Stage 1: Unified Visual Grounding Pretrain ✅

**Training on COCO + CLEVR images** to establish "visual feature → coordinate" mapping from the start. Special tokens (`<|box|>`, `<|point|>`) are randomly initialized and learned alongside the LoRA adapter — no separate text-only format pretrain needed.

> **Current config**: `num_box=30000`, `num_point=10000`, `num_clevr=5000` (total 45K samples), `num_epochs=2`, `batch_size=1`, `gradient_accumulation_steps=4` (effective batch=4), `max_seq_length=2048`, `lora_r=256`, `lora_alpha=512`, `learning_rate=2e-6`, curriculum enabled.
> - Output: `outputs/stage1_visual_pretrain/`

```bash
python scripts/run_stage1_visual_pretrain.py --config configs/stage1_visual_pretrain.yaml
```

**Output**: `outputs/stage1_visual_pretrain/`

---

### Merge Stage 2 LoRA

**Must merge!** Avoid stacking double LoRA layers.

Special token embeddings are learned during Stage 1 visual pretrain together with the LoRA adapter — no separate pretrain embedding injection needed.

```bash
python scripts/run_stage2_merge.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage1_visual_pretrain \
    --output_dir outputs/stage2_merged_base
```

**Output**: `outputs/stage2_merged_base/` (full bf16 model, ~8.8GB `model.safetensors`)

> **Do I need to validate the merged model before Stage 3?**  
> Strictly speaking, no — `run_stage2_merge.py` is deterministic and Stage 3 SFT will immediately fail to load if the merge is corrupt. However, because you just fixed the data format / reward functions, I recommend a **5-minute smoke test**: load `outputs/stage2_merged_base` on one COCO image and check that it emits spatial coordinates inside `<think>` tags.
>
> ```bash
> python scripts/diagnostics/smoke_test_stage2.py
> # or specify image / question
> python scripts/diagnostics/smoke_test_stage2.py \
>     --image_path data/coco/train2017/000000000009.jpg \
>     --question "Locate the main object in the image. Mark it with a box."
> ```
>
> If reasoning + coordinates appear, proceed directly to Stage 3a.

### Stage 3a: Box Expert SFT ✅

> **Current config**: 15K box localization + 10K coarse-grained counting + 5K CLEVR spatial/VQA + 2K negative box, plus general pretrain mix. `num_epochs=2`, `max_seq_length=4096`, `batch_size=1`, `grad_accum=8` (effective batch=8), `lr=1e-4`.
>
> **Recent improvements**:
> - SFT targets are now passed through `clean_primitive_tags()` to fix any wrong-order / duplicate tags in the generated data.
> - `WeightedSFTTrainer` up-weights visual-primitive and `<think>` tokens (`format_token_weight=5.0`) so the format syntax is learned faster.
> - Supports `--resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-XXX` to continue training.
>
> **Note**: Stage 3a does not enable data caching (pickle cache) to preserve accurate timing data. From Stage 3b onward, all scripts support training data pickle caching — auto-saved on first run, loaded directly on subsequent runs.

```bash
# From scratch
python scripts/run_stage3a_sft_box.py --config configs/stage3a_sft_box.yaml

# Resume from checkpoint
python scripts/run_stage3a_sft_box.py \
    --config configs/stage3a_sft_box.yaml \
    --resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-500
```

### Stage 3b: Point Expert SFT ✅

> **Benchmark**: ~96K samples (25K general + 10K point + 50K maze + 10K path tracing + 1K negative point), 1 epoch, batch_size=4, grad_accum=2 (effective batch=8), lr=1e-4, ~12K steps, ~2.9s/step, duration **~9.7h** (estimated with negatives, including resume).
>
> Mid-training VRAM fragmentation once caused speed degradation from 3s/it to 30s/it, resolved by resuming with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
# Normal run from scratch
python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 1 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2

# Resume with env var if VRAM fragmentation causes slowdown
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 1 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2 \
    --resume_from_checkpoint outputs/stage3b_sft_point/checkpoint-5000
```

### Stage 4a: Box Expert GRPO (3 rounds by default)

> **Note**: GRPO uses a multi-round loop structure. After each round, the model is reloaded — inter-round gaps are natural checkpoints. Each round should run to completion; if interrupted mid-round, the script auto-detects the latest `checkpoint-*` and resumes from it. Completed rounds are skipped on restart.
>
> **VRAM tip**: All stage scripts now set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` internally to mitigate CUDA memory fragmentation during long runs.

```bash
python scripts/run_stage4a_grpo_box.py \
    --model_path outputs/stage3a_sft_box \
    --output_dir outputs/stage4a_grpo_box \
    --num_samples 5000
```

### Stage 4b: Point Expert GRPO (3 rounds by default)

```bash
python scripts/run_stage4b_grpo_point.py \
    --model_path outputs/stage3b_sft_point \
    --output_dir outputs/stage4b_grpo_point \
    --num_point 2000 --num_maze 5000
```

### Stage 5: Unified RFT (Expert-generated rollouts)

```bash
python scripts/run_stage5_rft_unified.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage5_rft_unified
```

### Stage 6: OPD (On-Policy Distillation)

```bash
python scripts/run_stage6_opd.py \
    --student_path outputs/stage5_rft_unified/final_model \
    --output_dir outputs/stage6_opd
```

### One-Click Run (Recommended)

```bash
bash scripts/run_pipeline.sh
```

---

## 🔬 Inference Examples

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

model, processor = load_qlora_model("outputs/stage6_opd")

image = Image.open("example.jpg")
messages = [
    {"role": "system", "content": "You are a helpful visual reasoning assistant. Think step by step."},
    {"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "How many cats are in the image? Mark each with a box."},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True,
)
response = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
print(response)
```

Expected output includes:
```
<think>
I can see two cats in the image. Let me mark them.
<|box|>[[120, 80, 340, 290]]<|/box|>
<|box|>[[410, 95, 620, 310]]<|/box|>
</think>

The answer is 2.
```

### Batch Inference

Supports batch inference on JSONL files for evaluation or large-scale processing:

```python
import json
from pathlib import Path
from src.models.qwen_vl_loader import load_qlora_model
from src.utils.metrics import process_reward
from PIL import Image

model, processor = load_qlora_model("outputs/stage6_opd")

results = []
with open("eval_data.jsonl") as f:
    for line in f:
        item = json.loads(line)
        image = Image.open(item["image_path"]).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a helpful visual reasoning assistant. Think step by step."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": item["question"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.7, do_sample=True)
        pred = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Compute process reward (optional)
        reward = process_reward(pred, item["answer"], task_type=item.get("task_type", "box"))
        results.append({"pred": pred, "reward": reward, "gt": item["answer"]})

with open("eval_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
```

### Cross-Stage Model Comparison

Compare output quality across training stages to verify capability progression:

```python
from src.models.qwen_vl_loader import load_qlora_model
from PIL import Image

# Load models from various stages
stages = {
    "Stage 2 (Pretrain)": "outputs/stage2_merged_base",
    "Stage 3a (Box SFT)": "outputs/stage3a_sft_box",
    "Stage 6 (OPD)":      "outputs/stage6_opd",
}

image = Image.open("test_image.jpg")
prompt = "Locate the dog in the image."

for name, path in stages.items():
    model, processor = load_qlora_model(path)
    messages = [
        {"role": "system", "content": "You are a helpful visual reasoning assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    print(f"=== {name} ===")
    print(processor.tokenizer.decode(outputs[0], skip_special_tokens=False))
    print()
    del model  # Free VRAM
```

> **Expected behavior**: Pretrain stage outputs correct format but imprecise boxes; SFT Box stage shows structured thinking + precise boxes; OPD stage has both Box and Point capabilities.

---

## 🧠 Key Technical Design

### Visual Primitive (Spatial Marker) Format

The paper defines Visual Primitives as inline tokens in the Chain-of-Thought:

```
<|box|>[[x1, y1, x2, y2]]<|/box|>        # Single Bounding Box
<|box|>[[x1,y1,x2,y2],[x3,y3,x4,y4]]<|/box|>  # Multiple Boxes
<|point|>[[x, y]]<|/point|>              # Point coordinate (maze path, keypoints)
```

Coordinates are uniformly normalized to the `[0, 999]` range.

### Memory Optimization Strategies

| Technique | Effect |
|-----------|--------|
| 4-bit NF4 + Double Quantization | ~6GB per model instance |
| Gradient Checkpointing | Trade compute for memory, reduce activation footprint |
| Paged AdamW 8-bit | Optimizer state compression |
| bf16 computation | Speed + memory win-win |

A single 24GB GPU can hold both **Policy model + Reference model** (TRL's GRPOTrainer reuses base weights for PEFT models by disabling the adapter, peak VRAM ~14-18GB, with KV cache as the largest expense).

### VRAM Adaptation Guide

Recommended configurations for different GPU VRAM sizes:

| GPU VRAM | batch_size | grad_accum | LoRA r | image_size | max_length | Notes |
|----------|-----------|-----------|--------|-----------|-----------|-------|
| **24GB** (5090D / 4090) | 2 | 2 | 256 | 448 | 2048 | Default config for this project |
| **16GB** (4080 / 4070 Ti Super) | 1 | 4 | 128 | 384 | 1536 | Lower LoRA rank to save VRAM |
| **12GB** (4070 Ti / 3060 12G) | 1 | 8 | 64 | 336 | 1024 | Aggressive compression, GRPO `num_generations=3` |
| **80GB** (A100 / H100) | 4 | 1 | 256 | 448 | 4096 | Full parameters or larger batch possible |

> **Tips**:
> - 12GB GPUs: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce VRAM fragmentation.
> - OPD stage loads 3 models simultaneously (student + 2 experts); teacher models auto-use 4-bit quantization; peak VRAM is ~1.8x single-model SFT.
> - GRPO stage VRAM scales with `num_generations`; 12GB: recommend `num_generations=3`, 24GB: can use `num_generations=5`.

### Process Reward Function

Beyond merely checking final answer correctness, we designed fine-grained process rewards — inspired by the paper's three reward heads (**Format**, **Quality**, **Accuracy**):

- **Box tasks**: IoU matching, missed detections, format legality
- **Point / Maze tasks**: L2 distance, wall collision detection (Bresenham sampling), backtracking absence detection
- **General**: label pairing legality (`syntax_valid`), non-Latin script penalty, completion length penalty

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
# Returns: answer_correct, syntax_valid, box_avg_iou, point_avg_dist,
#          wall_collision_count, backtracking_missing, ...
```

### Configuration Management (YAML + argparse)

All stage scripts follow a three-layer default cascade:

```
argparse default (None) → YAML config value → CLI override
```

- **YAML configs** (`configs/*.yaml`) are the **single source of truth** for hyperparameters.
- **argparse defaults** are always `None` — YAML is required. If a key is missing from YAML, the script will error early.
- **CLI flags** override both YAML and argparse defaults, e.g. `--num_epochs 5`.
- `StageRunner` (in `src/training/stage_runner.py`) provides shared boilerplate: argparse setup, YAML loading (`apply_yaml_defaults`), logging, and a pickle data-cache helper.

### Visual Primitive Domain Seam

`PrimitiveParser` (in `src/models/visual_primitive_parser.py`) is the **single public API** for all visual primitive operations — parsing, validation, formatting, and geometry. The lower-level modules (`text_parsing.py`, `geometry.py`, `primitive_formatter.py`) are internal implementation details.

```python
from src.models.visual_primitive_parser import PrimitiveParser

boxes = PrimitiveParser.extract_boxes(text)       # parsing
tags  = PrimitiveParser.format_box([(10,20,100,200)])  # formatting
iou   = PrimitiveParser.box_iou(pred, gt)        # geometry
```

### Data Generation & Quality Control

Beyond the original COCO box/point and synthetic maze tasks, the pipeline now includes additional data generators to broaden the model's reasoning capabilities:

| Generator | Task Type | Description |
|-----------|-----------|-------------|
| `coco_box_generator.py` | Box / Counting | COCO bounding boxes with geometric filtering (mega/tiny/degenerate/edge boxes removed) + coarse-grained counting (3–30 instances) |
| `clevr_spatial.py` | Spatial VQA | 2D synthetic scenes (sphere/cube/cylinder) with counting, existence, spatial count, and attribute queries |
| `path_tracing.py` | Point | Intertwined Bézier curves where the model must trace a target path to its endpoint; supports uniform-color mode to force curvature-based reasoning |
| `synthetic_maze.py` | Point / Maze | Random maze generation with BFS path solving |

**Thinking-chain Verifier** (`thinking_verifier.py`): All generators are automatically filtered through a post-generation validation step that checks:
- Tag pairing (`<|box|>`/`<|/box|>`, `<|point|>`/`<|/point|>`)
- Coordinate range validity (0–999)
- Reference validity (thinking steps reference real primitives)
- Counting answer consistency with primitive count
- Maze self-contradiction detection

Samples failing any check are discarded before training, ensuring high-quality cold-start data for SFT and GRPO.

---

## 📁 Project Structure

```
tvp-4b-5090d/
├── configs/                          # YAML training configs
│   ├── stage1_visual_pretrain.yaml
│   ├── stage3a_sft_box.yaml
│   ├── stage3b_sft_point.yaml
│   ├── stage4a_grpo_box.yaml
│   ├── stage4b_grpo_point.yaml
│   ├── stage5_rft_unified.yaml
│   └── stage6_opd.yaml
├── src/
│   ├── models/
│   │   ├── qwen_vl_loader.py         # Qwen3VL + QLoRA loader
│   │   ├── pretrain_loader.py        # Pretrain model loader + embedding injection
│   │   └── visual_primitive_parser.py # **Domain seam** for all visual primitive ops (parsing, formatting, geometry)
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── sft_dataset.py        # SFT dataset (assistant-only loss mask)
│   │   │   ├── grpo_dataset.py       # GRPO dataset
│   │   │   └── image_loader.py       # Lazy image loading (OOM prevention)
│   │   ├── generators/
│   │   │   ├── __init__.py            # Generator registry
│   │   │   ├── coco_box_generator.py # COCO → box/point/counting samples (3-step thinking + geometric filtering)
│   │   │   ├── synthetic_maze.py     # Synthetic maze generator (3-step thinking)
│   │   │   ├── clevr_spatial.py      # CLEVR-style 2D spatial/VQA generator
│   │   │   ├── path_tracing.py       # Bézier curve path tracing generator
│   │   └── formatters/
│   │       └── primitive_formatter.py # Coordinate label formatting (internal)
│   ├── training/
│   │   ├── stage_runner.py           # **StageRunner**: shared argparse+YAML+logging boilerplate
│   │   ├── trainers/
│   │   │   └── sft_trainer.py        # SFT Trainer wrapper (WeightedSFTTrainer)
│   │   ├── opd_trainer.py            # OPD On-Policy Distillation trainer
│   │   ├── grpo_fixes.py             # GRPOTrainer multimodal monkey-patches
│   │   ├── grpo_utils.py             # GRPO helper utilities (completion text extraction)
│   │   ├── callbacks.py              # Training callbacks (memory monitoring)
│   │   ├── memory_utils.py           # GPU memory utilities (build_param_groups)
│   │   └── config_utils.py           # YAML config loading helpers for stage scripts
│   └── utils/
│       ├── constants.py              # Special token / hyperparameter constants
│       ├── conversation_builder.py   # **ConversationBuilder**: unified message construction (SFT/GRPO/OPD/pretrain)
│       ├── text_parsing.py           # Answer / reasoning / box / point parsing (internal)
│       ├── geometry.py               # IoU, point distance, maze geometry (internal)
│       ├── thinking_verifier.py      # Thinking-chain validation (tag pairing, coord range, ref checks)
│       ├── quality_rm_api.py         # LLM-as-Judge Quality RM (OpenAI-compatible API)
│       ├── logging_utils.py          # Logging initialization
│       ├── difficulty.py             # Easy/Normal/Hard difficulty grading
│       ├── batch_inference.py        # Batched generation helper
│       └── reward/
│           ├── format_rm.py          # Format Reward Model
│           ├── quality_rm.py         # Quality Reward Model (rule-based)
│           └── accuracy_rm.py        # Accuracy Reward Model (process_reward, compute_total_reward)
├── scripts/                          # Stage entry scripts
│   ├── run_stage1_visual_pretrain.py  # Stage 1: Unified Visual Grounding Pretrain
│   ├── run_stage2_merge.py            # Stage 2: Merge LoRA
│   ├── run_stage3a_sft_box.py        # Stage 3a: Box Expert SFT
│   ├── run_stage3b_sft_point.py      # Stage 3b: Point Expert SFT
│   ├── run_stage4a_grpo_box.py       # Stage 4a: Box Expert GRPO
│   ├── run_stage4b_grpo_point.py     # Stage 4b: Point Expert GRPO
│   ├── run_stage5_rft_unified.py     # Stage 5: Unified RFT
│   ├── run_stage6_opd.py             # Stage 6: OPD
│   ├── diagnostics/
│   │   ├── eval_stage2_structure.py      # Stage 2 structure evaluation
│   │   ├── eval_stage3a_paradigm.py      # Stage 3a paradigm check
│   │   ├── smoke_test_stage2.py          # Stage 2 smoke test
│   │   └── diagnose_stage2_resume_loss.py # Stage 2 loss diagnosis
│   └── run_pipeline.sh               # Master Pipeline (one-click run)
├── tests/
│   ├── test_primitive_parser.py      # PrimitiveParser unit tests (30 methods)
│   ├── test_primitive_formatter.py   # Box/point formatting tests
│   ├── test_metrics.py               # Reward function & geometry tool tests
│   ├── test_conversation_builder.py  # ConversationBuilder unit tests (21 tests)
│   ├── test_quality_rm_api.py        # Quality RM API integration tests (23 tests)
│   ├── test_pretrain_format.py       # Pretrain format tests
│   ├── test_weighted_sft_trainer.py  # WeightedSFTTrainer loss tests
│   ├── test_grpo_fixes.py            # GRPO monkey-patch unit tests
│   ├── test_grpo_reward_integration.py # GRPO reward integration tests
│   ├── test_stage_integration.py     # **Stage integration tests** (14 tests, all 8 stages)
│   ├── test_stage3a_data_composition.py # Stage 3a data composition tests
│   ├── test_logging_utils.py         # Logging utility tests
│   └── test_filter_normal_level_data.py # Difficulty filter tests
├── outputs/                          # Training artifacts (organized by stage)
│   ├── stage1_visual_pretrain/       # LoRA adapter + checkpoints
│   ├── stage2_merged_base/           # Merged full model
│   ├── stage3a_sft_box/              # Box Expert SFT adapter
│   ├── stage3b_sft_point/            # Point Expert SFT adapter
│   ├── stage4a_grpo_box/             # Box Expert GRPO adapter
│   ├── stage4b_grpo_point/           # Point Expert GRPO adapter
│   ├── stage5_rft_unified/           # Unified RFT adapter
│   └── stage6_opd/                   # On-Policy Distillation output
├── logs/                             # Training logs per stage
├── data/
│   ├── coco/                         # COCO dataset (manual download)
│   └── cache/maze/                   # Maze image cache
├── models/Qwen3-VL-4B-Thinking/     # Base model (manual download)
├── requirements.txt
├── README.md
└── README_zh.md
```

---

## 🧪 Running Tests

```bash
# Unit tests (fast, no GPU required for most)
pytest tests/ -v --ignore=tests/test_grpo_reward_integration.py --ignore=tests/test_stage_integration.py

# Integration tests (require models + COCO data; auto-skip if unavailable)
pytest tests/test_stage_integration.py -v

# All tests
pytest tests/ -v
```

---

## 🙏 Acknowledgements

This project's implementation references the following works:

- **Paper**: *Thinking with Visual Primitives* (Lu et al., DeepSeek, 2026) — Proposes treating spatial markers (bounding boxes and points) as "minimal units of thought" in multimodal Chain-of-Thought reasoning, closing the Reference Gap in complex visual reasoning. The core idea of this project originates from this paper.
- **[vra/Thinking-with-Visual-Primitives-pytorch](https://github.com/vra/Thinking-with-Visual-Primitives-pytorch)** (Author: Yunfeng Wang): An unofficial PyTorch reproduction using Qwen2-VL-2B + LoRA, achieving a complete Pretrain → SFT → OPD pipeline on a single 12GB+ GPU. This project's overall training pipeline design (Separated Experts + On-Policy Distillation), Visual Primitive format definitions, and process reward function design all drew significant inspiration and reference from this work.
- **[ailuntx/Thinking-with-Visual-Primitives](https://github.com/ailuntx/Thinking-with-Visual-Primitives)**: A community archive/mirror preserved after the original paper's repository was deleted, retaining the original technical report, code, and documentation as an alternative reference for the DeepSeek paper's original implementation.

> **Disclaimer**: This project is an independent reproduction and extension of the above works, with different technical choices in base model (Qwen3-VL-4B), training framework (TRL + QLoRA), and hardware constraints (single RTX 5090D 24GB). Issues and suggestions are welcome.

---

## 📚 Citation

First, cite the original paper:

```bibtex
@article{lu2026think,
  title={Thinking with Visual Primitives},
  author={Lu, Ruijie and Ma, Yiyang and Chen, Xiaokang and Luo, Lingxiao and Wu, Zhiyu and Pan, Zizheng and Liu, Xingchao and Lin, Yutong and Li, Hao and Liu, Wen and Hao, Zhewen and Gao, Xi and Nie, Shaoheng and Wei, Yixuan and Xie, Zhenda and Chen, Ting and Zeng, Gang},
  year={2026}
}
```

And the referenced PyTorch reproduction:

```bibtex
@software{wang2026tvp_pytorch,
  title={Thinking with Visual Primitives --- PyTorch Implementation},
  author={Wang, Yunfeng},
  url={https://github.com/vra/Thinking-with-Visual-Primitives-pytorch},
  year={2026}
}
```

If you use this code, please also cite Qwen3-VL:

```bibtex
@article{bai2025qwen3vl,
  title={Qwen3-VL Technical Report},
  author={Bai, Shuai and Cai, Yuxuan and Chen, Ruizhe and others},
  journal={arXiv preprint arXiv:2511.21631},
  year={2025}
}
```

And this project:

```bibtex
@misc{tvp4b5090d2026,
  title={TVP-4B-5090D: Thinking with Visual Primitives on Qwen3-VL-4B},
  author={Edmund724},
  howpublished={\url{https://github.com/Edmund724/Thinking-with-Visual-Primitives-Qwen3-VL-4B}},
  note={Single-GPU reproduction with QLoRA + TRL GRPO + RFT},
  year={2026}
}
```

---

## 🔧 Closing the Gap to the Original Paper

This reproduction prioritizes the **core idea** (visual primitives as reasoning units) within single-GPU constraints. Below are the main remaining gaps and concrete ways to move closer to the paper without re-implementing trillion-scale pretraining:

### Stage 1 — Pretraining
- **Current**: Unified visual grounding pretrain on COCO + CLEVR images via QLoRA. Special tokens are randomly initialized and learned alongside visual features. No separate text-only format pretrain.
- **Paper**: Large-scale multimodal pretraining on 40M+ curated web grounding samples.
- **Practical next steps**:
  1. Increase data diversity beyond COCO + CLEVR — add Flickr30k Entities, RefCOCO, SA-1B samples, or domain-specific grounding datasets. Even 100K–1M real samples from diverse domains would improve generalization.
  2. If web scraping is infeasible, use publicly available detection/grounding datasets and apply the same two-step filtering logic (semantic + geometric) described in Sec 2.3.3.
  3. Unfreeze the last few ViT layers (`--unfreeze_vit_layers 2-4`) to allow visual features to better adapt to the coordinate prediction task.

### Stage 1 Visual Pretrain — Further Data Diversity
- **Current**: COCO + CLEVR synthetic data trained via QLoRA with ViT frozen.
- **Paper**: DeepSeek-ViT + 3×3 token compression + CSA 4× KV-cache compression, trained end-to-end on massive data.
- **Practical next steps**:
  1. Increase the diversity of visual pretraining data beyond COCO + CLEVR (e.g., add SA-1B samples, synthetic shapes, or domain-specific grounding datasets).
  2. Unfreeze the last few ViT layers (`--unfreeze_vit_layers 2-4`) with a very low LR for better coordinate precision.

### Stage 3 — Cold-Start SFT
- **Current**: COCO box/point/counting + synthetic CLEVR + recursive-backtracking mazes + path tracing. **✅ Implemented `<|ref|>` tokens end-to-end: coarse-counting uses batch ref, fine-grained/Spatial uses individual ref, negative samples use no ref.**
- **Paper**: MLLM-generated thinking chains on GQA scene graphs, 460K mazes with DFS/Prim/Kruskal and rectangular/circular/hexagonal topologies, 125K path-tracing samples.
- **Practical next steps**:
  1. **Fine-grained counting**: Integrate GQA scene-graph data and use an MLLM/API to generate attribute-constrained questions and thinking chains, then verify with `thinking_verifier.py`.
  2. **Maze diversity**: Add Prim and Kruskal generators alongside the existing DFS generator, and add circular/hexagonal topologies. Even a few thousand extra samples per topology improves generalization.
  3. **Spatial/VQA**: Expand CLEVR questions to multi-hop chains and add negative samples (faithful refusals) as the paper emphasizes.
  4. **MLLM-generated thinking**: Wherever you have annotations (GQA, COCO panoptic, SA-1B), use a cheap local MLLM or API to synthesize "Intent Analysis → Grounding → Summarization" chains rather than hand-crafting templates.

### Stage 4 — Specialized RL
- **Current**: Rule-based Quality RM and binary-correctness difficulty grading (now aligned with the paper's "correct rollout count" criterion). **✅ Path Tracing now uses the full paper 4-component Accuracy RM** (forward/reverse/endpoint/continuity + answer correctness). **✅ Complex CLEVR questions (multihop/compare/spatial_existence/spatial_count) use LLM API judge** instead of simple answer matching.
- **Paper**: LLM-based Generative Reward Model (GRM) for Quality RM.
- **Practical next steps**:
  1. Replace `quality_reward_text` with a small local LLM judge (e.g., Qwen2.5-3B-Instruct or a distilled critic) called via a lightweight API, or call it only on a subset of rollouts to control cost.
  2. Use the rule-based QM as a fast filter and the LLM judge for tie-breaking / borderline cases.

### Stage 5 — RFT
- **Current**: Expert rollout → difficulty grading → Normal + 5% Easy → SFT. **✅ Prompt pool now includes path tracing data.**

### Stage 6 — OPD
- **Current**: **✅ Gradient-accumulation parallel distillation** (`train_opd_parallel()`). Within each epoch: Box Expert processes box data → accumulates gradients → swaps to Point Expert for point/maze data → single `optimizer.step()`. Gradient direction is the sum of both expert signals. Only one expert in GPU at a time. Distillation temperature raised from 1.0 → 1.2.

### Observability
- **Current**: **✅ TensorBoard primitive metrics callback** implemented. Every N steps logs format compliance rate, coordinate validity, ref usage rate, and average rewards. All stage config YAMLs set `report_to: tensorboard`. Launch: `tensorboard --logdir outputs/stageX_xxx/tb_primitive_logs`

## ⚠️ Known Limitations

1. **GRPO online rollout overhead**: `num_generations=5` is the limit on a single 24GB GPU; more rollouts require gradient accumulation or offloading.
2. **Flash Attention compatibility**: Blackwell (RTX 5090D) support for flash-attn 2.8.3 is still maturing; code has built-in `eager` fallback.
3. **COCO data**: Initial download is ~18GB; loaded on-demand during training.
4. **This is a reproduction**: The paper's original pipeline includes more stages and larger-scale data; this project is simplified for single-GPU constraints.
5. **vLLM not supported**: vLLM is incompatible with TRL GRPO generation for Qwen3-VL; all GRPO stages use HuggingFace native generation exclusively.
6. **Small sample sizes for fast run-through**: The default configs are intentionally trimmed (e.g., 10K pretrain samples, 15K visual pretrain samples, 2 GRPO rounds with 2 generations) to let the pipeline complete quickly on a single GPU. **Do not expect high final accuracy or production-quality weights** from these defaults; they are for verifying the training flow. Scale the numbers back up if you want better quality.

---

## 📄 License

MIT
