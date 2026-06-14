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
| Data Scale | 460K+ mazes / 125K+ paths | 50K mazes / 15K paths (extensible) |

Since 24GB VRAM cannot accommodate online multi-rollout training for a 284B MoE model, this project adopts a **lightweight Separated Experts (Specialist) architecture + On-Policy Distillation (OPD)**, preserving the core idea while achieving single-GPU feasibility through **4-bit QLoRA (r=256) + Gradient Checkpointing + Paged AdamW 8-bit**.

> **⚠️ Pretrain Limitation**: The original paper's pretrain involves **trillion-scale multimodal pretraining**, where the model builds the foundational ability of "visual primitives as thinking units" on massive web data. Due to compute constraints, Stage 1 of this project only performs **Format Pretraining** — teaching the model the syntax of outputting `<|box|>`, `<|point|>` and other special tokens. Stage 2 Visual Pretrain on COCO images compensates for the visual→coordinate grounding ability.

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
| flash-attn | 2.8.0+ (auto fallback to eager) |
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
Stage 1:  Text Pretrain          Text-only embedding initialization           ~1.5h  ✅
Stage 2:  Visual Pretrain        COCO images + box/point visual pretrain      ~14h   ✅
Stage 2M: Merge LoRA             Merge visual pretrain LoRA into base model   ~18s   ✅
Stage 3a: Box Expert SFT         70% general + 30% Box-specific SFT           ~4.2h  ✅
Stage 3b: Point Expert SFT       70% general + 30% Point+Maze SFT            ~8.5h  ✅ (including resume)
Stage 4a: Box Expert GRPO        Box expert GRPO (3 rounds, default)          ~6h    (est.)
Stage 4b: Point Expert GRPO      Point expert GRPO (3 rounds, default)        ~6h    (est.)
Stage 5:  Unified RFT            Expert-generated rollouts → Unified learning ~5h    (est.)
Stage 6:  OPD                    On-Policy Distillation (D_KL(student||expert)) ~7h  (est.)
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

### Stage 1: Text Pretrain (Format Pretraining) ✅

**Text-only training, no images**. Only trains `embed_tokens` layers. 25K programmatically generated samples, 3 epochs.

> **Benchmark**: 25K samples, 3 epochs, 18750 steps (batch_size=4, lr=2e-4), Epoch 1/2/3 Avg Loss: 1.0399 / 0.9944 / 0.9899, duration **~1h25min**.

```bash
python scripts/run_stage1_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --output_dir outputs/stage1_pretrain \
    --num_epochs 3
```

**Output**: `outputs/stage1_pretrain/pretrain_state_dict.pt`

---

### Stage 2: Visual Pretrain ✅

**Training on COCO images** to establish real "visual features → coordinates" mapping. No random coordinate guessing.

> **Benchmark**: 60000 samples (50K box + 10K point), 1 epoch, batch_size=2, grad_accum=2 (effective batch=4), duration **~14h**.

```bash
python scripts/run_stage2_visual_pretrain.py \
    --model_path models/Qwen3-VL-4B-Thinking \
    --pretrain_embedding_path outputs/stage1_pretrain \
    --output_dir outputs/stage2_visual_pretrain \
    --num_box 50000 --num_point 10000 \
    --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 2
```

**Output**: `outputs/stage2_visual_pretrain/`

---

### Merge Stage 2 LoRA

**Must merge!** Avoid stacking double LoRA layers.

> **Benchmark duration**: **~18s**.

```bash
python scripts/merge_stage2.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage2_visual_pretrain \
    --output_dir outputs/stage2_merged_base
```

### Stage 3a: Box Expert SFT ✅

> **Benchmark**: 40000 samples (25K general + 15K box), 1 epoch, batch_size=1, grad_accum=8 (effective batch=8), lr=1e-4, 5000 steps, ~3.0s/step, duration **~4h12min**.
>
> **Note**: Stage 3a does not enable data caching (pickle cache) to preserve accurate timing data. From Stage 3b onward, all scripts support training data pickle caching — auto-saved on first run, loaded directly on subsequent runs.

```bash
python scripts/run_stage3a_sft_box.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3a_sft_box \
    --num_box 15000 --num_epochs 1 --learning_rate 1e-4
```

### Stage 3b: Point Expert SFT ✅

> **Benchmark**: 85000 samples (25K general + 10K point + 50K maze), 1 epoch, batch_size=4, grad_accum=2 (effective batch=8), lr=1e-4, 10625 steps, ~2.9s/step, duration **~8.5h** (including resume).
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
│   ├── stage2_visual_pretrain.yaml
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
│   │   └── visual_primitive_parser.py # Visual primitive parser
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── sft_dataset.py        # SFT dataset (assistant-only loss mask)
│   │   │   ├── grpo_dataset.py       # GRPO dataset
│   │   │   └── image_loader.py       # Lazy image loading (OOM prevention)
│   │   ├── generators/
│   │   │   ├── coco_box_generator.py # COCO → box/point/counting samples (3-step thinking + geometric filtering)
│   │   │   ├── synthetic_maze.py     # Synthetic maze generator (3-step thinking)
│   │   │   ├── clevr_spatial.py      # CLEVR-style 2D spatial/VQA generator
│   │   │   ├── path_tracing.py       # Bézier curve path tracing generator
│   │   │   └── synthetic_path.py     # Synthetic path generator (unused)
│   │   └── formatters/
│   │       └── primitive_formatter.py # Coordinate label formatting
│   ├── training/
│   │   ├── trainers/
│   │   │   └── sft_trainer.py        # SFT Trainer wrapper
│   │   ├── pretrain_trainer.py       # Pretrain Trainer (embedding only)
│   │   ├── opd_trainer.py            # OPD On-Policy Distillation trainer
│   │   ├── grpo_fixes.py             # GRPOTrainer multimodal monkey-patches
│   │   ├── callbacks.py              # Training callbacks (memory monitoring)
│   │   └── memory_utils.py           # GPU memory utilities
│   └── utils/
│       ├── constants.py              # Special token / hyperparameter constants
│       ├── metrics.py                # Format RM + Accuracy RM + difficulty grading + length penalty
│       ├── thinking_verifier.py       # Thinking-chain validation (tag pairing, coord range, ref checks)
│       └── logging_utils.py          # Logging initialization
├── scripts/                          # Stage entry scripts
│   ├── generate_pretrain_data.py     # Pretrain data generator
│   ├── run_stage1_pretrain.py        # Stage 1: Text Pretrain
│   ├── run_stage2_visual_pretrain.py # Stage 2: Visual Pretrain
│   ├── merge_stage2.py               # Stage 2 LoRA Merge
│   ├── run_stage3a_sft_box.py        # Stage 3a: Box Expert SFT
│   ├── run_stage3b_sft_point.py      # Stage 3b: Point Expert SFT
│   ├── run_stage4a_grpo_box.py       # Stage 4a: Box Expert GRPO
│   ├── run_stage4b_grpo_point.py     # Stage 4b: Point Expert GRPO
│   ├── run_stage5_rft_unified.py     # Stage 5: Unified RFT
│   ├── run_stage6_opd.py             # Stage 6: OPD
│   └── run_pipeline.sh               # Master Pipeline (one-click run)
├── tests/
│   ├── test_primitive_parser.py      # Coordinate parser unit tests
│   ├── test_metrics.py               # Reward function & geometry tool tests
│   ├── test_pretrain_format.py       # Pretrain format tests
│   ├── test_grpo_fixes.py            # GRPO monkey-patch unit tests
│   ├── test_grpo_reward_integration.py # GRPO reward integration tests
│   └── test_logging_utils.py         # Logging utility tests
├── outputs/                          # Training artifacts (organized by stage)
│   ├── stage1_pretrain/              # embedding state_dict
│   ├── stage2_visual_pretrain/       # LoRA adapter + checkpoints
│   ├── stage2_merged_base/           # Merged full model
│   ├── stage3a_sft_box/              # Box Expert SFT adapter
│   ├── stage3b_sft_point/            # Point Expert SFT adapter
│   ├── stage4a_grpo_box/             # Box Expert GRPO adapter
│   ├── stage4b_grpo_point/           # Point Expert GRPO adapter
│   ├── stage5_rft_unified/           # Unified RFT adapter
│   └── stage6_opd/                   # On-Policy Distillation output
├── logs/                             # Training logs per stage
├── data/
│   ├── pretrain/pretrain_data.json   # Format pretrain data
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

## ⚠️ Known Limitations

1. **GRPO online rollout overhead**: `num_generations=5` is the limit on a single 24GB GPU; more rollouts require gradient accumulation or offloading.
2. **Flash Attention compatibility**: Blackwell (RTX 5090D) support for flash-attn 2.8.0+ is still maturing; code has built-in `eager` fallback.
3. **COCO data**: Initial download is ~18GB; loaded on-demand during training.
4. **This is a reproduction**: The paper's original pipeline includes more stages and larger-scale data; this project is simplified for single-GPU constraints.
5. **vLLM not supported**: vLLM is incompatible with TRL GRPO generation for Qwen3-VL; all GRPO stages use HuggingFace native generation exclusively.

---

## 📄 License

MIT

---

## 🤗 Model Weights (Planned Upload After Training)

**⚠️ Current status: Training in progress, weights not yet uploaded.**

After training completes, **full model weights** (base + LoRA merged full bf16 parameters) will be uploaded to **ModelScope**, ready to use out-of-the-box without needing to separately load the base model.

Planned upload location:

```bash
# Available in the future (not currently available)
modelscope download Edmund724/tvp-4b-5090d-qwen3-vl-4b --local-dir ./weights
```

Usage after upload:
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./weights",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("./weights", trust_remote_code=True)
```

> Estimated full weights ~8-9GB (bf16), supporting direct inference and continued fine-tuning. Actual download links will be updated here once available.
