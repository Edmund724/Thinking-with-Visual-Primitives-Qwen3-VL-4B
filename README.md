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
Stage 1:  Unified Visual Pretrain  COCO + CLEVR images, box/point grounding  ~7.4h   ✅
Stage 2:  Merge LoRA               Merge visual pretrain LoRA into base      ~24s    ✅
Stage 3a: Box Expert SFT           Box-specific SFT with format-token weighting ~13.4h ✅
Stage 3b: Point Expert SFT         Point+Maze SFT                             ~?    ✅ (including resume)
Stage 4a: Box Expert GRPO          Box expert GRPO (1 round, no early stop)   ~20.1h  ✅
Stage 4b: Point Expert GRPO        Point expert GRPO (1 round, no early stop) ~6h    (est.)
Stage 5:  Unified RFT              Expert-generated rollouts → Unified learning ~5h  (est.)
Stage 6:  OPD                      On-Policy Distillation (D_KL(student||expert)) ~7h  (est.)
                                ──────────────────────────────────────────────
                                Total（已实测部分）:                         ~72h
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
>
> **Results**: 22,500 steps (2 epochs), loss 6.88 → 2.34 (−66%), grad norm 14.48 → 1.25, stable convergence. **Duration: ~7.4h wall-clock time** (2026-06-23 13:34:45 → 20:57:45; 45K samples, 2 epochs, data cache hit).
> - Output: `outputs/stage1_visual_pretrain/`

```bash
python scripts/run_stage1_visual_pretrain.py --config configs/stage1_visual_pretrain.yaml
```

**Output**: `outputs/stage1_visual_pretrain/`

> **Data cache**: Stage 1 now pickles the generated COCO + CLEVR training data to `outputs/stage1_visual_pretrain/train_data_cache_<hash>.pkl`. On resume or re-run, the cache is loaded directly and the slow data-generation step is skipped. The cache key covers `num_box`, `num_point`, `num_clevr`, `coco_image_dir`, and `coco_ann_file`, so changing any of these creates a fresh cache automatically. Use `--regenerate_data` to force rebuilding the cache.
>
> **Auto-resume**: If `--resume_from_checkpoint` is omitted and `outputs/stage1_visual_pretrain/checkpoint-*` exists, the script automatically resumes from the latest checkpoint.
>
> **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step.

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

> **Current config**: 15K box localization + 10K coarse-grained counting + 5K CLEVR spatial/VQA + 2K negative box, plus general pretrain mix. `num_epochs=3`, `max_seq_length=2048`, `batch_size=2`, `grad_accum=6` (effective batch=12), `lr=1e-4`, `format_token_weight=10.0`, `max_grad_norm=1.0`.
>
> **VRAM note**: `max_seq_length` was lowered from 4096 → 2048 and `batch_size` from 4 → 2 (with `gradient_accumulation_steps` 3 → 6) to keep Stage 3a within the RTX 5090D 24 GB budget. Effective batch size is preserved.
>
> **Recent improvements**:
> - SFT targets are now passed through `clean_primitive_tags()` to fix any wrong-order / duplicate tags in the generated data.
> - `WeightedSFTTrainer` up-weights visual-primitive and `<think>` tokens (`format_token_weight=10.0`) so the format syntax is learned faster and the ref-token gradient signal is stronger.
> - `max_grad_norm=1.0` stabilizes training under high format-token loss weights.
> - Supports `--resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-XXX` to continue training.
> - **Auto-resume**: If `--resume_from_checkpoint` is omitted and `outputs/stage3a_sft_box/checkpoint-*` exists, the script automatically resumes from the latest checkpoint. This makes it safe to split long runs across multiple sessions.
> - **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step, so the log file shows exactly when training progressed and how long each segment took.
> - **Data cache**: Training data is cached to `outputs/stage3a_sft_box/train_data_cache_<hash>.pkl`. On resume or re-run, the cache is loaded directly and the slow data-generation step is skipped. Use `--regenerate_data` to force rebuilding.
>
> **Note**: All training stages (Stage 1, 3a, 3b, 4a, 4b, 5, 6) now support training-data pickle caching — auto-saved on first run and loaded directly on subsequent runs/resumes. Each stage uses parameter-based cache keys so changing any data-generation parameter automatically creates a fresh cache. Use `--regenerate_data` to force rebuilding.
>
> **Results**: 14,250 steps (2 epochs), loss 2.87 → 1.62 (−44%), average 1.65, grad norm 6.20 → 0.44, stable convergence. 57,000 samples (15K box + 10K counting + 5K CLEVR + 2K negative + 25K general). **Duration: ~13.4h wall-clock** @ ~2.3 samples/sec. Split into 3 segments: (1) 2026-06-25 11:40→19:12 ~7.5h, (2) 2026-06-26 16:09→17:35 ~1.4h, (3) 2026-06-26 17:35→22:01 ~4.4h. Output: `outputs/stage3a_sft_box/`.
>
> **⚠️ Important (post-2026-06-22 fix)**: If Stage 4a/4b still outputs garbled non-Latin characters (e.g. `personsยิง药材[[...]]`), the root cause is that the special-token embeddings (`<|box|>`, `<|ref|>`, etc.) were frozen during SFT. Make sure you are using the latest code where `src/models/qwen_vl_loader.py` adds `embed_tokens` / `lm_head` to `modules_to_save`, then re-train Stage 3a/3b (ideally from Stage 1 so the merged base also carries trained embeddings). The Stage 3 config's `format_token_weight` has also been lowered from 40.0 to 10.0; 40.0 was a workaround for the frozen-embedding bug.
>
> **Why both `embed_tokens` and `lm_head` are trained separately**: Qwen3-VL-4B has `tie_word_embeddings=True`, so the base model shares the input-embedding and output-projection weights. However, PEFT's `ensure_weight_tying` cannot detect this tie for Qwen3-VL's nested architecture (`_get_module_names_tied_with_embedding` is invoked on the tuner object rather than the base model), so it warns and falls back to placing both layers in `modules_to_save`. We intentionally keep this setup: it makes the special-token embeddings trainable and fixes the garbled-output bug, at the cost of ~2× trainable parameters for these layers. The warning is harmless.

```bash
# From scratch
python scripts/run_stage3a_sft_box.py --config configs/stage3a_sft_box.yaml

# Resume from checkpoint
python scripts/run_stage3a_sft_box.py \
    --config configs/stage3a_sft_box.yaml \
    --resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-500
```

### Stage 3b: Point Expert SFT ✅

> **Current config**: 5K point + 10K maze + 5K path tracing + 500 negative point, plus general pretrain mix. `num_epochs=3`, `max_seq_length=2048`, `batch_size=1`, `grad_accum=8` (effective batch=8), `lr=1e-4`, `format_token_weight=40.0`, `max_grad_norm=1.0`.
>
> **Recent improvements (aligned with Stage 3a)**:
> - SFT targets are now passed through `clean_primitive_tags()` to fix any wrong-order / duplicate tags in the generated data.
> - `WeightedSFTTrainer` up-weights visual-primitive and `<think>` tokens (`format_token_weight=40.0`) so the format syntax is learned faster and the ref-token gradient signal is stronger.
> - `max_grad_norm=1.0` stabilizes training under high format-token loss weights.
> - 3 epochs give embeddings more update opportunities, matching the stage3a training schedule.
> - **Auto-resume**: If `--resume_from_checkpoint` is omitted and `outputs/stage3b_sft_point/checkpoint-*` exists, the script automatically resumes from the latest checkpoint.
> - **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step.
>
> Mid-training VRAM fragmentation once caused speed degradation from 3s/it to 30s/it, resolved by resuming with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

```bash
# Normal run from scratch
python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 3 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2 \
    --format_token_weight 40.0 --max_grad_norm 1.0

# Resume with env var if VRAM fragmentation causes slowdown
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 3 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2 \
    --format_token_weight 40.0 --max_grad_norm 1.0 \
    --resume_from_checkpoint outputs/stage3b_sft_point/checkpoint-5000
```

### Stage 4a: Box Expert GRPO (1 round, no early stopping)

> **Note**: The default config now uses a single GRPO round with `num_epochs=1` and disables the tiny-subset early-stopping callback (`early_stopping_subset_size: 0`). The multi-round loop structure is still supported; if you raise `num_rounds`, each round runs to completion and the script auto-detects the latest `checkpoint-*` to resume an interrupted round.
>
> **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step.
>
> **VRAM tip**: All stage scripts now set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` internally to mitigate CUDA memory fragmentation during long runs.

> **Garbled output tip**: If completions contain non-Latin characters around coordinates, the Stage 3a/3b adapter was likely trained without trainable special-token embeddings. Re-train Stage 3a/3b with the latest `qwen_vl_loader.py` (`embed_tokens` / `lm_head` in `modules_to_save`) before running GRPO.

> **Dtype mismatch fix**: A `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16` during GRPO generation is fixed by `_patch_lm_head_dtype_cast()` in `src/models/qwen_vl_loader.py`. It casts fp32 layer-norm outputs to the `lm_head` weight dtype before the final linear projection. No config change is required.

> **Multi-round OOM tip**: If you increase `num_rounds` and Round 2 (or later) fills up VRAM/RAM, it is usually because the previous round left PyTorch/BitsAndBytes allocator pools populated. `src/training/grpo_runner.py` moves the policy model back to CPU and runs an aggressive cache clear between rounds. The default Stage 4a config uses `batch_size=2`, `gradient_accumulation_steps=3`, `generation_batch_size=8`, `num_generations=8` to keep per-step rollout memory low. If you still hit it, delete the interrupted `outputs/stage4a_grpo_box/round_N` and re-run; completed rounds are skipped.

> **Resume VRAM fix**: Previously, resuming a GRPO round from `round_N/checkpoint-*` loaded the adapter once in `grpo_runner.py` and then again inside `Trainer.train(resume_from_checkpoint=...)`, leaving several extra GB in the CUDA memory pool and often pushing optimizer states to system RAM. The resume path now keeps the policy model loaded from the round's starting point and lets the Trainer load the checkpoint only once. Additionally, the GRPO reference adapter (`ref`) is cast from fp32 to bf16 at training start, saving roughly half of the reference adapter's VRAM (~2.6 GB). Both changes apply automatically; no config change is required.

> **Difficulty filter (paper Sec 2.5.2)**: The paper pre-filters GRPO data to keep only "Normal" difficulty samples (model sometimes succeeds, sometimes fails). For single-GPU setups this step can cause OOM due to the extra on-policy rollouts required. We **skip it by default** (`skip_difficulty_filter: true`) and keep the default dataset small (Stage 4a: 4K samples, Stage 4b: 4K samples) so the pipeline finishes quickly. Hard samples (all rollouts wrong) produce zero reward variance and thus near-zero gradients during GRPO — they waste compute but do not harm training. Easy samples (all correct) behave the same way. To re-enable filtering on a multi-GPU setup, set `skip_difficulty_filter: false` in the YAML and scale the sample counts back up.

```bash
# Default: 2000 box + 1000 counting + 1000 CLEVR (4K samples)
python scripts/run_stage4a_grpo_box.py \
    --model_path outputs/stage3a_sft_box \
    --output_dir outputs/stage4a_grpo_box
```

> **Results**: 4,000 steps (1 epoch, 4K samples), completed in **~20.1h wall-clock** across 3 resume segments: (1) 2026-06-27 16:53 → 2026-06-28 00:03 ~7h 10m (checkpoint-3800), (2) 2026-06-28 06:53 → 08:26 ~1h 33m (checkpoint-4000), (3) 2026-06-28 08:58 → 20:22 ~11h 24m (final). Output: `outputs/stage4a_grpo_box/`.

### Stage 4b: Point Expert GRPO (1 round, no early stopping)

> **Note**: Like Stage 4a, the default config uses a single round with `num_epochs=1` and disabled tiny-subset early stopping. The multi-round loop is still supported; completed rounds are skipped on restart and interrupted rounds auto-resume from the latest `round_N/checkpoint-*`.
>
> **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step.
>
> **Dtype mismatch fix**: Same Stage 4a fix applies: `src/models/qwen_vl_loader.py` injects a dtype-cast wrapper around `lm_head` to prevent `float != c10::BFloat16` errors during GRPO generation.

```bash
# Default: 1000 point + 2000 maze + 1000 path (4K samples)
python scripts/run_stage4b_grpo_point.py \
    --model_path outputs/stage3b_sft_point \
    --output_dir outputs/stage4b_grpo_point
```

### Stage 5: Unified RFT (Expert-generated rollouts)

> **Data cache**: Both the prompt pool and expert-generated filtered training data are cached to `outputs/stage5_rft_unified/`. On resume or re-run, prompts and filtered data are loaded directly from cache, skipping expert model loading and generation. Cache keys include expert model paths, so changing experts automatically triggers regeneration. Use `--regenerate_data` to force rebuilding.
>
> **Auto-resume**: If `--resume_from_checkpoint` is omitted and `outputs/stage5_rft_unified/checkpoint-*` exists, the script automatically resumes the SFT phase from the latest checkpoint.
>
> **Timestamped training logs**: A `TimeLoggingCallback` records the wall-clock timestamp, step, loss/learning_rate/epoch, and elapsed time at every logging step.

```bash
python scripts/run_stage5_rft_unified.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage5_rft_unified
```

### Stage 6: OPD (On-Policy Distillation)

> **Auto-resume**: If `--resume_from_checkpoint` is omitted and `outputs/stage6_opd/checkpoint-*` exists, the script automatically resumes the parallel distillation from the latest checkpoint (optimizer/scheduler/epoch/step state restored).
>
> **Timestamped training logs**: Per-step OPD logs already carry wall-clock timestamps from the stage logger, so you can track progress across multiple sessions.

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
> - **Windows shared GPU memory**: If Task Manager shows shared GPU memory being used even though total GPU memory is far from the limit, this is usually a WDDM allocation issue, not a capacity issue. Stage scripts already set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation. For persistent issues, disable Windows Hardware-Accelerated GPU Scheduling (HAGS) and ensure no other apps are reserving VRAM.

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
