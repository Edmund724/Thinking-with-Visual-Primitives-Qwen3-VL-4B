# Training Guide

This document describes the full 6-stage training pipeline. For a high-level overview, see [README.md](../README.md).

## Pipeline Overview

```
Stage 1:  Unified Visual Pretrain   COCO + CLEVR, box/point grounding      ~7.4h
Stage 2:  Merge LoRA                Merge visual pretrain adapter into base  ~1m
Stage 3a: Box Expert SFT            Box-specific SFT with format weight    ~13.4h
Stage 3b: Point Expert SFT          Point + maze SFT                       ~16h
Stage 4a: Box Expert GRPO           Box GRPO (1 epoch, no early stop)      ~20.1h
Stage 4b: Point Expert GRPO         Point GRPO (1 epoch, no early stop)    ~36.4h
Stage 5:  Unified RFT               Expert rollouts → unified SFT          ~2.7h
Stage 6:  OPD                       On-policy distillation (KL)            ~7h
                                        ───────────────────────────────
                                        Total verified runtime: ~96h
```

**Core design**:
- **Separated Experts (Specialists)**: Box Specialist and Point Specialist share the same 4-bit base model but each carries an independent LoRA adapter.
- **Frozen Specialists**: After Stage 3, both specialists are frozen and serve as fixed teacher models.
- **Expert Rollouts**: In Stage 5 RFT, specialists generate rollouts; the unified model learns from them.
- **Difficulty Grading**: Easy / Normal / Hard. Only Normal-level samples are used for training.
- **On-Policy Distillation (OPD)**: Distills both specialists into a single unified model via `D_KL(student || expert)`.
- **Three-step Chain-of-Thought**: Intent Analysis → Grounding → Summarization.

Run the entire pipeline with:

```bash
bash scripts/run_pipeline.sh
```

---

## Stage 1: Unified Visual Grounding Pretrain

Train on COCO + CLEVR images to establish the "visual feature → coordinate" mapping. Special tokens (`<|box|>`, `<|point|>`) are randomly initialized and learned jointly with the LoRA adapter.

**Default config** (`configs/stage1_visual_pretrain.yaml`):
- `num_box=30000`, `num_point=10000`, `num_clevr=5000` (45K samples)
- `num_epochs=2`, `batch_size=1`, `gradient_accumulation_steps=4`
- `max_seq_length=2048`, `lora_r=256`, `lora_alpha=512`, `lr=2e-6`
- Curriculum enabled

```bash
python scripts/run_stage1_visual_pretrain.py --config configs/stage1_visual_pretrain.yaml
```

**Output**: `outputs/stage1_visual_pretrain/`

**Data caching**: Training data is pickled to `outputs/stage1_visual_pretrain/train_data_cache_<hash>.pkl`. The cache key includes `num_box`, `num_point`, `num_clevr`, `coco_image_dir`, and `coco_ann_file`. Use `--regenerate_data` to force rebuild.

**Auto-resume**: If `outputs/stage1_visual_pretrain/checkpoint-*` exists, the script automatically resumes from the latest checkpoint when `--resume_from_checkpoint` is not specified.

---

## Stage 2: Merge LoRA

**Required**: merge the Stage 1 adapter into the base model to avoid stacking LoRAs.

```bash
python scripts/run_stage2_merge.py \
    --base_model models/Qwen3-VL-4B-Thinking \
    --adapter_path outputs/stage1_visual_pretrain \
    --output_dir outputs/stage2_merged_base
```

**Output**: `outputs/stage2_merged_base/` (~8.8GB `model.safetensors`)

**Smoke test** (recommended, ~5 minutes):

```bash
python scripts/diagnostics/smoke_test_stage2.py
# or with custom image/question
python scripts/diagnostics/smoke_test_stage2.py \
    --image_path data/coco/train2017/000000000009.jpg \
    --question "Locate the main object in the image. Mark it with a box."
```

---

## Stage 3a: Box Expert SFT

**Default config** (`configs/stage3a_sft_box.yaml`):
- 15K box + 10K coarse counting + 5K CLEVR spatial/VQA + 2K negative box samples, mixed with general pretrain data
- `num_epochs=3`, `batch_size=2`, `grad_accum=6` (effective batch=12)
- `lr=1e-4`, `format_token_weight=10.0`, `max_grad_norm=1.0`
- `max_seq_length=2048`

```bash
python scripts/run_stage3a_sft_box.py --config configs/stage3a_sft_box.yaml

# Resume
python scripts/run_stage3a_sft_box.py \
    --config configs/stage3a_sft_box.yaml \
    --resume_from_checkpoint outputs/stage3a_sft_box/checkpoint-500
```

**Output**: `outputs/stage3a_sft_box/`

**Key features**:
- Targets are cleaned with `clean_primitive_tags()` to fix misordered/duplicate tags.
- `WeightedSFTTrainer` applies `format_token_weight=10.0` to visual primitive and `<think>` tokens.
- Data caching and auto-resume are supported.

---

## Stage 3b: Point Expert SFT

**Default config** (`configs/stage3b_sft_point.yaml`):
- 5K point + 10K maze + 5K path tracing + 500 negative point samples, mixed with general pretrain data
- `num_epochs=3`, `batch_size=1`, `grad_accum=8` (effective batch=8)
- `lr=1e-4`, `format_token_weight=40.0`, `max_grad_norm=1.0`
- `max_seq_length=2048`

```bash
python scripts/run_stage3b_sft_point.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage3b_sft_point \
    --num_point 10000 --num_maze 50000 \
    --num_epochs 3 --learning_rate 1e-4 \
    --batch_size 4 --gradient_accumulation_steps 2 \
    --format_token_weight 40.0 --max_grad_norm 1.0

# Resume with expandable segments if fragmentation occurs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_stage3b_sft_point.py ...
```

**Output**: `outputs/stage3b_sft_point/`

---

## Stage 4a: Box Expert GRPO

Default config uses a single epoch and disables tiny-subset early stopping (`early_stopping_subset_size: 0`). Multi-round loops are still supported.

```bash
python scripts/run_stage4a_grpo_box.py \
    --model_path outputs/stage3a_sft_box \
    --output_dir outputs/stage4a_grpo_box
```

**Output**: `outputs/stage4a_grpo_box/`

**Notes**:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically in all stage scripts.
- Difficulty filtering is skipped by default (`skip_difficulty_filter: true`) to avoid OOM on a single GPU.
- If you increase `num_rounds` and a round is killed, delete the interrupted `outputs/stage4a_grpo_box/round_N` and rerun; completed rounds are skipped automatically.

---

## Stage 4b: Point Expert GRPO

Same design as Stage 4a.

```bash
python scripts/run_stage4b_grpo_point.py \
    --model_path outputs/stage3b_sft_point \
    --output_dir outputs/stage4b_grpo_point
```

**Output**: `outputs/stage4b_grpo_point/`

---

## Stage 5: Unified RFT

Experts generate rollouts; the unified model learns from filtered Normal-level samples.

**Fast mode** (`configs/stage5_rft_unified.yaml`):
- Small prompt budget and `num_rollouts: 2`
- `min_normal_samples=10`
- Keeps the full pipeline (expert → rejection sampling → difficulty grading → Normal + 5% Easy → SFT) but runs in minutes instead of hours.

```bash
python scripts/run_stage5_rft_unified.py \
    --model_path outputs/stage2_merged_base \
    --output_dir outputs/stage5_rft_unified
```

**Output**: `outputs/stage5_rft_unified/`

**Options**:
- `--skip_expert_generation` / `skip_expert_generation: true`: skip rollout generation and use existing filtered data.
- Expert paths in the config are auto-resolved to the latest `round_N/adapter_model.safetensors`.
- Auto-resume from the latest `checkpoint-*` is supported.

---

## Stage 6: OPD (On-Policy Distillation)

Distills the box and point specialists into a single unified model with `D_KL(student || expert)`.

```bash
python scripts/run_stage6_opd.py \
    --student_path outputs/stage5_rft_unified/final_model \
    --output_dir outputs/stage6_opd
```

**Output**: `outputs/stage6_opd/`

**Memory optimization**:
- `embed_tokens` / `lm_head` are frozen after earlier stages.
- 8-bit AdamW reduces trainable parameters from ~917M to ~528M and optimizer-state memory by ~6GB.
- Peak allocated VRAM is ~18.5GB under the default config (`max_new_tokens=512`, one image per batch).
- Before freezing, OPD checks the L2 norm of visual primitive token embeddings and warns if they appear uninitialized.

---

## Common Training Utilities

### Timestamped logs

`TimeLoggingCallback` records wall-clock time, step, loss/learning_rate/epoch, and elapsed time at each logging step.

### Data caching

All stages support pickle caching of generated training data. The cache key is based on data-generation parameters, so changing any parameter creates a new cache automatically. Use `--regenerate_data` to force rebuild.

### YAML + argparse config cascade

```
argparse default (None) → YAML config value → CLI override
```

- YAML configs in `configs/*.yaml` are the single source of truth.
- argparse defaults are `None`; missing YAML keys raise an error early.
- CLI arguments override YAML values.
