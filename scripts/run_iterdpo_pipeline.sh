#!/bin/bash
# Master Pipeline: Separated Experts + OPD
#
# Stages:
#   0.0  Text Pretrain (pre-run)
#   0.5  Visual Pretrain (COCO images + box/point)
#   0.5M Merge LoRA into base
#   1a   Box Expert SFT
#   1b   Point Expert SFT
#   2a   Box Expert GRPO
#   2b   Point Expert GRPO
#   3    Unified RFT (experts as generators)
#   4    OPD (reverse KL distillation)
#
# Each stage checks for prerequisite outputs before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================================"
echo "IterDPO Pipeline: Separated Experts + OPD"
echo "============================================================"

# ── Stage 0: Text Pretrain ──────────────────────────────────
PRETRAIN_DIR="outputs/stage0_pretrain"
PRETRAIN_STATE="${PRETRAIN_DIR}/pretrain_state_dict.pt"

if [ -f "$PRETRAIN_STATE" ]; then
    echo "✅ Stage 0 Text Pretrain already done (${PRETRAIN_STATE})"
else
    echo "❌ Stage 0 must be run first: python scripts/run_pretrain.py"
    exit 1
fi

# ── Stage 0.5: Visual Pretrain ──────────────────────────────
STAGE05_DIR="outputs/stage0_5_visual_pretrain"
STAGE05_ADAPTER="${STAGE05_DIR}/adapter_model.safetensors"

if [ -f "$STAGE05_ADAPTER" ]; then
    echo "✅ Stage 0.5 Visual Pretrain already done"
else
    echo "🔄 Running Stage 0.5: Visual Pretrain..."
    python scripts/run_stage0_5_visual_pretrain.py \
        --model_path models/Qwen3-VL-4B-Thinking \
        --pretrain_embedding_path outputs/stage0_pretrain \
        --output_dir outputs/stage0_5_visual_pretrain
    echo "✅ Stage 0.5 complete."
fi

# ── Merge Stage 0.5 LoRA ────────────────────────────────────
MERGED_DIR="outputs/stage0_5_merged_base"
MERGED_CONFIG="${MERGED_DIR}/config.json"

if [ -f "$MERGED_CONFIG" ]; then
    echo "✅ Stage 0.5 Merge already done"
else
    echo "🔄 Merging Stage 0.5 LoRA into base model..."
    python scripts/merge_stage0_5.py \
        --base_model models/Qwen3-VL-4B-Thinking \
        --adapter_path outputs/stage0_5_visual_pretrain \
        --output_dir outputs/stage0_5_merged_base
    echo "✅ Merge complete."
fi

# ── Stage 1a: Box Expert SFT ────────────────────────────────
STAGE1A_DIR="outputs/stage1a_sft_box"
STAGE1A_ADAPTER="${STAGE1A_DIR}/adapter_model.safetensors"

if [ -f "$STAGE1A_ADAPTER" ]; then
    echo "✅ Stage 1a Box Expert SFT already done"
else
    echo "🔄 Running Stage 1a: Box Expert SFT..."
    python scripts/run_stage1a_sft_box.py \
        --model_path outputs/stage0_5_merged_base \
        --output_dir outputs/stage1a_sft_box
    echo "✅ Stage 1a complete."
fi

# ── Stage 1b: Point Expert SFT ──────────────────────────────
STAGE1B_DIR="outputs/stage1b_sft_point"
STAGE1B_ADAPTER="${STAGE1B_DIR}/adapter_model.safetensors"

if [ -f "$STAGE1B_ADAPTER" ]; then
    echo "✅ Stage 1b Point Expert SFT already done"
else
    echo "🔄 Running Stage 1b: Point Expert SFT..."
    python scripts/run_stage1b_sft_point.py \
        --model_path outputs/stage0_5_merged_base \
        --output_dir outputs/stage1b_sft_point
    echo "✅ Stage 1b complete."
fi

# ── Stage 2a: Box Expert GRPO ───────────────────────────────
STAGE2A_DIR="outputs/stage2a_grpo_box"
STAGE2A_FINAL="${STAGE2A_DIR}/round_3/adapter_model.safetensors"

if [ -f "$STAGE2A_FINAL" ]; then
    echo "✅ Stage 2a Box Expert GRPO already done"
else
    echo "🔄 Running Stage 2a: Box Expert GRPO..."
    python scripts/run_stage2a_grpo_box.py \
        --model_path outputs/stage1a_sft_box \
        --output_dir outputs/stage2a_grpo_box
    echo "✅ Stage 2a complete."
fi

# ── Stage 2b: Point Expert GRPO ─────────────────────────────
STAGE2B_DIR="outputs/stage2b_grpo_point"
STAGE2B_FINAL="${STAGE2B_DIR}/round_3/adapter_model.safetensors"

if [ -f "$STAGE2B_FINAL" ]; then
    echo "✅ Stage 2b Point Expert GRPO already done"
else
    echo "🔄 Running Stage 2b: Point Expert GRPO..."
    python scripts/run_stage2b_grpo_point.py \
        --model_path outputs/stage1b_sft_point \
        --output_dir outputs/stage2b_grpo_point
    echo "✅ Stage 2b complete."
fi

# ── Stage 3: Unified RFT ────────────────────────────────────
STAGE3_DIR="outputs/stage3_rft_unified"
STAGE3_FINAL="${STAGE3_DIR}/final_model/adapter_model.safetensors"

if [ -f "$STAGE3_FINAL" ]; then
    echo "✅ Stage 3 Unified RFT already done"
else
    echo "🔄 Running Stage 3: Unified RFT..."
    python scripts/run_stage3_rft_unified.py \
        --model_path outputs/stage0_5_merged_base \
        --output_dir outputs/stage3_rft_unified
    echo "✅ Stage 3 complete."
fi

# ── Stage 4: OPD ────────────────────────────────────────────
STAGE4_DIR="outputs/stage4_opd"
STAGE4_ADAPTER="${STAGE4_DIR}/adapter_model.safetensors"

if [ -f "$STAGE4_ADAPTER" ]; then
    echo "✅ Stage 4 OPD already done"
else
    echo "🔄 Running Stage 4: OPD..."
    python scripts/run_stage4_opd.py \
        --student_path outputs/stage3_rft_unified/final_model \
        --output_dir outputs/stage4_opd
    echo "✅ Stage 4 complete."
fi

echo "============================================================"
echo "🎉 Full pipeline complete!"
echo "Final model: outputs/stage4_opd/"
echo "============================================================"
