#!/bin/bash
# Master Pipeline: Separated Experts + OPD
#
# Stages:
#   1    Text Pretrain (pre-run)
#   2    Visual Pretrain (COCO images + box/point)
#   2M   Merge LoRA into base
#   3a   Box Expert SFT
#   3b   Point Expert SFT
#   4a   Box Expert GRPO
#   4b   Point Expert GRPO
#   5    Unified RFT (experts as generators)
#   6    OPD (reverse KL distillation)
#
# Each stage checks for prerequisite outputs before running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================================"
echo "IterDPO Pipeline: Separated Experts + OPD"
echo "============================================================"

# ── Stage 1: Pretrain (Embedding-only) ──────────────────────────────────
STAGE1_DIR="outputs/stage1_pretrain"
STAGE1_STATE="${STAGE1_DIR}/pretrain_state_dict.pt"

if [ -f "$STAGE1_STATE" ]; then
    echo "✅ Stage 1 Pretrain already done (${STAGE1_STATE})"
else
    echo "❌ Stage 1 must be run first: python scripts/run_stage1_pretrain.py"
    exit 1
fi

# ── Stage 2: Visual Pretrain ──────────────────────────────────
STAGE2_DIR="outputs/stage2_visual_pretrain"
STAGE2_ADAPTER="${STAGE2_DIR}/adapter_model.safetensors"

if [ -f "$STAGE2_ADAPTER" ]; then
    echo "✅ Stage 2 Visual Pretrain already done"
else
    echo "🔄 Running Stage 2: Visual Pretrain..."
    python scripts/run_stage2_visual_pretrain.py \
        --model_path models/Qwen3-VL-4B-Thinking \
        --pretrain_embedding_path outputs/stage1_pretrain \
        --output_dir outputs/stage2_visual_pretrain \
        --num_epochs 1 --batch_size 2 --gradient_accumulation_steps 2
    echo "✅ Stage 2 complete."
fi

# ── Merge Stage 2 LoRA ────────────────────────────────────────
MERGED_DIR="outputs/stage2_merged_base"
MERGED_CONFIG="${MERGED_DIR}/config.json"

if [ -f "$MERGED_CONFIG" ]; then
    echo "✅ Stage 2 Merge already done"
else
    echo "🔄 Merging Stage 2 LoRA into base model..."
    python scripts/merge_stage2.py \
        --base_model models/Qwen3-VL-4B-Thinking \
        --adapter_path outputs/stage2_visual_pretrain \
        --output_dir outputs/stage2_merged_base
    echo "✅ Merge complete."
fi

# ── Stage 3a: Box Expert SFT ──────────────────────────────────
STAGE3A_DIR="outputs/stage3a_sft_box"
STAGE3A_ADAPTER="${STAGE3A_DIR}/adapter_model.safetensors"

if [ -f "$STAGE3A_ADAPTER" ]; then
    echo "✅ Stage 3a Box Expert SFT already done"
else
    echo "🔄 Running Stage 3a: Box Expert SFT..."
    python scripts/run_stage3a_sft_box.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage3a_sft_box
    echo "✅ Stage 3a complete."
fi

# ── Stage 3b: Point Expert SFT ────────────────────────────────
STAGE3B_DIR="outputs/stage3b_sft_point"
STAGE3B_ADAPTER="${STAGE3B_DIR}/adapter_model.safetensors"

if [ -f "$STAGE3B_ADAPTER" ]; then
    echo "✅ Stage 3b Point Expert SFT already done"
else
    echo "🔄 Running Stage 3b: Point Expert SFT..."
    python scripts/run_stage3b_sft_point.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage3b_sft_point
    echo "✅ Stage 3b complete."
fi

# ── Stage 4a: Box Expert GRPO ─────────────────────────────────
STAGE4A_DIR="outputs/stage4a_grpo_box"
STAGE4A_FINAL="${STAGE4A_DIR}/round_3/adapter_model.safetensors"

if [ -f "$STAGE4A_FINAL" ]; then
    echo "✅ Stage 4a Box Expert GRPO already done"
else
    echo "🔄 Running Stage 4a: Box Expert GRPO..."
    python scripts/run_stage4a_grpo_box.py \
        --model_path outputs/stage3a_sft_box \
        --output_dir outputs/stage4a_grpo_box
    echo "✅ Stage 4a complete."
fi

# ── Stage 4b: Point Expert GRPO ───────────────────────────────
STAGE4B_DIR="outputs/stage4b_grpo_point"
STAGE4B_FINAL="${STAGE4B_DIR}/round_3/adapter_model.safetensors"

if [ -f "$STAGE4B_FINAL" ]; then
    echo "✅ Stage 4b Point Expert GRPO already done"
else
    echo "🔄 Running Stage 4b: Point Expert GRPO..."
    python scripts/run_stage4b_grpo_point.py \
        --model_path outputs/stage3b_sft_point \
        --output_dir outputs/stage4b_grpo_point
    echo "✅ Stage 4b complete."
fi

# ── Stage 5: Unified RFT ──────────────────────────────────────
STAGE5_DIR="outputs/stage5_rft_unified"
STAGE5_FINAL="${STAGE5_DIR}/final_model/adapter_model.safetensors"

if [ -f "$STAGE5_FINAL" ]; then
    echo "✅ Stage 5 Unified RFT already done"
else
    echo "🔄 Running Stage 5: Unified RFT..."
    python scripts/run_stage5_rft_unified.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage5_rft_unified
    echo "✅ Stage 5 complete."
fi

# ── Stage 6: OPD ──────────────────────────────────────────────
STAGE6_DIR="outputs/stage6_opd"
STAGE6_ADAPTER="${STAGE6_DIR}/adapter_model.safetensors"

if [ -f "$STAGE6_ADAPTER" ]; then
    echo "✅ Stage 6 OPD already done"
else
    echo "🔄 Running Stage 6: OPD..."
    python scripts/run_stage6_opd.py \
        --student_path outputs/stage5_rft_unified/final_model \
        --output_dir outputs/stage6_opd
    echo "✅ Stage 6 complete."
fi

echo "============================================================"
echo "🎉 Full pipeline complete!"
echo "Final model: outputs/stage6_opd/"
echo "============================================================"
