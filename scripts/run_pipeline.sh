#!/bin/bash
# Master Pipeline: Separated Experts + OPD
#
# Stages:
#   1    Unified Visual Grounding Pretrain (COCO + CLEVR)
#   2    Merge LoRA into base
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

# Helper: find latest checkpoint-* directory under a stage output dir.
_latest_checkpoint() {
    local dir="$1"
    ls -d "${dir}/checkpoint-"* 2>/dev/null | sort -V | tail -n 1
}

# Helper: find latest checkpoint-* directory inside GRPO round_* subdirs.
_latest_grpo_checkpoint() {
    local dir="$1"
    find "$dir" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1
}

echo "============================================================"
echo "TVP Pipeline: Separated Experts + OPD"
echo "============================================================"

# ── Stage 1: Unified Visual Grounding Pretrain ──────────────────────
STAGE1_DIR="outputs/stage1_visual_pretrain"
STAGE1_ADAPTER="${STAGE1_DIR}/adapter_model.safetensors"

if [ -f "$STAGE1_ADAPTER" ]; then
    echo "✅ Stage 1 Visual Pretrain already done"
else
    LATEST_CKPT=$(_latest_checkpoint "$STAGE1_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 1: Unified Visual Grounding Pretrain from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 1: Unified Visual Grounding Pretrain..."
    fi
    python scripts/run_stage1_visual_pretrain.py \
        --model_path models/Qwen3-VL-4B-Thinking \
        --output_dir outputs/stage1_visual_pretrain
    echo "✅ Stage 1 complete."
fi

# Stage 2: Merge Stage 1 LoRA ──
MERGED_DIR="outputs/stage2_merged_base"
MERGED_CONFIG="${MERGED_DIR}/config.json"

if [ -f "$MERGED_CONFIG" ]; then
    echo "✅ Stage 2 Merge already done"
else
    echo "🔄 Running Stage 2: Merge LoRA into base..."
    python scripts/run_stage2_merge.py \
        --base_model models/Qwen3-VL-4B-Thinking \
        --adapter_path outputs/stage1_visual_pretrain \
        --output_dir outputs/stage2_merged_base
    echo "✅ Merge complete."
fi

# ── Stage 3a: Box Expert SFT ──────────────────────────────────────
STAGE3A_DIR="outputs/stage3a_sft_box"
STAGE3A_ADAPTER="${STAGE3A_DIR}/adapter_model.safetensors"

if [ -f "$STAGE3A_ADAPTER" ]; then
    echo "✅ Stage 3a Box Expert SFT already done"
else
    # If previous run was interrupted, resume from the latest checkpoint automatically.
    LATEST_CKPT=$(ls -d "${STAGE3A_DIR}/checkpoint-"* 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 3a: Box Expert SFT from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 3a: Box Expert SFT..."
    fi
    python scripts/run_stage3a_sft_box.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage3a_sft_box
    echo "✅ Stage 3a complete."
fi

# ── Stage 3b: Point Expert SFT ────────────────────────────────────
STAGE3B_DIR="outputs/stage3b_sft_point"
STAGE3B_ADAPTER="${STAGE3B_DIR}/adapter_model.safetensors"

if [ -f "$STAGE3B_ADAPTER" ]; then
    echo "✅ Stage 3b Point Expert SFT already done"
else
    LATEST_CKPT=$(_latest_checkpoint "$STAGE3B_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 3b: Point Expert SFT from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 3b: Point Expert SFT..."
    fi
    python scripts/run_stage3b_sft_point.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage3b_sft_point
    echo "✅ Stage 3b complete."
fi

# ── Stage 4a: Box Expert GRPO ─────────────────────────────────────
STAGE4A_DIR="outputs/stage4a_grpo_box"
STAGE4A_FINAL="${STAGE4A_DIR}/round_2/adapter_model.safetensors"

if [ -f "$STAGE4A_FINAL" ]; then
    echo "✅ Stage 4a Box Expert GRPO already done"
else
    LATEST_CKPT=$(_latest_grpo_checkpoint "$STAGE4A_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 4a: Box Expert GRPO from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 4a: Box Expert GRPO..."
    fi
    python scripts/run_stage4a_grpo_box.py \
        --model_path outputs/stage3a_sft_box \
        --output_dir outputs/stage4a_grpo_box
    echo "✅ Stage 4a complete."
fi

# ── Stage 4b: Point Expert GRPO ───────────────────────────────────
STAGE4B_DIR="outputs/stage4b_grpo_point"
STAGE4B_FINAL="${STAGE4B_DIR}/round_2/adapter_model.safetensors"

if [ -f "$STAGE4B_FINAL" ]; then
    echo "✅ Stage 4b Point Expert GRPO already done"
else
    LATEST_CKPT=$(_latest_grpo_checkpoint "$STAGE4B_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 4b: Point Expert GRPO from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 4b: Point Expert GRPO..."
    fi
    python scripts/run_stage4b_grpo_point.py \
        --model_path outputs/stage3b_sft_point \
        --output_dir outputs/stage4b_grpo_point
    echo "✅ Stage 4b complete."
fi

# ── Stage 5: Unified RFT ──────────────────────────────────────────
STAGE5_DIR="outputs/stage5_rft_unified"
STAGE5_FINAL="${STAGE5_DIR}/final_model/adapter_model.safetensors"

if [ -f "$STAGE5_FINAL" ]; then
    echo "✅ Stage 5 Unified RFT already done"
else
    LATEST_CKPT=$(_latest_checkpoint "$STAGE5_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 5: Unified RFT from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 5: Unified RFT..."
    fi
    python scripts/run_stage5_rft_unified.py \
        --model_path outputs/stage2_merged_base \
        --output_dir outputs/stage5_rft_unified
    echo "✅ Stage 5 complete."
fi

# ── Stage 6: OPD ──────────────────────────────────────────────────
STAGE6_DIR="outputs/stage6_opd"
STAGE6_ADAPTER="${STAGE6_DIR}/adapter_model.safetensors"

if [ -f "$STAGE6_ADAPTER" ]; then
    echo "✅ Stage 6 OPD already done"
else
    LATEST_CKPT=$(_latest_checkpoint "$STAGE6_DIR")
    if [ -n "$LATEST_CKPT" ]; then
        echo "🔄 Resuming Stage 6: OPD from ${LATEST_CKPT}..."
    else
        echo "🔄 Running Stage 6: OPD..."
    fi
    python scripts/run_stage6_opd.py \
        --student_path outputs/stage5_rft_unified/final_model \
        --output_dir outputs/stage6_opd
    echo "✅ Stage 6 complete."
fi

echo "============================================================"
echo "🎉 Full pipeline complete!"
echo "Final model: outputs/stage6_opd/"
echo "============================================================"
