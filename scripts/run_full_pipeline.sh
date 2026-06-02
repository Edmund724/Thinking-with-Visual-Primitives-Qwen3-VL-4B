#!/usr/bin/env bash
# TVP-4B-5090D: Full 4-Stage Pipeline Runner
# Stage 0: Pretrain -> Stage 1: SFT Unified -> Stage 2: GRPO -> Stage 3: RFT
#
# Usage:
#   bash scripts/run_full_pipeline.sh [--skip-data-check] [--from-stage N]
#
# WARNING: This pipeline runs ~50-60 hours on single RTX 5090D.
# Each stage saves checkpoints; you can resume from any stage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse args
FROM_STAGE=0
SKIP_DATA_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-stage) FROM_STAGE="$2"; shift 2 ;;
        --skip-data-check) SKIP_DATA_CHECK=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================"
echo "TVP-4B-5090D: Full 4-Stage Pipeline"
echo "Starting from Stage $FROM_STAGE"
echo "================================================"

# Ensure dependencies
pip install -q -r requirements.txt 2>&1 | tail -1
echo "Dependencies OK."

mkdir -p outputs logs data

# =====================================================
# Stage 0: Pretrain (COCO Box Grounding — No Thinking)
# =====================================================
if [ "$FROM_STAGE" -le 0 ]; then
    echo ""
    echo "================================================"
    echo "Stage 0/3: Pretrain (COCO Box Grounding)"
    echo "  - No Chain-of-Thought"
    echo "  - Pure grounding: see object -> output box"
    echo "  - Following paper's curriculum learning"
    echo "Estimated: ~10 hours"
    echo "================================================"

    COCO_DIR="data/coco"
    if [ ! -f "$COCO_DIR/annotations/instances_train2017.json" ] && [ "$SKIP_DATA_CHECK" = false ]; then
        echo "WARNING: COCO data not found at $COCO_DIR"
        echo "Please download first:"
        echo "  wget http://images.cocodataset.org/zips/train2017.zip -P data/coco"
        echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco"
        echo "  unzip data/coco/train2017.zip -d data/coco"
        echo "  unzip data/coco/annotations_trainval2017.zip -d data/coco"
        exit 1
    fi

    python scripts/run_stage0_pretrain.py \
        --config configs/stage0_pretrain.yaml \
        --coco_image_dir "$COCO_DIR/train2017" \
        --coco_ann_file "$COCO_DIR/annotations/instances_train2017.json" \
        --num_coco 40000
fi

# =====================================================
# Stage 1: SFT Unified (Box + Maze + Path + Thinking)
# =====================================================
if [ "$FROM_STAGE" -le 1 ]; then
    echo ""
    echo "================================================"
    echo "Stage 1/3: SFT Unified"
    echo "  - COCO Box + Synthetic Maze + Synthetic Path"
    echo "  - Chain-of-Thought reasoning added"
    echo "  - Loads from Stage 0 adapter (curriculum)"
    echo "Estimated: ~27 hours"
    echo "================================================"

    MODEL_PATH=""
    if [ -d "outputs/stage0_pretrain" ]; then
        MODEL_PATH="outputs/stage0_pretrain"
        echo "Using Stage 0 checkpoint: $MODEL_PATH"
    else
        echo "WARNING: Stage 0 checkpoint not found, using base model from config"
    fi

    python scripts/run_stage1_sft_unified.py \
        --config configs/stage1_sft_unified.yaml \
        ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
        --coco_image_dir data/coco/train2017 \
        --coco_ann_file data/coco/annotations/instances_train2017.json
fi

# =====================================================
# Stage 2: GRPO (3 rounds with tightening thresholds)
# =====================================================
if [ "$FROM_STAGE" -le 2 ]; then
    echo ""
    echo "================================================"
    echo "Stage 2/3: GRPO (Group Relative Policy Optimization)"
    echo "  - 3 rounds, tightening IoU / point-dist thresholds"
    echo "Estimated: ~20-25 hours"
    echo "================================================"

    MODEL_PATH=""
    if [ -d "outputs/stage1_sft_unified" ]; then
        MODEL_PATH="outputs/stage1_sft_unified"
        echo "Using Stage 1 checkpoint: $MODEL_PATH"
    else
        echo "WARNING: Stage 1 checkpoint not found, using base model from config"
    fi

    python scripts/run_stage2_grpo.py \
        --config configs/stage2_grpo.yaml \
        ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
        --coco_image_dir data/coco/train2017 \
        --coco_ann_file data/coco/annotations/instances_train2017.json \
        --num_rounds 3
fi

# =====================================================
# Stage 3: RFT (Rejection Sampling Fine-Tuning)
# =====================================================
if [ "$FROM_STAGE" -le 3 ]; then
    echo ""
    echo "================================================"
    echo "Stage 3/3: RFT (Rejection Sampling Fine-Tuning)"
    echo "  - Rollout 5x per sample"
    echo "  - Filter by process reward"
    echo "  - SFT on high-quality subset"
    echo "Estimated: ~3-5 hours"
    echo "================================================"

    MODEL_PATH=""
    if [ -d "outputs/stage2_grpo/round_3" ]; then
        MODEL_PATH="outputs/stage2_grpo/round_3"
        echo "Using Stage 2 Round 3 checkpoint: $MODEL_PATH"
    elif [ -d "outputs/stage2_grpo" ]; then
        MODEL_PATH="$(find outputs/stage2_grpo -maxdepth 1 -type d -name 'round_*' | sort | tail -1)"
        echo "Using latest GRPO checkpoint: $MODEL_PATH"
    else
        echo "WARNING: Stage 2 checkpoint not found, using base model from config"
    fi

    python scripts/run_stage3_rft.py \
        --config configs/stage3_rft.yaml \
        ${MODEL_PATH:+--model_path "$MODEL_PATH"} \
        --coco_image_dir data/coco/train2017 \
        --coco_ann_file data/coco/annotations/instances_train2017.json \
        --accept_threshold 1.2
fi

echo ""
echo "================================================"
echo "Pipeline Complete!"
echo "Final model: outputs/stage3_rft/final_model"
echo "================================================"
