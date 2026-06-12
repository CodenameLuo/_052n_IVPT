#!/usr/bin/env bash
# ============================================================================
# IVPT Interpretability Evaluation Script
# ============================================================================
# Evaluate interpretability metrics (NMI, ARI, KPR, FgBgIoU) on CUB-200.
#
# Usage (from project root):
#   bash scripts/run_eval.sh
#
# Adjust DATA_ROOT, SNAPSHOT_DIR, and eval_mode as needed.
# Supported eval modes: nmi_ari | kpr | fg_bg_iou
# ============================================================================

set -e

# cd to project root (parent of scripts/)
cd "$(dirname "$0")/.."

# ---------- Add project root to PYTHONPATH ----------
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# ---------- Paths ----------
MODEL_ARCH="vit_base_patch14_reg4_dinov2.lvd142m"
DATA_ROOT="./datasets"
SNAPSHOT_DIR="./snapshot"
OUTPUT_DIR="${SNAPSHOT_DIR}/output"

# ---------- Evaluation settings ----------
EVAL_MODE="nmi_ari"     # nmi_ari | kpr | fg_bg_iou
NUM_PARTS=4
BATCH_SIZE=12
NUM_WORKERS=4
IMAGE_SIZE=518
HALF_SIZE=36

# ---------- Model ----------
N_PRO="17,14,11,8,5"
MODULATION_TYPE="layer_norm"

# ============================================================================
python eval/evaluate_consistency.py \
    --model_path ${SNAPSHOT_DIR}/snapshot_best.pt \
    --dataset cub \
    --center_crop \
    --eval_mode ${EVAL_MODE} \
    --num_parts ${NUM_PARTS} \
    --model_arch ${MODEL_ARCH} \
    --data_path ${DATA_ROOT}/cub200_cropped \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --image_size ${IMAGE_SIZE} \
    --modulation_type ${MODULATION_TYPE} \
    --half_size ${HALF_SIZE} \
    --n_pro ${N_PRO} \
    --output_dir ${OUTPUT_DIR}
