#!/usr/bin/env bash
# ============================================================================
# IVPT Classification Evaluation Script
# ============================================================================
# Evaluate a trained IVPT model on CUB-200-2011 (classification accuracy).
#
# Usage (from project root):
#   bash scripts/run_test.sh
#
# This is the same as run_train.sh but with --eval_only appended.
# Make sure SNAPSHOT_DIR points to the directory containing your checkpoint.
# ============================================================================

set -e

# cd to project root (parent of scripts/)
cd "$(dirname "$0")/.."


# ---------- Paths ----------
MODEL_ARCH="vit_base_patch14_reg4_dinov2.lvd142m"
DATA_ROOT="./datasets"
SNAPSHOT_DIR="./snapshot"

# ---------- Distributed ----------
NUM_NODES=1
NUM_GPUS=1

# ---------- Evaluation ----------
BATCH_SIZE=12
EPOCHS=25
NUM_WORKERS=4
IMAGE_SIZE=518

# ---------- Model ----------
N_PRO="17,14,11,8,5"
MODULATION_TYPE="layer_norm"

# ============================================================================
torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=${NUM_GPUS} \
    train_net.py \
    --model_arch ${MODEL_ARCH} \
    --pretrained_start_weights \
    --data_path ${DATA_ROOT}/CUB_200_2011 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dataset cub \
    --save_every_n_epochs 8 \
    --num_workers ${NUM_WORKERS} \
    --image_sub_path_train images \
    --image_sub_path_test images \
    --train_split 1 \
    --eval_mode test \
    --snapshot_dir ${SNAPSHOT_DIR} \
    --lr 0.866e-6 \
    --optimizer_type adam \
    --scheduler_type steplr \
    --scheduler_gamma 0.5 \
    --scheduler_step_size 4 \
    --scratch_lr_factor 1e4 \
    --modulation_lr_factor 1e4 \
    --finer_lr_factor 2e2 \
    --drop_path 0.0 \
    --smoothing 0 \
    --augmentations_to_use cub_original \
    --image_size ${IMAGE_SIZE} \
    --weight_decay 0 \
    --classification_loss 1 \
    --presence_loss 1 \
    --equivariance_loss 1 \
    --total_variation_loss 1 \
    --enforced_presence_loss 1 \
    --enforced_presence_loss_type enforced_presence \
    --pixel_wise_entropy_loss 1 \
    --gumbel_softmax \
    --freeze_backbone \
    --presence_loss_type original \
    --modulation_type ${MODULATION_TYPE} \
    --grad_norm_clip 2.0 \
    --n_pro ${N_PRO} \
    --eval_only
