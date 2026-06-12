#!/usr/bin/env bash
# ============================================================================
# IVPT Training Script
# ============================================================================
# Multi-GPU training on CUB-200-2011 using DINOv2 ViT-Base backbone.
#
# Usage (from project root):
#   bash scripts/run_train.sh
#
# Before running, adjust DATA_ROOT and SNAPSHOT_DIR to match your setup.
# ============================================================================

set -e  # exit on error

# cd to project root (parent of scripts/)
cd "$(dirname "$0")/.."


# ---------- Paths ----------
MODEL_ARCH="vit_base_patch14_reg4_dinov2.lvd142m"
DATA_ROOT="./datasets"
SNAPSHOT_DIR="./snapshot"

# ---------- Distributed ----------
NUM_NODES=1
NUM_GPUS=1       # number of GPUs on this machine

# ---------- Training ----------
BATCH_SIZE=12          # per-GPU batch size
EPOCHS=25
SAVE_EVERY=10
NUM_WORKERS=4
IMAGE_SIZE=518
EPOCH_FRACTION=1.0    # fraction of training data per epoch (1.0 = full)
EVAL_EVERY=5          # run eval every N epochs
EVAL_FRACTION=0.1     # fraction of test data for periodic eval (final eval always uses full set)

# ---------- Optimizer & Scheduler ----------
# LR=1e-6
LR=0.866e-6
OPTIMIZER="adam"
SCHEDULER="steplr"
SCHEDULER_GAMMA=0.5
SCHEDULER_STEP_SIZE=4
SCRATCH_LR_FACTOR=1e4
MODULATION_LR_FACTOR=1e4
FINER_LR_FACTOR=2e2
WEIGHT_DECAY=0
GRAD_NORM_CLIP=2.0

# ---------- Loss weights ----------
CLASSIFICATION_LOSS=1
PRESENCE_LOSS=1
EQUIVARIANCE_LOSS=1
TOTAL_VARIATION_LOSS=1
ENFORCED_PRESENCE_LOSS=1
PIXEL_WISE_ENTROPY_LOSS=1

# ---------- Model ----------
N_PRO="17,14,11,8,5"
MODULATION_TYPE="layer_norm"
PRESENCE_LOSS_TYPE="original"
ENFORCED_PRESENCE_LOSS_TYPE="enforced_presence"

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
    --save_every_n_epochs ${SAVE_EVERY} \
    --num_workers ${NUM_WORKERS} \
    --image_sub_path_train images \
    --image_sub_path_test images \
    --train_split 1 \
    --eval_mode test \
    --snapshot_dir ${SNAPSHOT_DIR} \
    --lr ${LR} \
    --optimizer_type ${OPTIMIZER} \
    --scheduler_type ${SCHEDULER} \
    --scheduler_gamma ${SCHEDULER_GAMMA} \
    --scheduler_step_size ${SCHEDULER_STEP_SIZE} \
    --scratch_lr_factor ${SCRATCH_LR_FACTOR} \
    --modulation_lr_factor ${MODULATION_LR_FACTOR} \
    --finer_lr_factor ${FINER_LR_FACTOR} \
    --drop_path 0.0 \
    --smoothing 0 \
    --augmentations_to_use cub_original \
    --image_size ${IMAGE_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --classification_loss ${CLASSIFICATION_LOSS} \
    --presence_loss ${PRESENCE_LOSS} \
    --equivariance_loss ${EQUIVARIANCE_LOSS} \
    --total_variation_loss ${TOTAL_VARIATION_LOSS} \
    --enforced_presence_loss ${ENFORCED_PRESENCE_LOSS} \
    --enforced_presence_loss_type ${ENFORCED_PRESENCE_LOSS_TYPE} \
    --pixel_wise_entropy_loss ${PIXEL_WISE_ENTROPY_LOSS} \
    --gumbel_softmax \
    --freeze_backbone \
    --presence_loss_type ${PRESENCE_LOSS_TYPE} \
    --modulation_type ${MODULATION_TYPE} \
    --grad_norm_clip ${GRAD_NORM_CLIP} \
    --n_pro ${N_PRO} \
    --epoch_fraction ${EPOCH_FRACTION} \
    --eval_every_n_epochs ${EVAL_EVERY} \
    --eval_fraction ${EVAL_FRACTION}
