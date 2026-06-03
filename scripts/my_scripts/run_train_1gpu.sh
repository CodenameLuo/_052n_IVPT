#!/usr/bin/env bash
# IVPT 单卡训练（1×RTX 3090）—— 基于官方 scripts/run_train.sh 改成单卡，已 smoke 验证可跑通
# 用法（从任意目录）： bash scripts/my_scripts/run_train_1gpu.sh
# 与官方版的区别：① 用 python 直跑（不用 torchrun，单卡 use_ddp=False）
#                 ② 下预训练权重时清掉本地 socks 代理 + 走国内 hf-mirror（否则 httpx 报 socks 错）
set -e

# ---------- 路径（换机器改这两行）----------
ENV_PY="/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_004n_IVPT/bin/python"
REPO="/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
cd "$REPO"

# ---------- 下权重的网络设置 ----------
# 机器上有本地 socks 代理(127.0.0.1:10808)，会让 huggingface 的 httpx 走 socks 报错；
# 这里清掉代理 + 用国内镜像直连，最稳。权重只第一次下，之后缓存在 ~/.cache/huggingface 复用。
export HF_ENDPOINT=https://hf-mirror.com
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

# ---------- 训练超参（对齐官方 run_train.sh）----------
MODEL_ARCH="vit_base_patch14_reg4_dinov2.lvd142m"
DATA_ROOT="./datasets"
SNAPSHOT_DIR="./snapshot"
BATCH_SIZE=4          # 单卡 3090 24G + 518 分辨率 + 冻结 backbone，batch 4 稳妥；显存有余可调大
EPOCHS=25
N_PRO="17,14,11,8,5"

"$ENV_PY" train_net.py \
    --model_arch ${MODEL_ARCH} \
    --pretrained_start_weights \
    --data_path ${DATA_ROOT}/CUB_200_2011 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dataset cub \
    --save_every_n_epochs 10 \
    --num_workers 4 \
    --image_sub_path_train images \
    --image_sub_path_test images \
    --train_split 1 \
    --eval_mode test \
    --snapshot_dir ${SNAPSHOT_DIR} \
    --lr 1e-6 \
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
    --image_size 518 \
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
    --modulation_type layer_norm \
    --grad_norm_clip 2.0 \
    --n_pro ${N_PRO} \
    --epoch_fraction 1.0 \
    --eval_every_n_epochs 5 \
    --eval_fraction 0.1
