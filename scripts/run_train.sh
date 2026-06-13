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

# ============================================================================
# 这个脚本是训练的“启动器”：它本身不写任何训练逻辑，只是把一堆超参数
# (路径 / 卡数 / batch / 学习率 / 损失权重 / 模型设置等) 整理成命令行参数，
# 再用 torchrun 把它们一次性喂给 train_net.py 去跑。
# 真正的训练流程在 train_net.py::ivpt_train_eval() 里。
#
# 注：下面有不少 “# XXX=...” 被注释掉的行，是作者(论文)的原始多卡配置；
#     当前生效的是紧跟其后、没被注释的那一行——已改成单卡(NUM_GPUS=1)、batch=12 的本机配置。
# ============================================================================

# set -e：脚本中任意一条命令出错(返回非 0)就立即整体退出，避免带着错误状态继续往下跑
set -e  # exit on error

# 把工作目录切到项目根目录(scripts/ 的上一级)，这样后面写相对路径(如 ./datasets)才对得上
# "$(dirname "$0")" 取本脚本所在目录(scripts/)，再 /.. 回到根目录
# cd to project root (parent of scripts/)
cd "$(dirname "$0")/.."


# ---------- 路径相关 ----------
# ---------- Paths ----------
# 主干网络结构名(timm 模型名)：reg4 = 带 4 个 register token 的 DINOv2 ViT-Base，patch14、预训练于 LVD-142M
MODEL_ARCH="vit_base_patch14_reg4_dinov2.lvd142m"
# 数据集根目录(下面会拼成 ${DATA_ROOT}/CUB_200_2011)
DATA_ROOT="./datasets"
# 检查点(模型权重)保存目录
SNAPSHOT_DIR="./snapshot"

# ---------- 分布式相关 ----------
# ---------- Distributed ----------
# 机器(节点)数量，单机就是 1
NUM_NODES=1
# 每个节点用几张 GPU；论文原配置是 4 卡，这里改成单卡
# NUM_GPUS=4       # number of GPUs on this machine
NUM_GPUS=1       # number of GPUs on this machine

# ---------- 训练相关 ----------
# ---------- Training ----------
# 每张卡的 batch size(总 batch = BATCH_SIZE × NUM_GPUS)；论文原配置每卡 4，这里改成 12
# BATCH_SIZE=4          # per-GPU batch size
BATCH_SIZE=12          # per-GPU batch size
# 总训练轮数
EPOCHS=25
# 每隔多少个 epoch 存一次检查点
SAVE_EVERY=10
# DataLoader 取数据用的子进程数
NUM_WORKERS=4
# 输入图像分辨率(DINOv2 默认 518)
IMAGE_SIZE=518
# 每个 epoch 用训练数据的多少比例(1.0=全量)；设小于 1 可加快单轮、靠多轮轮换覆盖全量
EPOCH_FRACTION=1.0    # fraction of training data per epoch (1.0 = full)
# 训练中途每隔多少个 epoch 评估一次
EVAL_EVERY=5          # run eval every N epochs
# 中途评估只用测试集的多少比例(0.1=10%，加快中途评估)；训练结束后的最终评估始终用全量测试集
EVAL_FRACTION=0.1     # fraction of test data for periodic eval (final eval always uses full set)

# ---------- 优化器与调度器 ----------
# ---------- Optimizer & Scheduler ----------
# 学习率：论文基准是 batch=16 → lr=1e-6；按“平方根缩放规则”换算到 batch=12：
#   1e-6 × sqrt(12/16) = 1e-6 × sqrt(0.75) ≈ 1e-6 × 0.866 = 0.866e-6
# LR=1e-6
LR=0.866e-6
# 优化器类型
OPTIMIZER="adam"
# 学习率调度器类型：steplr = 阶梯式衰减
SCHEDULER="steplr"
# StepLR 的衰减系数：每到一个 step 就把学习率乘以 0.5
SCHEDULER_GAMMA=0.5
# StepLR 的步长：每 4 个 epoch 衰减一次
SCHEDULER_STEP_SIZE=4
# 下面三个是“分组学习率倍率”：不同用途的参数组在基准 lr 上再乘一个倍率(见 train_net.py 的参数分组)
# 从头训练层(scratch)的学习率倍率(×1e4)
SCRATCH_LR_FACTOR=1e4
# 调制(modulation)层的学习率倍率(×1e4)
MODULATION_LR_FACTOR=1e4
# finer 层的学习率倍率(×2e2)
FINER_LR_FACTOR=2e2
# 权重衰减(本次为 0)
WEIGHT_DECAY=0
# 梯度裁剪上限：反向后把梯度范数裁到不超过 2.0，防止梯度爆炸
GRAD_NORM_CLIP=2.0

# ---------- 各项损失的权重 ----------
# ---------- Loss weights ----------
# 分类损失(交叉熵)权重
CLASSIFICATION_LOSS=1
# presence 损失：约束每个部件“该出现就出现”(避免部件全程不被激活)
PRESENCE_LOSS=1
# 等变损失：图像做仿射变换后，部件注意力图应同步变换(几何一致性)
EQUIVARIANCE_LOSS=1
# 全变差损失：让部件注意力图在空间上平滑、连成片、不破碎
TOTAL_VARIATION_LOSS=1
# enforced presence 损失：进一步强制部件在前景上有响应
ENFORCED_PRESENCE_LOSS=1
# 像素级熵损失：让每个像素尽量只归属一个部件(分配更“硬”)
PIXEL_WISE_ENTROPY_LOSS=1

# ---------- 模型相关 ----------
# ---------- Model ----------
# 各注入层的原型(prototype)数量，逗号分隔：在最后 5 个 ViT block 分别注入 17/14/11/8/5 个原型
N_PRO="17,14,11,8,5"
# 原型调制类型：layer_norm
MODULATION_TYPE="layer_norm"
# presence 损失的具体形式
PRESENCE_LOSS_TYPE="original"
# enforced presence 损失的具体形式
ENFORCED_PRESENCE_LOSS_TYPE="enforced_presence"

# ============================================================================
# torchrun：PyTorch 的分布式启动器，按 --nproc_per_node 拉起对应数量的训练进程(每进程一张卡)，
# 单卡时就是拉起 1 个进程。下面用 \ 续行把上面所有变量拼成 train_net.py 的命令行参数。
#
# 注：torchrun 的续行命令中间不能插注释(\ 后面必须紧跟换行，否则会断开命令)，
#     所以这里把几个“直接写死、不来自变量的 flag”的含义集中说明在前面：
#   --pretrained_start_weights         : 主干加载预训练权重(需联网，权重缓存在 ~/.cache/torch/hub)
#   --image_sub_path_train/test images : CUB 的训练/测试图都在数据集的 images/ 子目录
#   --train_split 1                    : 用全部官方训练集训练(train_split=1，不再切出验证集)
#   --eval_mode test                   : 评估在测试集上做
#   --drop_path 0.0                    : drop path 概率 0(主干冻结时用不到)
#   --smoothing 0                      : 标签平滑 0(等价于普通交叉熵)
#   --augmentations_to_use cub_original: 用细粒度分类常用的标准增强(而非 timm 的自动增强)
#   --gumbel_softmax                   : 部件注意力图用 Gumbel-Softmax(让“像素→部件”的分配更接近硬分配)
#   --freeze_backbone                  : 冻结主干，只训练新增的部件发现层 / 分类头等
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
