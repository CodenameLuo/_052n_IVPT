"""Miscellaneous utility functions for IVPT.

Includes attention computation, attention rollout, SyncBN conversion,
snapshot directory management, and small numerical helpers.
"""

# ======================================
#
# 一个杂物抽屉。本次训练路径上真正用到的是文件末尾两个：
#   sync_bn_conversion：多卡时把 BN 换成跨卡同步的 SyncBN(本次单卡，不会被调)；
#   check_snapshot    ：train_net.py 第 3 步调用，建好存检查点的目录。
# 另外 file_line_count 被(评估用的)Parts 数据集调用、factors 被可视化调用。
# 中间一段 compute_attention / rollout / compute_* 是注意力可视化/分析用的，训练不碰(下面有横幅)。
#
# ======================================

import math
import os
from functools import reduce
from pathlib import Path

import numpy as np
import torch


# 求 n 的所有因子(可视化里用来把“patch 数”分解成接近正方形的网格行列数)
def factors(n):
    return reduce(list.__add__,
                  ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))


# 数文件行数(被评估用的 Parts 数据集用来数关键点种类数)
def file_line_count(filename: str) -> int:
    """Count the number of lines in a file"""
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)


# ============================================================================
# ↓↓↓ 以下 compute_attention / compute_dot_product_similarity / compute_cross_entropy / rollout
# 都【不在本次训练路径上】↓↓↓ 它们是注意力图可视化、注意力 rollout、相似度/交叉熵等分析小工具，
# 训练 forward 与 _run_batch 都不调用，这里保持原样、不逐行展开。
# ============================================================================

def compute_attention(qkv, scale=None):
    """
    Compute attention matrix (same as in the pytorch scaled dot product attention)
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    :param qkv: Query, key and value tensors concatenated along the first dimension
    :param scale: Scale factor for the attention computation
    :return:
    """
    if isinstance(qkv, torch.Tensor):
        query, key, value = qkv.unbind(0)
    else:
        query, key, value = qkv
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    L, S = query.size(-2), key.size(-2)
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_out = attn_weight @ value
    return attn_weight, attn_out


def compute_dot_product_similarity(a, b):
    scores = a @ b.transpose(-1, -2)
    return scores


def compute_cross_entropy(p, q):
    q = torch.nn.functional.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def rollout(attentions, discard_ratio=0.9, head_fusion="max", device=torch.device("cuda")):
    """
    Perform attention rollout, 
    Ref: https://github.com/jacobgil/vit-explain/blob/15a81d355a5aa6128ea4e71bbd56c28888d0f33b/vit_rollout.py#L9C1-L42C16
    Parameters
    ----------
    attentions : list
        List of attention matrices, one for each transformer layer
    discard_ratio : float
        Ratio of lowest attention values to discard
    head_fusion : str
        Type of fusion to use for attention heads. One of "mean", "max", "min"
    device : torch.device
        Device to use for computation
    Returns
    -------
    mask : np.ndarray
        Mask of shape (width, width), where width is the square root of the number of patches
    """
    result = torch.eye(attentions[0].size(-1), device=device)
    attentions = [attention.to(device) for attention in attentions]
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1).values
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1).values
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1), device=device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Normalize the result by max value in each row
    result = result / result.max(dim=-1, keepdim=True)[0]
    return result


# ============================================================================
# ↓↓↓ 以下两个【在本次训练路径上】↓↓↓
# ============================================================================

# 多卡时把模型里所有 BatchNorm 换成跨卡同步版 SyncBatchNorm(让各卡共享统计量)
# train_net.py 第 7 步：仅 use_ddp=True 才调；本次单卡不会进来
def sync_bn_conversion(model: torch.nn.Module):
    """
    Convert BatchNorm to SyncBatchNorm (used for DDP)
    :param model: PyTorch model
    :return:
    model: PyTorch model with SyncBatchNorm layers
    """
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


# train_net.py 第 3 步：准备存检查点的目录
def check_snapshot(args):
    """
    Create directory to save training checkpoints, otherwise load the existing checkpoint.
    Additionally, if it is an array training job, create a new directory for each training job.
    :param args: Arguments from the argument parser
    :return:
    """

    # Check if it is an array training job (i.e. training with multiple random seeds on the same settings)
    # 多种子批量作业(本次 False)：给每个种子单独建一个子目录
    if args.array_training_job and not args.resume_training:
        args.snapshot_dir = os.path.join(args.snapshot_dir, str(args.seed))
        if not os.path.exists(args.snapshot_dir):
            save_dir = Path(args.snapshot_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Create directory to save training checkpoints, otherwise load the existing checkpoint
        # 普通情形【本次走这条】：目录不存在就建出来
        if not os.path.exists(args.snapshot_dir):
            # 注：这里用 or 判断 .pt/.pth，逻辑有点绕——对普通目录路径(本次 ./snapshot)恒成立、直接建目录；
            #     本意大概是“若路径像个检查点文件就报错(它本该已存在)”，但 or 写法对个别带 .pt 的路径会误判，本次不涉及
            if ".pt" not in args.snapshot_dir or ".pth" not in args.snapshot_dir:
                save_dir = Path(args.snapshot_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError('Snapshot checkpoint does not exist.')
