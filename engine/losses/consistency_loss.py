"""Consistency loss (KL-divergence) between cross-layer prototype maps."""

# ======================================
# consistency 损失：让“浅层(前几个注入位点)的部件图”向“最深层那张部件图”看齐。
# 做法是算 KL 散度(以深层图为目标、且 detach 不回传)，逼浅层分布逼近深层分布。
# 在 _run_batch 里对 m_buffer[:-1] 的每一层都算一次(target 固定取 m_buffer[-1])。
# ======================================

import torch


def consistency_loss(pred, target, eps=1e-10):
    """Compute KL-divergence-based consistency loss.

    Encourages the prototype assignment maps at shallower layers to be
    consistent with the map at the deepest layer.

    Args:
        pred: Predicted map of shape ``(B, N+1, L)``.
        target: Target (detached) map of shape ``(B, N+1, L)``.
        eps: Small constant for numerical stability.

    Returns:
        Scalar consistency loss.
    """
    # 加微小量，避免 log(0) / 除 0
    pred = pred + eps
    target = target + eps
    # KL(target || pred) = Σ target·log(target/pred)，逐元素算
    kl_loss = target * torch.log(target / pred)
    kl_loss = kl_loss.sum(dim=1)     # 对“部件”维求和 -> [B, L]
    kl_loss = kl_loss.mean(dim=1)    # 对空间维求平均 -> [B]
    return kl_loss.mean()            # 对 batch 求平均 -> 标量
