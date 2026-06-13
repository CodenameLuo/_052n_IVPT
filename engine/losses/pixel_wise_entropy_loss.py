"""Pixel-wise entropy loss to sharpen per-pixel prototype assignments."""

# ======================================
# 像素级熵损失：把“每个像素属于各部件”的分布往“只属于一个部件”推(让分配更硬)。
# 做法 = 对每个像素算它在部件维上的分类熵，再取所有像素的平均；熵越低=越接近 one-hot。
# 配合 gumbel-softmax 用(gumbel 出来的值可能略超出 [0,1]，所以先 clamp + 重归一化保数值稳定)。
# ======================================

import torch


def pixel_wise_entropy_loss(maps):
    """
    Calculate pixel-wise entropy loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the pixel-wise entropy loss
    """
    # Clamp to valid probability range for numerical stability (esp. under AMP / gumbel_softmax)
    # 夹到 [1e-6, 1] 保证是合法概率(gumbel/AMP 下可能略越界)
    maps = maps.float().clamp(min=1e-6, max=1.0)
    maps = maps / maps.sum(dim=1, keepdim=True)  # re-normalise after clamp  夹完重归一化，使各部件概率和为 1
    # Calculate entropy for each pixel
    # 把部件维换到最后一维(Categorical 要求类别在最后一维)，逐像素算分类熵 -> [B, H, W]
    entropy = torch.distributions.categorical.Categorical(probs=maps.permute(0, 2, 3, 1).contiguous()).entropy()
    # Take the mean of the entropy
    # 所有像素的平均熵(标量)
    return entropy.mean()
