"""Presence loss variants to encourage prototype activation."""

# ======================================
# presence(存在性)损失：逼“每个部件至少在某处被强烈激活”，避免某些部件全程没用上(死部件)。
# 共 4 种写法，本次用 original。在 _run_batch 里调用时只喂前景通道 maps[:, :-1](不含背景)。
# 共同套路：先 avg_pool 平滑、再 adaptive_max_pool 取每张图每部件的“最强响应”，然后惩罚“最强响应还不够大”。
# ======================================

import torch


# soft_constraint 变体：用 detach 的激活当“软目标”，让梯度只调空间位置、不直接抬激活值本身(本次不用)
def presence_loss_soft_constraint(maps: torch.Tensor, beta: float = 0.1):
    """
    Calculate presence loss for a feature map
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :param beta: Weight of soft constraint
    :return: value of the presence loss
    """
    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0]
    loss_max_detach = loss_max.detach().clone()
    loss_max_p1 = 1 - loss_max
    loss_max_p2 = ((1 - beta) * loss_max_detach) + beta
    loss_max_final = (loss_max_p1 * loss_max_p2).mean()
    return loss_max_final


# tanh 变体(PIPNet 风格)：各部件最强响应跨 batch 求和 -> tanh -> 与全 1 做 BCE(本次不用)
def presence_loss_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation from the paper PIP-NET
    Ref: https://github.com/M-Nauta/PIPNet/blob/68054822ee405b5f292369ca846a9c6233f2df69/pipnet/train.py#L111
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = torch.nn.functional.binary_cross_entropy(pooled_maps, target=torch.ones_like(pooled_maps))

    return loss_max


# soft_tanh 变体：tanh 后直接用 1-tanh 当损失(去掉 log，更软)(本次不用)
def presence_loss_soft_tanh(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map with tanh formulation (non-log/softer version)
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return:
    """
    pooled_maps = torch.tanh(torch.sum(torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1), dim=0))

    loss_max = 1 - pooled_maps

    return loss_max.mean()


# original 变体【本次用】：
def presence_loss_original(maps: torch.Tensor):
    """
    Calculate presence loss for a feature map
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L181
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :return: value of the presence loss
    """
    # avg_pool2d(3,stride=1) 平滑 -> adaptive_max_pool2d(1) 取每图每部件空间最大值 -> [B, N']
    # .max(dim=0)[0] 跨 batch 取每部件的全局最强响应 -> [N'] -> .mean() 对部件平均 -> 标量
    loss_max = torch.nn.functional.adaptive_max_pool2d(torch.nn.functional.avg_pool2d(
        maps, 3, stride=1), 1).flatten(start_dim=1).max(dim=0)[0].mean()

    # 1 - 最强响应：响应越接近 1，损失越小(逼每个部件在 batch 里至少有一处被强烈激活)
    return 1 - loss_max


# 按 loss_type 分派到上面四个变体之一(本次 'original')
class PresenceLoss(torch.nn.Module):
    """
    This class defines the presence loss.
    """

    def __init__(self, loss_type: str = "original", beta: float = 0.1):
        super(PresenceLoss, self).__init__()
        self.loss_type = loss_type
        self.beta = beta

    def forward(self, maps):
        """
        Forward function for the presence loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The presence loss
        """
        if self.loss_type == "original":
            return presence_loss_original(maps)
        elif self.loss_type == "soft_constraint":
            return presence_loss_soft_constraint(maps, beta=self.beta)
        elif self.loss_type == "tanh":
            return presence_loss_tanh(maps)
        elif self.loss_type == "soft_tanh":
            return presence_loss_soft_tanh(maps)
        else:
            raise NotImplementedError(f"Presence loss {self.loss_type} not implemented")
