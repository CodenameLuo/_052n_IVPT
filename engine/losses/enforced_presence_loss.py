"""Enforced presence loss to encourage background part coverage."""

# ======================================
# enforced presence(强制存在)损失：逼“背景”那个槽去覆盖图像边缘区域，从而把前景部件“挤”到物体主体上。
# 本次 loss_type='enforced_presence'：用一张“离中心越远权重越大”的径向 mask 加权后，
#   取背景通道的最强响应、与全 1 做 BCE —— 即鼓励背景在图像四周保持高响应。
# 另外几种(log/linear/mse)更简单：直接惩罚背景通道的平均响应偏低，没有空间径向加权。
# ======================================

import torch


class EnforcedPresenceLoss(torch.nn.Module):
    """
    This class defines the Enforced Presence loss.
    """

    def __init__(self, loss_type: str = "log", eps: float = 1e-10):
        super(EnforcedPresenceLoss, self).__init__()
        self.loss_type = loss_type
        self.eps = eps
        # 径向 mask 与坐标网格只在第一次 forward 时按图尺寸算一次，之后复用
        self.grid_x = None
        self.grid_y = None
        self.mask = None

    def forward(self, maps):
        """
        Forward function for the Enforced Presence loss.
        :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
        :return: The Enforced Presence loss
        """
        # 【本次】enforced_presence：径向 mask 加权 -> 取背景通道最强响应 -> 与 1 做 BCE
        if self.loss_type == "enforced_presence":
            avg_pooled_maps = torch.nn.functional.avg_pool2d(
                maps, 3, stride=1)                       # 先平滑
            # 懒计算径向 mask：坐标归一化到 [-1,1]，mask = (x²+y²) 再归一化 -> 中心 0、四周接近 1
            if self.grid_x is None or self.grid_y is None:
                grid_x, grid_y = torch.meshgrid(torch.arange(avg_pooled_maps.shape[2]),
                                                torch.arange(avg_pooled_maps.shape[3]), indexing='ij')
                grid_x = grid_x.unsqueeze(0).unsqueeze(0).contiguous().to(avg_pooled_maps.device,
                                                                          non_blocking=True)
                grid_y = grid_y.unsqueeze(0).unsqueeze(0).contiguous().to(avg_pooled_maps.device,
                                                                          non_blocking=True)
                grid_x = (grid_x / grid_x.max()) * 2 - 1
                grid_y = (grid_y / grid_y.max()) * 2 - 1

                mask = grid_x ** 2 + grid_y ** 2
                mask = mask / mask.max()
                self.grid_x = grid_x
                self.grid_y = grid_y
                self.mask = mask

            masked_part_activation = avg_pooled_maps * self.mask          # 给四周加权、压低中心
            masked_bg_part_activation = masked_part_activation[:, -1, :, :]  # 取背景通道(最后一个)

            # 背景通道空间最大值 -> 与全 1 做 BCE：逼背景在(加权后的)四周有强响应
            max_pooled_maps = torch.nn.functional.adaptive_max_pool2d(masked_bg_part_activation, 1).flatten(start_dim=0)
            loss_area = torch.nn.functional.binary_cross_entropy(max_pooled_maps, torch.ones_like(max_pooled_maps))
        # 其余变体：不做空间加权，直接惩罚“背景通道平均响应偏低”(本次不走)
        else:
            part_activation_sums = torch.nn.functional.adaptive_avg_pool2d(maps, 1).flatten(start_dim=1)
            background_part_activation = part_activation_sums[:, -1]
            if self.loss_type == "log":
                loss_area = torch.nn.functional.binary_cross_entropy(background_part_activation,
                                                                     torch.ones_like(background_part_activation))

            elif self.loss_type == "linear":
                loss_area = (1 - background_part_activation).mean()

            elif self.loss_type == "mse":
                loss_area = torch.nn.functional.mse_loss(background_part_activation,
                                                         torch.ones_like(background_part_activation))
            else:
                raise ValueError(f"Invalid loss type: {self.loss_type}")

        return loss_area
