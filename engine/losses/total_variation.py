# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ======================================
# 全变差(Total Variation)损失：惩罚部件图相邻像素的剧烈跳变，逼图变得空间平滑、连成片、不破碎。
# 做法 = 相邻像素差的绝对值之和(水平 + 垂直)，即图梯度幅值的近似。在 _run_batch 里对每张部件图算并求平均。
# (本文件改编自 torchmetrics/kornia 的 TV 实现)
# ======================================

from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from typing_extensions import Literal


# 算一个 batch 的 TV 统计量：相邻像素差绝对值之和(竖直方向 diff1 + 水平方向 diff2)
def _total_variation_update(img: Tensor) -> Tuple[Tensor, int]:
    """Compute total variation statistics on current batch."""
    if img.ndim != 4:
        raise RuntimeError(f"Expected input `img` to be an 4D tensor, but got {img.shape}")
    diff1 = img[..., 1:, :] - img[..., :-1, :]   # 上下相邻像素差 (B,C,H-1,W)
    diff2 = img[..., :, 1:] - img[..., :, :-1]   # 左右相邻像素差 (B,C,H,W-1)

    res1 = diff1.abs().sum([1, 2, 3])            # 每张图竖直方向的总变差 (B,)
    res2 = diff2.abs().sum([1, 2, 3])            # 每张图水平方向的总变差 (B,)
    score = res1 + res2                          # 合计 (B,)
    return score


# 把每张图的 TV 分数按 reduction 汇总(mean=除以元素数，得每像素平均变差)
def _total_variation_compute(
        score: Tensor, num_elements: Union[int, Tensor], reduction: Optional[Literal["mean", "sum", "none"]]
) -> Tensor:
    """Compute final total variation score."""
    if reduction == "mean":
        return score.sum() / num_elements
    if reduction == "sum":
        return score.sum()
    if reduction is None or reduction == "none":
        return score
    raise ValueError("Expected argument `reduction` to either be 'sum', 'mean', 'none' or None")


# 对外的函数式接口：算更新 + 汇总
def total_variation(img: Tensor, reduction: Optional[Literal["mean", "sum", "none"]] = "sum", num_elements: int = 0) -> Tensor:
    """Compute total variation loss.

    Args:
        img: A `Tensor` of shape `(N, C, H, W)` consisting of images
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample
        num_elements: The number of elements in the input tensor

    Returns:
        A loss scalar value containing the total variation

    Raises:
        ValueError:
            If ``reduction`` is not one of ``'sum'``, ``'mean'``, ``'none'`` or ``None``
        RuntimeError:
            If ``img`` is not 4D tensor

    """
    # code adapted from:
    # from kornia.losses import total_variation as kornia_total_variation
    score = _total_variation_update(img)
    return _total_variation_compute(score, num_elements, reduction)


# 训练器实际用的是这个类(reduction='mean')：第一次 forward 时记下归一化元素数 = B*H*W
class TotalVariationLoss(torch.nn.Module):
    """
    Compute total variation loss.
    args:
        reduction: a method to reduce metric score over samples.

            - ``'mean'``: takes the mean over samples
            - ``'sum'``: takes the sum over samples
            - ``None`` or ``'none'``: return the score per sample
    """

    def __init__(self, reduction: Optional[Literal["mean", "sum", "none"]] = "mean") -> None:
        super(TotalVariationLoss, self).__init__()
        self.reduction = reduction
        self.num_elements = None

    def forward(self, img: Tensor) -> Tensor:
        # 注：归一化用 B*H*W(不含通道数 C)，即“每像素位置的平均变差”
        if self.num_elements is None:
            self.num_elements = img.shape[0] * img.shape[2] * img.shape[3]
        return total_variation(img=img, reduction=self.reduction, num_elements=self.num_elements)
