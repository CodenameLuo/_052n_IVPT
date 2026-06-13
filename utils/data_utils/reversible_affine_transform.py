"""Reversible affine transformations for equivariance loss computation."""

# ======================================
#
# 等变损失(equivariance)的工具：先随机生成一组仿射变换参数，再用它去变换图像/特征图，并能“反着变回去”。
# 等变的思路：把输入图做个仿射变换 T、过模型得到部件图；理想情况下，这等于“先对原图的部件图做 T”。
# 所以训练时：对变换后图得到的部件图再用 T⁻¹ 变回去，应当和原图的部件图对齐——对不齐就罚(见 equivarance_loss.py)。
#
# 两个函数：generate_affine_trans_params(在范围内随机采一组参数)、rigid_transform(施加变换，invert=True 则施加逆变换)。
#
# ======================================

from typing import List, Optional, Tuple, Any

import torch
import torchvision.transforms as transforms


# 在给定范围内随机采一组仿射参数(角度/平移/缩放/错切)；每个 batch 采一次
def generate_affine_trans_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
) -> Tuple[float, Tuple[int, int], float, Any]:
    """Get parameters for affine transformation

    Returns:
        params to be passed to the affine transformation
    """
    # 旋转角度：在 [degrees[0], degrees[1]] 内均匀采(本次 [-90, 90])
    angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
    # 平移：把“比例”乘以图边长得到最大像素位移，再在 ±max 内采整数像素(本次比例 0.11)
    if translate is not None:
        max_dx = float(translate[0] * img_size[0])
        max_dy = float(translate[1] * img_size[1])
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = (tx, ty)
    else:
        translations = (0, 0)

    # 缩放：在 [下界, 上界] 内均匀采(本次 [0.8, 1.4])
    if scale_ranges is not None:
        scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
    else:
        scale = 1.0

    # 错切：本次 shears=None -> 不采，shear=0
    shear_x = shear_y = 0.0
    if shears is not None:
        shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
        if len(shears) == 4:
            shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

    shear = (shear_x, shear_y)
    if shear_x == 0.0 and shear_y == 0.0:
        shear = 0.0

    return angle, translations, scale, shear


# 对图像/特征图施加仿射变换；invert=True 时施加“逆变换”(把已经变换过的图变回去)
def rigid_transform(img, angle, translate, scale, invert=False, shear=0,
                    interpolation=transforms.InterpolationMode.BILINEAR):
    """
    Affine transforms input image
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/lib.py#L54
    Parameters
    ----------
    img: Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation
    shear: float
        Shear angle in degrees
    interpolation: InterpolationMode
        Interpolation mode to calculate output values
    Returns
    ----------
    img: Tensor
        Transformed image

    """
    # 正向：一次性施加 旋转 angle + 平移 translate + 缩放 scale + 错切 shear
    if not invert:
        img = transforms.functional.affine(img, angle=angle, translate=translate, scale=scale, shear=shear,
                                           interpolation=interpolation)
    # 逆向：把正向操作按相反顺序、相反量逐步抵消 —— 先反向平移(-t)，再反向旋转(-angle)且缩放用 1/scale
    else:
        translate = [-t for t in translate]
        img = transforms.functional.affine(img=img, angle=0, translate=translate, scale=1, shear=shear)
        img = transforms.functional.affine(img=img, angle=-angle, translate=[0, 0], scale=1 / scale, shear=shear)

    return img
