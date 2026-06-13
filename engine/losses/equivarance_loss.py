"""Equivariance loss for part prototype attention maps.

Encourages the detected part locations to be equivariant under spatial
affine transformations (rotation, scale, shear, translation).
"""

# ======================================
# 等变损失：希望模型“认部件”这件事对几何变换稳定——把图做仿射变换 T 再过模型，得到的部件图
# 应当约等于“对原图部件图做同样的 T”。这里的检验方式：把变换后图的部件图用 T⁻¹ 变回去(rot_back)，
# 再和原图的部件图比余弦相似度；越像越好，所以损失 = 1 - 平均余弦相似度。
# (equiv_maps 由 _run_batch 把“变换后的图”再过一遍模型得到，见 P7b)
# ======================================

import torch
from utils.data_utils.reversible_affine_transform import rigid_transform


def equivariance_loss(maps, equiv_maps, source, num_landmarks, translate, angle, scale, shear=0.0):
    """
    This function calculates the equivariance loss
    Modified from: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L67
    :param maps: Attention map with shape (batch_size, channels, height, width) where channels is the landmark probability
    :param equiv_maps: Attention maps for same images after an affine transformation and then passed through the model
    :param source: Original mini-batch of images
    :param num_landmarks: Number of landmarks/parts
    :param translate: Translation parameters for the affine transformation
    :param angle: Angle parameter for the affine transformation
    :param scale: Scale parameter for the affine transformation
    :param shear: Shear parameter for the affine transformation
    :return:
    """

    # 平移量是按原图像素(518)定的，但部件图只有 37×37，所以要按分辨率比例缩放平移量
    translate = [(t * maps.shape[-1] / source.shape[-1]) for t in translate]
    # 把“变换后图的部件图”用逆变换 T⁻¹ 变回去，理论上应与原图部件图对齐
    rot_back = rigid_transform(img=equiv_maps, angle=angle, translate=translate,
                               scale=scale, shear=shear, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    num_landmarks = maps.shape[1]-1 # TODO  只比前景部件(去掉背景通道)
    # 原图部件图(前景)摊平成向量 [B, N, L]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :],
                                       (-1, num_landmarks,
                                        num_elements_per_map))
    # 变回去的部件图(前景)摊平成向量 [B, N, L]
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :],
                                         (-1, num_landmarks,
                                          num_elements_per_map))
    # 逐部件算两者的余弦相似度
    cos_sim_equiv = torch.nn.functional.cosine_similarity(orig_attmap_vector,
                                                          transf_attmap_vector, -1)
    # 损失 = 1 - 平均余弦相似度(越对齐损失越小)
    loss_equiv = (1 - torch.mean(cos_sim_equiv))

    return loss_equiv
