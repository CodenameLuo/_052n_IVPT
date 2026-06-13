"""Orthogonality loss to encourage diversity among part prototypes."""

# ======================================
# 正交损失：让不同部件的特征向量彼此“正交/不相似”，逼各部件学到不同的东西(别都盯同一块)。
# 做法 = 把各部件特征单位化后两两算余弦相似度(Gram 矩阵)，扣掉对角线(自己和自己=1)，
# 再对剩下的“非对角元素”取平方均值——越接近 0 越正交。在 _run_batch 里对 all_features 算一次。
# ======================================

import torch


def orthogonality_loss(all_features):
    """
    Calculate orthogonality loss for a feature map
    Ref: https://github.com/robertdvdk/part_detection/blob/eec53f2f40602113f74c6c1f60a2034823b0fcaf/train.py#L44
    :param all_features: The feature map with shape (batch_size, feature_dim, num_landmarks + 1)
    :return:
    """
    # 沿特征维单位化 -> 之后的内积即“余弦相似度”
    normed_feature = torch.nn.functional.normalize(all_features, dim=1)
    total_landmarks = all_features.shape[-1]   # N+1
    # 两两余弦相似度矩阵(Gram)：[B, N+1, N+1]
    similarity_fg = torch.matmul(normed_feature.permute(0, 2, 1).contiguous(), normed_feature)
    # 减掉单位阵 -> 把对角线(自相似=1)清零，只留“不同部件之间”的相似度
    similarity_fg = torch.sub(similarity_fg, torch.eye(total_landmarks, device=all_features.device))
    # 非对角元素的平方均值：越小越正交(各部件越互不相似)
    orth_loss = torch.mean(torch.square(similarity_fg))
    return orth_loss
