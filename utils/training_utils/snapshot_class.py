"""Snapshot dataclass for checkpoint serialisation."""

# ======================================
#
# 一个纯数据容器(dataclass)，规定“一个检查点(snapshot)里到底存哪几样东西”。
# 训练器存档时把这四样塞进 Snapshot 再 torch.save；读档时反过来按这四个字段还原。
#
# ======================================

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'   # 模型权重(state_dict)
    optimizer_state: Dict[str, Any]                 # 优化器状态(Adam 的动量等，用于无缝续训)
    finished_epoch: int                             # 已经跑完到第几个 epoch(续训从这之后接着跑)
    epoch_test_accuracies: List[float] = None       # 各次评估的测试准确率列表(没评估过就是 None)
