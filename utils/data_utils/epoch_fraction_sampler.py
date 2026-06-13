"""Distributed sampler that uses only a fraction of data per epoch.

Splits the full dataset into ``ceil(1/fraction)`` non-overlapping shards.
Each epoch uses one shard (cycling through shards across epochs), so that
after enough epochs every sample is seen exactly once per cycle.

This guarantees:
    1. Each epoch only iterates ``fraction`` of the data (fewer batches).
    2. All samples are uniformly covered across consecutive epochs.
"""

# ======================================
#
# 采样器(Sampler)决定“DataLoader 每个 epoch 按什么顺序、取哪些下标”。
# 这里实现“每个 epoch 只用一部分数据”：把全集切成 ceil(1/fraction) 份(shard)，
# 每个 epoch 只用其中一份、并随 epoch 轮换，凑满 ceil(1/fraction) 个 epoch 就正好把全集过一遍。
#
# 本次用途：eval_fraction=0.1 -> 训练中途的“评估 loader”用 EpochFractionSampler(单卡版)，
#           每次中途评估只过 1/10 的测试集，省时间(最终评估仍用全量)。
#           epoch_fraction=1.0，所以训练 loader 不用这个采样器。
#
# 两个类：EpochFractionDistributedSampler(多卡版，多一步“按卡再切分”)；EpochFractionSampler(单卡版，本次用)。
#
# ======================================

import math
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# 多卡版：在“按 epoch 取一份 shard”之外，还要把这份 shard 再均分给各张卡。本次单卡不走这里
class EpochFractionDistributedSampler(Sampler):
    """DistributedSampler that only yields a fraction of the dataset per epoch.

    Parameters
    ----------
    dataset : Dataset
        The full training dataset.
    fraction : float
        Fraction of the dataset to use per epoch, e.g. 0.1 = 10%.
    num_replicas : int, optional
        Number of distributed processes (default: world_size).
    rank : int, optional
        Rank of the current process (default: current rank).
    shuffle : bool
        Whether to shuffle indices (default: True).
    seed : int
        Random seed for reproducibility (default: 0).
    drop_last : bool
        Whether to drop the last incomplete batch (default: False).
    """

    def __init__(self, dataset: Dataset, fraction: float = 0.1,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 shuffle: bool = True, seed: int = 0,
                 drop_last: bool = False) -> None:
        # 没给卡数/rank 就从分布式环境里取(非分布式则卡数=1、rank=0)
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        # fraction 必须在 (0,1] 内
        assert 0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"

        self.dataset = dataset
        self.fraction = fraction
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 要切成几份才能覆盖全集(如 fraction=0.1 -> 10 份)
        # Total number of shards needed to cover the full dataset
        self.num_shards = math.ceil(1.0 / fraction)
        self.total_dataset_size = len(dataset)

        # 每份(shard)多少个样本
        # Samples per shard (before distributing across replicas)
        self.shard_size = math.ceil(self.total_dataset_size / self.num_shards)

        # 每张卡每个 epoch 拿多少(把一份 shard 再均分给各卡)
        # Samples per replica per epoch
        if self.drop_last:
            self.num_samples = math.floor(self.shard_size / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.shard_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # 用 seed+epoch 做确定性打乱(同一 epoch 跨进程/跨运行得到一致顺序)
        # Deterministic shuffling based on epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(self.total_dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.total_dataset_size))

        # 选本 epoch 用哪一份 shard(随 epoch 轮换：epoch%份数)
        # Select the shard for this epoch (cycling)
        shard_id = self.epoch % self.num_shards
        start = shard_id * self.shard_size
        end = min(start + self.shard_size, self.total_dataset_size)
        indices = indices[start:end]

        # 补齐到能被卡数整除(最后不够时用开头几个补上)
        # Pad to make evenly divisible by num_replicas
        if len(indices) < self.total_size:
            padding = self.total_size - len(indices)
            indices += indices[:padding]

        # 按 rank 间隔取样，得到“分给本卡”的那部分
        # Subsample for this replica
        indices = indices[self.rank:self.total_size:self.num_replicas]

        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    # 训练/评估循环每个 epoch 调一次，更新 self.epoch -> 从而切到下一份 shard
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# 单卡版【本次走这个】：逻辑同上，但没有“按卡再切分/补齐”这一步
class EpochFractionSampler(Sampler):
    """Single-GPU version of EpochFractionDistributedSampler."""

    def __init__(self, dataset: Dataset, fraction: float = 0.1,
                 shuffle: bool = True, seed: int = 0) -> None:
        assert 0 < fraction <= 1.0, f"fraction must be in (0, 1], got {fraction}"

        self.dataset = dataset
        self.fraction = fraction
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # 份数 / 全集大小 / 每份大小(fraction=0.1 -> 10 份，每份约 1/10)
        self.num_shards = math.ceil(1.0 / fraction)
        self.total_dataset_size = len(dataset)
        self.shard_size = math.ceil(self.total_dataset_size / self.num_shards)

    def __iter__(self):
        # 同样用 seed+epoch 确定性打乱
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = torch.randperm(self.total_dataset_size, generator=g).tolist()
        else:
            indices = list(range(self.total_dataset_size))

        # 取本 epoch 对应的那一份 shard(随 epoch 轮换)；这一份就是本次中途评估实际过的样本
        shard_id = self.epoch % self.num_shards
        start = shard_id * self.shard_size
        end = min(start + self.shard_size, self.total_dataset_size)
        indices = indices[start:end]

        return iter(indices)

    def __len__(self) -> int:
        return self.shard_size

    # 更新 epoch -> 切到下一份 shard(由训练/评估循环调用)
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
