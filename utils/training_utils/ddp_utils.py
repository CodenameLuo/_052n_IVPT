"""Distributed Data Parallel (DDP) setup and seed utilities."""

# ======================================
#
# 这个文件是“多卡训练(DDP)的基础设施”：检测几张卡、初始化进程组、设种子、跨卡汇总张量等。
# 本次是单卡(NUM_GPUS=1)：
#   - multi_gpu_check() 返回 False(train_net.py 第 7 步据此跳过 SyncBN)；
#   - ddp_setup() 不会被调用(只有多卡时训练器才调)；
#   - reduce_tensor / concat_all_gather 这些跨卡操作也用不上；
#   - set_seeds() 仍会被调用(rank=0)，calculate_effective_batch_size() 会被 weight decay 计算用到(world_size=1)。
#
# ======================================

import os
import random
import socket
import subprocess

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group


# 初始化 PyTorch 分布式进程组(只有多卡训练才调；本次单卡不进来)
def ddp_setup():
    # 是否在 SLURM 集群作业里(看有没有 SLURM 环境变量)
    is_slurm_job = "SLURM_NODEID" in os.environ
    if is_slurm_job:
        # —— SLURM 情形：从 SLURM 环境变量推断节点数/rank/卡数，并设好分布式要用的环境变量 ——
        # Define the process group based on SLURM env variables
        # number of nodes / node ID
        n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])   # 本机内第几张卡
        global_rank = int(os.environ['SLURM_PROCID'])   # 全局第几个进程

        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])    # 总进程数(=总卡数)
        n_gpu_per_node = world_size // n_nodes

        # define master address and master port
        # 主节点地址(从 SLURM 节点列表里取第一个)，供各进程互联
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')

        # set environment variables for 'env://'
        # 把这些写进环境变量，供下面 init_process_group(init_method='env://') 读取
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)

        # define whether this is the master process / if we are in distributed mode
        is_master = node_id == 0 and local_rank == 0
        multi_node = n_nodes > 1
        multi_gpu = world_size > 1

        # summary
        # 只在每台机的 0 号卡打印一份配置摘要(避免每个进程都刷屏)
        prefix = "%i - " % global_rank
        if local_rank == 0:
            print(prefix + "Number of nodes: %i" % n_nodes)
            print(prefix + "Node ID        : %i" % node_id)
            print(prefix + "Local rank     : %i" % local_rank)
            print(prefix + "Global rank    : %i" % global_rank)
            print(prefix + "World size     : %i" % world_size)
            print(prefix + "GPUs per node  : %i" % n_gpu_per_node)
            print(prefix + "Master         : %s" % str(is_master))
            print(prefix + "Multi-node     : %s" % str(multi_node))
            print(prefix + "Multi-GPU      : %s" % str(multi_gpu))
            print(prefix + "Hostname       : %s" % socket.gethostname())
    else:
        # —— 非 SLURM(如本机用 torchrun)：local_rank 由 torchrun 通过环境变量 LOCAL_RANK 给出 ——
        local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print("Initializing PyTorch distributed ...")
    # 用 NCCL 后端初始化进程组(env:// 表示从上面那些环境变量读配置)
    init_process_group(init_method='env://', backend="nccl")
    print(local_rank)
    # 把当前进程绑定到对应的 GPU
    torch.cuda.set_device(local_rank)
    return


# 固定随机种子(保证可复现)；训练器开跑前调用
def set_seeds(seed_value: int = 42):
    # # Set the manual seeds
    # （以下是早期版本的设种子写法，已注释保留）
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # np.random.seed(seed_value)

    # random.seed(seed_value)
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # # Ensure deterministic algorithms
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Get the rank of the current process (GPU index in multi-GPU distributed mode)
    # 多卡时给每张卡的种子加上 rank 偏移，让各卡的随机序列不同但仍可复现(单卡 rank=0，无偏移)
    rank = dist.get_rank() if dist.is_initialized() else 0
    seed = seed_value + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Set the same seed for all GPUs
    # Ensure deterministic algorithms
    # 强制确定性算法、关掉 cuDNN autotune(注：这会盖过 train_net.py 里的 benchmark=True，以确定性优先)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 固定 Python 哈希种子(注：这里写死 "42"，没用上 seed_value 参数)
    os.environ["PYTHONHASHSEED"] = "42"


# 跨卡求和再除以卡数 = 跨卡求平均(多卡汇总指标时用；本次单卡用不到)
def reduce_tensor(tensor: torch.Tensor, world_size: int):
    """Reduce tensor across all nodes."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# 把张量(标量)转成 Python float
def to_python_float(t: torch.Tensor):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


# 判断分布式是否可用且已初始化
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


# 取当前进程的 rank(非分布式时恒为 0)；常用于“只在 0 号卡打印/存档”
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# 把各卡上的张量收集起来沿 batch 维拼接(注意：该操作不带梯度)；本次单卡用不到
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# train_net.py 第 7 步调用：可见 GPU 多于 1 张就走 DDP。本次单卡 -> 返回 False
def multi_gpu_check():
    """
    Check if there are multiple GPUs available for DDP
    :return:
    use_ddp: bool, whether to use DDP or not
    """
    if torch.cuda.device_count() > 1:
        use_ddp = True
    else:
        use_ddp = False
    return use_ddp


# 算“有效(全局)batch size” = 每卡 batch × 卡数；给 weight decay 归一化用(本次单卡 = batch_size×1)
def calculate_effective_batch_size(args):
    """
    Calculate the effective batch size for DDP
    :param args: Arguments from the argument parser
    :return:
    effective_batch_size: int, effective batch size
    """
    batch_size = args.batch_size
    use_ddp = multi_gpu_check()
    is_slurm_job = "SLURM_NODEID" in os.environ
    if is_slurm_job:
        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
    else:
        # 非 SLURM：多卡从 WORLD_SIZE 环境变量取卡数，单卡则 world_size=1
        if use_ddp:
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = 1

    effective_batch_size = batch_size * world_size
    return effective_batch_size
