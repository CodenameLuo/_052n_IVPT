"""
Distributed trainer for IVPT.

Implements ``PDiscoTrainer`` which handles multi-GPU (DDP) training,
evaluation, checkpointing, and attention-map / hierarchical prototype
visualization.
"""

# ======================================
#
# 这个文件是“真正干活的训练器”。train_net.py 最后一步交给它后，这里负责把数据/模型/优化器/损失串成
# 完整的训练循环：搭 DataLoader、跑 epoch、每个 batch 前向+8损失+反向、定期评估、存检查点、可视化部件图。
#
# 本 Part(P6)只注释“构造阶段”：launch_ivpt_trainer(入口) + PDiscoTrainer.__init__ + 各 _init_* +
# _prepare_dataloader*(搭 3 个 DataLoader) + _load_snapshot。
# 真正的循环 train / _run_epoch / _run_batch(含 8 个损失)、定期评估、可视化留到 Part 7 / Part 8。
#
# ======================================

import copy
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fsspec
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchmetrics
from timm.data import Mixup
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 下面两个类平衡采样器是死代码(数据集没实现对应方法，开了会崩)，本次也没开
from utils.data_utils.class_balanced_distributed_sampler import ClassBalancedDistributedSampler
from utils.data_utils.class_balanced_sampler import ClassBalancedRandomSampler
# 按比例取数据的采样器：本次 eval_fraction=0.1，会用到它来搭“中途评估只用 10% 测试集”的 loader
from utils.data_utils.epoch_fraction_sampler import EpochFractionDistributedSampler, EpochFractionSampler
# 生成随机仿射变换参数(等变损失要用)
from utils.data_utils.reversible_affine_transform import generate_affine_trans_params
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.training_utils.engine_utils import AverageMeter, load_state_dict_ivpt
from utils.training_utils.snapshot_class import Snapshot
from utils.visualize_att_maps import VisualizeAttentionMaps
from utils.wandb_params import init_wandb

# 导入 engine/losses 下的所有损失(consistency/presence/equivariance/total_variation/... 的函数与类)
from .losses import *

# 是否是“主进程”(非分布式恒为 True；多卡时只有 rank0 为 True)——用于“只让一个进程打印/存档/上传日志”
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# 张量打印设置：关科学计数法、保留 3 位小数(让调试时打印的张量更易读)
torch.set_printoptions(sci_mode=False, precision=3)

class PDiscoTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            batch_size: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            loss_fn: List[torch.nn.Module],
            save_every: int,
            snapshot_path: str,
            loggers: List,
            log_freq: int = 10,
            use_amp: bool = False,
            grad_norm_clip: float = 1.0,
            max_epochs: int = 100,
            num_workers: int = 4,
            mixup_fn: Optional[Mixup] = None,
            eval_only: bool = False,
            loss_hyperparams: Optional[Dict] = None,
            eq_affine_transform_params: Optional[Dict] = None,
            use_ddp: bool = True,
            sub_path_test: str = "",
            dataset_name: str = "",
            amap_saving_prob: float = 0.05,
            class_balanced_sampling: bool = False,
            num_samples_per_class: int = 100,
            n_pro: str = "",
            enable_hierarchy_vis: bool = False,
            epoch_fraction: float = 1.0,
            eval_every_n_epochs: int = 1,
            eval_fraction: float = 1.0,
    ) -> None:
        # 解析原型数列表(可视化时要按各位点的原型数排版)
        self.n_pro = [int(n) for n in n_pro.split(',')]
        self.enable_hierarchy_vis = enable_hierarchy_vis
        self.epoch_fraction = epoch_fraction          # 每轮用多少训练数据(本次 1.0)
        self.eval_every_n_epochs = eval_every_n_epochs # 每多少 epoch 评估一次(本次 5)
        self.eval_fraction = eval_fraction             # 中途评估用多少测试集(本次 0.1)
        # 确定 rank / world_size(本次单卡 -> local_rank=0, world_size=1)
        self._init_ddp(use_ddp)
        self.num_landmarks = model.num_landmarks       # 部件数(本次 4)
        self.num_classes = model.num_classes           # 类别数(本次 200)
        # Top-k accuracy metrics for evaluation
        # 建好 top1/top5、micro/macro 等准确率指标(torchmetrics)
        self._init_accuracy_metrics()
        # 把模型搬到对应 GPU
        self.model = model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.name_dataset = dataset_name
        self.sub_path_test = sub_path_test
        self.batch_size = batch_size
        self.eval_only = eval_only
        # Number of samples per class for class balanced sampling
        self.num_samples_per_class = num_samples_per_class
        # === 搭 3 个 DataLoader ===
        # ① 训练 loader：is_train=True；本次 epoch_fraction=1.0 -> 不用分片采样器 -> 单卡时直接 shuffle 全量
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers,
                                                     class_balanced_sampling=class_balanced_sampling,
                                                     is_train=True)
        # ② 中途评估 loader：本次 eval_fraction=0.1<1.0 -> is_eval_fraction=True -> 只取 10% 测试集(加速中途评估)
        # Partial eval loader (used during training for periodic eval)
        if self.eval_fraction < 1.0 and test_dataset is not None:
            self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers,
                                                        drop_last=False, is_eval_fraction=True)
        else:
            self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
        # ③ 最终评估 loader：训练全部结束后用，跑全量测试集
        # Full eval loader (used for final evaluation after training completes)
        self.test_loader_full = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
        # 分类损失：loss_fn 是 [训练损失, 评估损失]；只有一个就训练/评估共用
        if len(loss_fn) == 1:
            self.loss_fn_train = self.loss_fn_eval = loss_fn[0]
        else:
            self.loss_fn_train = loss_fn[0]
            self.loss_fn_eval = loss_fn[1]

        self.save_every = save_every                   # 每多少 epoch 存档(本次 10)
        self.amap_saving_prob = amap_saving_prob       # 存部件图的概率
        self.epochs_run = 0                            # 已跑完的 epoch(续训时会被读档覆盖)
        self.snapshot_path = snapshot_path
        # 判断 snapshot_path 是目录还是单个文件
        if os.path.isdir(snapshot_path):
            self.is_snapshot_dir = True
        else:
            self.is_snapshot_dir = False
        # 开了 wandb 才初始化 run(只在主进程)；本次 loggers=[]，跳过
        if loggers:
            if self.local_rank == 0 and self.global_rank == 0:
                loggers[0] = init_wandb(loggers[0])
        self.loggers = loggers
        self.log_freq = log_freq
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.mixup_fn = mixup_fn
        self.epoch_test_accuracies = []
        self.current_epoch = 0
        self.accum_steps = 1                           # 梯度累积步数(=1，即不累积)

        # Equivariance affine transform parameters
        # 把等变损失要用的仿射范围(degrees/translate/scale/shear)拆成实例属性
        self._init_affine_transform_params(eq_affine_transform_params)

        # Loss hyperparameters
        # 把各损失权重拆成实例属性，并实例化 presence/总变差/enforced presence 等损失模块
        self._init_losses(loss_hyperparams)

        # Loss dictionary
        # 给每一项损失建一个 AverageMeter，按 epoch 累计平均
        self._init_loss_dict()

        # 开了混合精度才建 GradScaler(本次未开)
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # === 续训：若 snapshot 目录里已有 best 检查点 / 或 snapshot_path 本身是个检查点文件，就读档接着练 ===
        # 本次是全新训练、./snapshot 为空，所以这两条都不触发，从头开始
        if os.path.isfile(os.path.join(snapshot_path, f"snapshot_best.pt")):
            print("Loading snapshot")
            self._load_snapshot()
        elif os.path.isfile(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()
            self.snapshot_path = os.path.dirname(snapshot_path)
            self.is_snapshot_dir = True
        self.batch_img_metas = None
        # Initialize the visualization class
        # 初始化“部件注意力图可视化器”(把模型关注的部件画到图上保存，供肉眼检查；P8 细看)
        self.vis_att_maps = VisualizeAttentionMaps(snapshot_dir=self.snapshot_path, sub_path_test=self.sub_path_test,
                                                   dataset_name=self.name_dataset, bg_label=self.num_landmarks,
                                                   batch_size=self.batch_size, num_parts=self.num_landmarks + 1, n_pro=self.n_pro)
        # 多卡才用 DDP 把模型包起来(各卡同步梯度)；本次单卡 -> 打印 "Using single GPU"
        if self.use_ddp:
            if self.local_rank == 0 and self.global_rank == 0:
                print(f"Using {self.world_size} GPUs, Broadcast Buffers")
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=True)
        else:
            print("Using single GPU")

        self.epoch_test_accuracies = []
        # 开了 wandb 才让 logger 跟踪模型参数/梯度(本次 loggers 为空，跳过)
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(model, log="all", log_freq=self.log_freq)

    # 确定分布式相关的 rank / world_size。三种情形：SLURM 集群 / 单卡 / torchrun 多卡
    def _init_ddp(self, use_ddp) -> None:
        self.is_slurm_job = "SLURM_NODEID" in os.environ
        self.use_ddp = use_ddp
        if self.is_slurm_job:
            # SLURM：从环境变量读节点/进程信息，并强制走 DDP
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.global_rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_world_size = self.world_size // n_nodes
            self.use_ddp = True
        else:
            # 单卡【本次走这条】：rank 全 0、world_size=1
            if not self.use_ddp:
                self.local_rank = 0
                self.global_rank = 0
                self.world_size = 1
                self.local_world_size = 1
            # torchrun 多卡：从 torchrun 注入的环境变量读 rank/world_size
            else:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # 把各损失权重从字典拆成实例属性；并实例化几个需要建模块的损失(放到 GPU 上)
    def _init_losses(self, loss_hyperparams: dict) -> None:
        # Loss hyperparameters
        self.l_classification = loss_hyperparams['l_class_att']          # 分类损失权重
        self.l_presence = loss_hyperparams['l_presence']                # presence 损失权重
        self.l_presence_beta = loss_hyperparams['l_presence_beta']      # presence 损失的 beta
        self.l_presence_type = loss_hyperparams['l_presence_type']      # presence 损失形式
        self.l_orth = loss_hyperparams['l_orth']                        # 正交损失权重
        self.l_equiv = loss_hyperparams['l_equiv']                      # 等变损失权重
        self.l_tv = loss_hyperparams['l_tv']                            # 全变差损失权重
        self.l_enforced_presence = loss_hyperparams['l_enforced_presence']                    # enforced presence 权重
        self.l_enforced_presence_loss_type = loss_hyperparams['l_enforced_presence_loss_type']  # 其形式
        self.l_pixel_wise_entropy = loss_hyperparams['l_pixel_wise_entropy']                  # 像素熵权重
        # 注：l_conc(concentration)没在这里取 -> 印证了它“装载了但训练器从不用”
        # 实例化需要状态的损失模块(presence / 全变差 / enforced presence)，搬到 GPU
        self.enforced_presence_loss = EnforcedPresenceLoss(loss_type=self.l_enforced_presence_loss_type).to(
            self.local_rank,
            non_blocking=True)
        self.total_variation_loss = TotalVariationLoss(reduction="mean").to(self.local_rank, non_blocking=True)
        self.presence_loss = PresenceLoss(beta=self.l_presence_beta,
                                          loss_type=self.l_presence_type).to(self.local_rank, non_blocking=True)
        # 分类损失(交叉熵)也搬到 GPU
        self.loss_fn_eval = self.loss_fn_eval.to(self.local_rank, non_blocking=True)
        self.loss_fn_train = self.loss_fn_train.to(self.local_rank, non_blocking=True)

    # 把等变损失要用的仿射变换范围拆成实例属性(degrees=[-90,90]/translate/scale/shear)
    def _init_affine_transform_params(self, eq_affine_transform_params: dict) -> None:
        # Equivariance affine transform parameters
        self.eq_degrees = eq_affine_transform_params['degrees']
        self.eq_translate = eq_affine_transform_params['translate']
        self.eq_scale_ranges = eq_affine_transform_params['scale_ranges']
        self.eq_shear = eq_affine_transform_params['shear']

    # 给每一项要记录的损失各建一个 AverageMeter(训练 9 项 + 验证 1 项)，按 epoch 累计平均
    def _init_loss_dict(self) -> None:
        self.loss_dict_train = {'loss_consistency': AverageMeter(),
                                'loss_classification_train': AverageMeter(),
                                'loss_presence_train': AverageMeter(),
                                'loss_orth_train': AverageMeter(),
                                'loss_equiv_train': AverageMeter(),
                                'loss_total_train': AverageMeter(),
                                'loss_tv': AverageMeter(),
                                'loss_enforced_presence': AverageMeter(),
                                'loss_pixel_wise_entropy': AverageMeter()}

        self.loss_dict_val = {'loss_total_val': AverageMeter()}

    # 建准确率指标(用 torchmetrics)：训练集和测试集各一套，每套含 top1/top5 × micro/macro 四个
    # micro=按样本平均(常说的整体准确率)；macro=按类平均(每类等权，类别不均衡时更能反映少数类)
    def _init_accuracy_metrics(self) -> None:
        self.acc_dict_train = {'train_acc': torchmetrics.classification.MulticlassAccuracy(
                                num_classes=self.num_classes, top_k=1,
                                average="micro").to(self.local_rank,
                                                    non_blocking=True),
                               'train_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="micro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top1_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=1,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True),
                               'macro_avg_acc_top5_train': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="macro").to(self.local_rank,
                                                       non_blocking=True)}

        self.acc_dict_test = {'test_acc': torchmetrics.classification.MulticlassAccuracy(
                                num_classes=self.num_classes, top_k=1,
                                average="micro").to(self.local_rank,
                                                    non_blocking=True),
                              'test_acc_top5': torchmetrics.classification.MulticlassAccuracy(
                                   num_classes=self.num_classes, top_k=5,
                                   average="micro").to(self.local_rank,
                                                       non_blocking=True),
                              'macro_avg_acc_top1_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=1,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True),
                              'macro_avg_acc_top5_test': torchmetrics.classification.MulticlassAccuracy(
                                  num_classes=self.num_classes, top_k=5,
                                  average="macro").to(self.local_rank,
                                                      non_blocking=True)}

    # 多卡版搭 DataLoader：按情形选采样器(类平衡/按比例分片/标准 DistributedSampler)。本次单卡不走这里
    def _prepare_dataloader_ddp(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                                class_balanced_sampling: bool = False, is_train: bool = False,
                                is_eval_fraction: bool = False, drop_last: bool = True):
        if class_balanced_sampling:
            sampler = ClassBalancedDistributedSampler(dataset, num_samples_per_class=self.num_samples_per_class)
        elif is_train and self.epoch_fraction < 1.0:
            sampler = EpochFractionDistributedSampler(dataset, fraction=self.epoch_fraction)
        elif is_eval_fraction and self.eval_fraction < 1.0:
            sampler = EpochFractionDistributedSampler(dataset, fraction=self.eval_fraction)
        else:
            sampler = DistributedSampler(dataset)
        # 用了 sampler 时 shuffle 必须为 False(打乱交给 sampler 管)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
        )

    # 搭 DataLoader 的总入口【本次单卡走这里】：多卡转给上面的 ddp 版；单卡按情形选采样器
    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                            class_balanced_sampling: bool = False, drop_last: bool = True,
                            is_train: bool = False, is_eval_fraction: bool = False):
        if self.use_ddp:
            return self._prepare_dataloader_ddp(dataset, num_workers, class_balanced_sampling,
                                                is_train=is_train, is_eval_fraction=is_eval_fraction,
                                                drop_last=drop_last)

        # 类平衡采样(本次 False，且会崩，不走)
        if class_balanced_sampling:
            sampler = ClassBalancedRandomSampler(dataset, num_samples_per_class=self.num_samples_per_class)
        # 训练且 epoch_fraction<1：按比例分片(本次 1.0，不走) -> 训练 loader 最终落到下面 sampler=None
        elif is_train and self.epoch_fraction < 1.0:
            sampler = EpochFractionSampler(dataset, fraction=self.epoch_fraction)
        # 中途评估且 eval_fraction<1【本次=0.1，走这条】：只取 10% 测试集
        elif is_eval_fraction and self.eval_fraction < 1.0:
            sampler = EpochFractionSampler(dataset, fraction=self.eval_fraction)
        else:
            sampler = None

        # 没有自定义 sampler 时(如训练 loader、全量评估 loader)就让 DataLoader 自己 shuffle
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=(sampler is None),
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
        )

    # 读档(续训/评估时用)：把检查点里的模型权重、优化器状态、已跑 epoch、历史准确率都恢复回来
    # 本次全新训练、./snapshot 里没有 best 检查点，所以构造时这个函数不会被调到
    def _load_snapshot(self) -> None:
        loc = f"cuda:{self.local_rank}"
        # 打开检查点文件(目录就取里面的 snapshot_best.pt，否则当作文件路径)；找不到就从头训练
        try:
            if self.is_snapshot_dir:
                snapshot = fsspec.open(os.path.join(self.snapshot_path, f"snapshot_best.pt"))
            else:
                snapshot = fsspec.open(self.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location=loc)
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        # 还原模型权重
        snapshot, state_dict = load_state_dict_ivpt(snapshot_data)
        self.model.load_state_dict(state_dict)
        # 纯评估模式：只要权重就够了，不用恢复优化器/进度
        if self.eval_only:
            return
        # 续训：恢复优化器状态、已跑到的 epoch，并把调度器快进到该 epoch；历史准确率也接上
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.scheduler.step(snapshot.finished_epoch)
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    # ============================================================================
    # ↓↓↓ 以下 _run_batch / _run_epoch / train 是训练的“循环主体”，留到 Part 7 ↓↓↓
    #   _run_batch ：跑一个 batch——前向、(等变)再前向、算 8 个损失求和、反向+梯度裁剪+更新；评估时则只算指标/可视化。
    #   _run_epoch ：把一个 epoch 的所有 batch 过一遍，累计 loss/准确率。
    #   train      ：外层 epoch 循环 + 定期评估 + 存档(_save_snapshot) + 训练后最终全量评估。
    #   test_only  ：纯评估模式(eval_only=True 才走)，本次不走。
    # 这些在 Part 7 / Part 8 里结合损失体逐行细注，这里(P6 构造阶段)先不展开、保持原样。
    # ============================================================================
    def _run_batch(self, source, targets, train: bool = True, vis_att_maps: bool = False, curr_iter: int = 0, vis_flag = None) -> \
            Tuple[Any, Any]:
        # train=True 时开启梯度；use_amp 时用 fp16 自动混合精度(本次未开 amp)
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=self.use_amp):
            # 前向(详见 individual_landmark_vit.forward / Part 7a)：拿到 5 个输出
            #   all_features 部件特征[B,D,N+1]；dis_sim_maps=maps_list[-3]；scores 各部件类别分[B,200,N]；
            #   maps_loss=maps_list(5 张部件图)；m_buffer(5 张部件空间图)；qm_buffer(各层软分配)
            all_features, dis_sim_maps, scores, maps_loss, (m_buffer, qm_buffer) = self.model(source) # (B, N+1, L) (B, P+1, N)

            # 把 N 个部件各自的类别分求平均 -> 最终每张图的类别 logits [B, 200]
            outputs = scores.mean(dim=-1)  # (batch_size, num_classes)

            if train:
                # consistency 损失的目标：最深层那张部件空间图(detach，不让梯度回流到它)
                target = m_buffer[-1].detach()

                # loss_consistency =

                # === 等变损失要的“第二次前向”：把图做随机仿射变换后再过一遍模型 ===
                # Forward pass of transformed images
                # 采一组随机仿射参数(本次范围 旋转±90/平移0.11/缩放[0.8,1.4]/无错切)
                angle, translate, scale, shear = generate_affine_trans_params(
                    degrees=self.eq_degrees, translate=self.eq_translate, scale_ranges=self.eq_scale_ranges,
                    shears=self.eq_shear, img_size=[source.shape[2], source.shape[3]])
                # Apply the affine transform to the source image
                # 对原图施加该变换
                source_transformed = rigid_transform(img=source, angle=angle,
                                                     translate=translate,
                                                     scale=scale,
                                                     shear=0.0, invert=False)
                # 变换后的图再过一遍模型，取它的部件图列表(第 4 个输出)，供等变损失比对
                equiv_maps = self.model(source_transformed)[3] # TODO

                # === 8 个损失(都用 train_net.py 传来的权重加权；很多项是“对 5 张部件图各算一次再平均”) ===
                # ① 分类损失：最终 logits vs 标签(交叉熵)
                # Classification loss
                loss_classification = self.loss_fn_train(outputs, targets) * self.l_classification

                # ② 一致性损失：前 4 层部件图都向最深层 target 看齐(KL)，求平均
                # Consistency loss
                loss_consistency = sum([consistency_loss(maps_train, target) for maps_train in m_buffer[:-1]]) / len(m_buffer[:-1])

                # ③ 全变差损失：逼每张部件图空间平滑
                # Total variation loss
                loss_tv = sum([self.total_variation_loss(maps) * self.l_tv for maps in maps_loss]) / len(maps_loss)

                # ④ presence 损失：只对前景通道 maps[:, :-1]，逼每个部件至少一处强激活
                # Presence loss (fg) for landmarks
                loss_presence = sum([self.presence_loss(maps=maps[:, :-1, :, :]) * self.l_presence for maps in maps_loss]) / len(maps_loss)

                # ⑤ 正交损失：逼各部件特征互不相似
                # Orthogonality loss
                loss_orth = orthogonality_loss(all_features) * self.l_orth

                # ⑥ 等变损失：把变换图的部件图 T⁻¹ 变回、与原图部件图比余弦相似度
                # Equivariance loss: calculate rotated landmarks distance
                loss_equiv = sum([equivariance_loss(maps_loss[i], equiv_maps[i], source, self.num_landmarks, translate, angle, scale,
                                               shear=0.0) * self.l_equiv for i in range(len(maps_loss))]) / len(maps_loss)

                # ⑦ 强制存在损失：逼背景占据图像边缘
                # Enforced presence loss
                loss_enforced_presence = sum([self.enforced_presence_loss(maps) * self.l_enforced_presence for maps in maps_loss]) / len(maps_loss)

                # ⑧ 像素级熵损失：逼每个像素的部件分配更“硬”
                # Pixel-wise entropy loss
                loss_pixel_wise_entropy = sum([pixel_wise_entropy_loss(maps) * self.l_pixel_wise_entropy for maps in maps_loss]) / len(maps_loss)

                # 8 项相加 = 总损失
                loss = loss_consistency + loss_presence + loss_classification + loss_orth + loss_equiv + loss_tv + loss_enforced_presence + loss_pixel_wise_entropy

                # === 反向 + 更新 ===
                self.optimizer.zero_grad(set_to_none=True)   # 清梯度(set_to_none 更省显存)
                # amp 分支(本次不走)：缩放 loss 再反向，配 GradScaler
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                # 【本次走这条】普通反向：backward -> 梯度裁剪(范数≤2.0) -> 更新参数
                else:
                    loss.backward()
                    if self.grad_norm_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    self.optimizer.step()

                # 把各项损失转成 Python 数(.item())记入字典，供打印/日志/累计平均
                losses_dict = {'loss_consistency': loss_consistency.item(),
                               'loss_classification_train': loss_classification.item(),
                               'loss_presence_train': loss_presence.item(),
                               'loss_orth_train': loss_orth.item(),
                               'loss_equiv_train': loss_equiv.item(),
                               'loss_total_train': loss.item(), 'loss_tv': loss_tv.item(),
                               'loss_enforced_presence': loss_enforced_presence.item(),
                               'loss_pixel_wise_entropy': loss_pixel_wise_entropy.item()}
            # —— 评估分支(train=False)：只算评估损失 + 整理可视化用的图；show_maps 等可视化细节见 Part 8 ——
            else:
                # 评估只算分类损失(不算那 8 项正则)
                loss = self.loss_fn_eval(outputs, targets)
                losses_dict = {'loss_total_val': loss.item()}
                # 整理各层“原型->部件”的软分配矩阵，给层级可视化统计关系用：
                maps_multi = []
                for lay, maps in enumerate(qm_buffer):
                    # 把等于 0.1 的位置清零(对应空原型被填的均匀值)；注：本次 N=4 时填充值是 1/4=0.25，与 0.1 不符，故这步实际不命中
                    maps[maps==0.1] = 0
                    maps = maps.mean(0)                      # 对 batch 求平均 -> [P, N]
                    maps = maps / maps.sum(-1, keepdim=True) # 沿部件维归一化
                    maps[torch.isnan(maps)] = 0              # 处理除零产生的 NaN
                    maps_multi.append(maps)

                # 需要可视化的 epoch：每 10 个 iter、且在主进程上，把部件图画出来存盘(细节见 visualize_att_maps.py)
                if vis_att_maps:
                    if vis_flag == None:
                        # 普通可视化：对每一层部件图各画一张叠加图
                        if curr_iter % 10 == 0 and is_main_process():
                            for lay, maps in enumerate(maps_loss):
                                self.vis_att_maps.show_maps(ims=source, maps=maps, epoch=self.current_epoch, # TODO
                                                            curr_iter=curr_iter, extra_info=lay)
                    else:
                        # 层级可视化(eval-only + enable_hierarchy_vis 才会传 vis_flag；本次不走)
                        if curr_iter % 10 == 0 and is_main_process():
                            self.vis_att_maps.show_maps_hie(ims=source, maps_loss=maps_loss, epoch=self.current_epoch, # TODO
                                                        curr_iter=curr_iter, extra_info=lay, vis_flag=vis_flag)

        # 训练返回(预测, 损失字典)；评估(普通模式)多返回 maps_multi；层级模式(vis_flag 非空)只为画图、无返回
        if train:
            return outputs, losses_dict
        else:
            if vis_flag == None:
                return outputs, losses_dict, maps_multi
        

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        """
        Runs one epoch of training or evaluation
        :param epoch: Current epoch
        :param dataloader: Dataloader to use
        :param train: If we are training or evaluating
        :return:
        loss: Average loss across all batches
        top1: Average top1 accuracy across all batches
        top5: Average top5 accuracy across all batches
        losses_dict: Dictionary of all losses
        """

        # 多卡时通知采样器当前 epoch(它据此换 shard / 重洗)
        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)

        # 本 epoch 的更新次数与起始计数(供 scheduler.step_update 按 iteration 推进；本次 accum_steps=1)
        updates_per_epoch = (len(dataloader) + self.accum_steps - 1) // self.accum_steps
        num_updates = (epoch - 1) * updates_per_epoch

        # 这个 epoch 要不要存部件可视化图：第 1 个 epoch、每 save_every 个、以及最后一个 epoch 都存
        vis_att_maps = True if epoch % self.save_every == 0 else False
        vis_att_maps = True if epoch == self.max_epochs else vis_att_maps
        vis_att_maps = True if epoch == 1 else vis_att_maps
        losses_dict = {}
        # 每个 epoch 开始：把所有 loss / 准确率累计器清零
        for key in self.loss_dict_train.keys():
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val.keys():
            self.loss_dict_val[key].reset()

        accuracies_dict = {}

        for key in self.acc_dict_train.keys():
            self.acc_dict_train[key].reset()
        for key in self.acc_dict_test.keys():
            self.acc_dict_test[key].reset()

        # === 遍历该 dataloader 的所有 batch ===
        maps_buffer = []
        for it, mini_batch in enumerate(dataloader):
            source = mini_batch[0]              # 图 [B,3,518,518]
            targets = mini_batch[1]             # 标签 [B]
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            # mixup 本次为 None，不触发
            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)

            # 跑一个 batch：训练分支返回(预测, 损失字典)；评估分支多返回可视化用的 maps_multi
            if train:
                batch_preds, losses_dict = self._run_batch(source, targets, train,
                                                        vis_att_maps=vis_att_maps, curr_iter=it)
            else:
                batch_preds, losses_dict, maps_multi = self._run_batch(source, targets, train,
                                                        vis_att_maps=vis_att_maps, curr_iter=it)
                maps_buffer.append(maps_multi)

            # 把这个 batch 的损失/准确率累计进对应 meter
            if train:
                for key in losses_dict.keys():
                    self.loss_dict_train[key].update(losses_dict[key], source.size(0))
                if self.mixup_fn is None:
                    for key in self.acc_dict_train.keys():
                        self.acc_dict_train[key].update(batch_preds, targets)
                # 按 iteration 推进调度器(StepLR 主要看 epoch，这步对它基本无影响)
                num_updates += 1
                self.scheduler.step_update(num_updates=num_updates)
            else:
                for key in losses_dict.keys():
                    self.loss_dict_val[key].update(losses_dict[key], source.size(0))
                for key in self.acc_dict_test.keys():
                    self.acc_dict_test[key].update(batch_preds, targets)
            # 每 log_freq 个 iter 打印一次进度
            if it % self.log_freq == 0:
                if train:
                    print(
                        f'[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} Total {losses_dict["loss_total_train"]:.3f}'
                        f'| Classification {losses_dict["loss_classification_train"]:.3f} | Consistency {losses_dict["loss_consistency"]:.3f}'
                        f'| Presence {losses_dict["loss_presence_train"]:.3f} | Orth {losses_dict["loss_orth_train"]:.3f}'
                        f'| Equiv {losses_dict["loss_equiv_train"]:.3f} | TV {losses_dict["loss_tv"]:.3f}'
                        f'| Enforced Presence {losses_dict["loss_enforced_presence"]:.3f}'
                        f'| Pixel-wise Entropy {losses_dict["loss_pixel_wise_entropy"]:.3f}')
                else:
                    print(
                        f'[GPU{self.global_rank}] Epoch {epoch} | Iter {it} | {step_type} '
                        f'Total Loss {losses_dict["loss_total_val"]:.5f}')

        # ============================================================================
        # ↓↓↓ 以下“层级原型可视化”整块【不在本次训练路径上】↓↓↓
        # 仅当 eval-only 且 enable_hierarchy_vis=True(本次都 False)才触发：跨层统计原型归属关系、
        # 把各层保存的部件裁图重排成层级目录、再跑一遍数据集生成多层叠加图。属于可视化，细节留到 Part 8。
        # ============================================================================
        # ---- Hierarchical prototype visualization ----
        # Activated when enable_hierarchy_vis=True during eval-only.
        # Computes cross-layer prototype relationships, reorganises saved
        # per-layer crops into a hierarchical folder layout, then re-runs
        # the dataloader to produce per-image multi-layer overlay figures.
        if not train and epoch == 0 and is_main_process() and self.enable_hierarchy_vis:
            map_dir = Path(os.path.join(self.snapshot_path, 'results_hie_' + self.sub_path_test))
            map_dir.mkdir(parents=True, exist_ok=True)
            maps_buffer = list(zip(*maps_buffer))

            relation_proto = []
            for i, mb in enumerate(maps_buffer):
                mb = torch.stack(mb, dim=0).mean(0)
                max_values, max_indices = torch.max(mb, dim=1)
                positions_column_first = [
                    (max_indices[ii].item(), ii, max_values[ii].item())
                    for ii in range(len(max_indices))
                    if max_values[ii].item() > 0.5
                ]
                relation_proto.append(positions_column_first)

                src_dir = map_dir / "middle" / str(i)
                tgt_dir = map_dir / "final"
                for pos in positions_column_first:
                    src_d = src_dir / str(pos[1])
                    tgt_d = tgt_dir / f"Proto_{pos[0]}" / f"Layer_{i + 12 - len(maps_buffer)}"
                    tgt_d.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_d), str(tgt_d / (src_d.name + "_" + str(round(pos[2], 2)))))

            for idx in range(mb.shape[-1]):
                src_dir = map_dir / "middle" / str(len(maps_buffer))
                tgt_dir = map_dir / "final"
                src_d = src_dir / str(idx)
                tgt_d = tgt_dir / f"Proto_{idx}" / "Layer_12"
                tgt_d.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_d), str(tgt_d / (src_d.name + "_1.00")))

            shutil.rmtree(map_dir / "middle", ignore_errors=True)

            # Re-run the dataloader to generate hierarchical overlay images
            for it, mini_batch in enumerate(dataloader):
                source = mini_batch[0]
                targets = mini_batch[1]
                source = source.to(self.local_rank, non_blocking=True)
                targets = targets.to(self.local_rank, non_blocking=True)
                self._run_batch(source, targets, train,
                                vis_att_maps=vis_att_maps, curr_iter=it,
                                vis_flag=relation_proto)

        # === epoch 结束：把各 meter 的累计平均收进字典返回；训练时还要按 epoch 推进 StepLR ===
        if train:
            # 取各项损失的 epoch 平均
            for key in self.loss_dict_train.keys():
                losses_dict[key] = self.loss_dict_train[key].avg
            # 取训练准确率(torchmetrics 全 epoch 聚合后 ×100 成百分比)
            if self.mixup_fn is None:
                for key in self.acc_dict_train.keys():
                    accuracies_dict[key] = self.acc_dict_train[key].compute().item() * 100
            # 按 epoch 推进学习率(StepLR：到步长就 ×gamma)
            self.scheduler.step(epoch)
        else:
            # 评估：取验证损失与测试准确率
            for key in self.loss_dict_val.keys():
                losses_dict[key] = self.loss_dict_val[key].avg
            for key in self.acc_dict_test.keys():
                accuracies_dict[key] = self.acc_dict_test[key].compute().item() * 100
        return losses_dict, accuracies_dict

    # 存检查点：把(模型权重 + 优化器状态 + 当前 epoch + 历史准确率)打包成 Snapshot 存盘
    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.model
        # 多卡时模型被 DDP 包了一层，真正的模型在 .module 里；这里取出未包装的原始模型存权重
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            epoch_test_accuracies=self.epoch_test_accuracies,
        )
        # save snapshot
        snapshot = asdict(snapshot)        # dataclass 转 dict 再存
        if self.is_snapshot_dir:
            save_path_base = self.snapshot_path
        else:
            save_path_base = os.path.dirname(self.snapshot_path)
        # 文件名：最后一轮存 snapshot_final.pt；最佳存 snapshot_best.pt；其余存 snapshot_<epoch>.pt
        if epoch == self.max_epochs:
            save_path = os.path.join(save_path_base, f"snapshot_final.pt")
        elif save_best:
            save_path = os.path.join(save_path_base, f"snapshot_best.pt")
        else:
            save_path = os.path.join(save_path_base, f"snapshot_{epoch}.pt")

        torch.save(snapshot, save_path)
        print(f"Snapshot saved at epoch {epoch}")

    def finish_logging(self):
        for logger in self.loggers:
            logger.finish()

    # 外层训练循环：逐 epoch 训练 -> 记录 LR/指标 -> 定期存档 -> 定期评估(存最佳) -> 全部跑完后做最终全量评估
    def train(self):
        # 从 epochs_run(续训时>0，本次=0)训到 max_epochs
        for epoch in range(self.epochs_run, self.max_epochs):
            epoch += 1                       # epoch 从 1 开始计
            self.current_epoch = epoch
            self.model.train()               # 切训练模式(开 dropout 等)
            # 跑一个训练 epoch
            loss_dict_train, acc_dict_train = self._run_epoch(epoch, self.train_loader, train=True)

            # 记录各参数组当前学习率(base / scratch / modulation / finer)，方便观察分组 LR 的衰减
            logging_dict = {'epoch': epoch,
                            'base_lr': self.optimizer.param_groups[0]['lr'],
                            'scratch_lr': self.optimizer.param_groups[-1]['lr'],
                            'modulation_lr': self.optimizer.param_groups[-2]['lr'],
                            'finer_lr': self.optimizer.param_groups[-3]['lr']}
            if self.local_rank == 0 and self.global_rank == 0:
                logging_dict.update(loss_dict_train)
                logging_dict.update(acc_dict_train)
            # 每 save_every 个 epoch 存一次普通检查点；最后一个 epoch 存 final
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)

            # === 定期评估：每 eval_every_n_epochs 个 epoch 且还没到最后一轮时，用部分测试集评一次 ===
            # Periodic eval (partial test set, every eval_every_n_epochs)
            should_eval = (self.test_loader and
                           epoch % self.eval_every_n_epochs == 0 and
                           epoch < self.max_epochs)
            if should_eval:
                self.model.eval()                       # 切评估模式
                with torch.inference_mode():            # 关梯度、更省显存
                    loss_dict_val, acc_dict_test = self._run_epoch(epoch, self.test_loader, train=False)
                if self.local_rank == 0 and self.global_rank == 0:
                    # 记录本次测试准确率；若刷新了历史最高，就把当前模型存成 best
                    test_acc = acc_dict_test['test_acc']
                    self.epoch_test_accuracies.append(test_acc)
                    max_acc = max(self.epoch_test_accuracies)
                    max_acc_index = self.epoch_test_accuracies.index(max_acc)
                    if max_acc_index == len(self.epoch_test_accuracies) - 1:
                        self._save_snapshot(epoch, save_best=True)

                    logging_dict.update(loss_dict_val)
                    logging_dict.update(acc_dict_test)
                    if self.eval_fraction < 1.0:
                        print(f"[Epoch {epoch}] Periodic eval ({self.eval_fraction*100:.0f}% test data) "
                              f"| Acc: {test_acc:.2f}%")
                    for logger in self.loggers:
                        logger.log(logging_dict)
            # 不评估的 epoch：只记训练指标
            elif self.local_rank == 0 and self.global_rank == 0:
                # Log training-only metrics when we skip eval
                for logger in self.loggers:
                    logger.log(logging_dict)

        # ---- Final evaluation on full test set ----
        # === 所有 epoch 跑完后：在“全量”测试集上做一次最终评估(中途评估只用了 10%) ===
        if self.test_loader_full:
            if self.local_rank == 0 and self.global_rank == 0:
                print("\n" + "=" * 60)
                print("Final evaluation on FULL test set")
                print("=" * 60)
            self.model.eval()
            with torch.inference_mode():
                # 用全量测试集(test_loader_full)跑一遍评估
                loss_dict_val, acc_dict_test = self._run_epoch(self.max_epochs, self.test_loader_full, train=False)
            if self.local_rank == 0 and self.global_rank == 0:
                # 同样：若最终全量准确率刷新历史最高，存成 best
                test_acc = acc_dict_test['test_acc']
                self.epoch_test_accuracies.append(test_acc)
                max_acc = max(self.epoch_test_accuracies)
                max_acc_index = self.epoch_test_accuracies.index(max_acc)
                if max_acc_index == len(self.epoch_test_accuracies) - 1:
                    self._save_snapshot(self.max_epochs, save_best=True)
                print(f"[Final] Full test set | Acc: {test_acc:.2f}% "
                      f"| Top5: {acc_dict_test['test_acc_top5']:.2f}% "
                      f"| Best: {max_acc:.2f}%")
                final_log = {'epoch': self.max_epochs}
                final_log.update(loss_dict_val)
                final_log.update(acc_dict_test)
                for logger in self.loggers:
                    logger.log(final_log)

        if self.local_rank == 0 and self.global_rank == 0:
            self.finish_logging()

    # 纯评估模式(eval_only=True 才被 launch_ivpt_trainer 调用；本次走 train()，不走这里)：
    # 切 eval -> 在测试集上跑一个 epoch -> 打印 top1/top5、micro/macro 准确率 -> 记日志
    def test_only(self):
        self.model.eval()
        logging_dict = {}
        with torch.inference_mode():
            if self.test_loader:
                loss_dict_val, acc_dict_test = self._run_epoch(0, self.test_loader, train=False)
            print(
                f'Test loss: {loss_dict_val["loss_total_val"]:.5f} '
                f'| Test acc: {acc_dict_test["test_acc"]:.5f} '
                f'| Test acc top5: {acc_dict_test["test_acc_top5"]:.5f} '
                f'| Macro avg acc top1: {acc_dict_test["macro_avg_acc_top1_test"]:.5f} '
                f'| Macro avg acc top5: {acc_dict_test["macro_avg_acc_top5_test"]:.5f}')

        if self.local_rank == 0 and self.global_rank == 0:
            logging_dict.update(loss_dict_val)
            logging_dict.update({'epoch': 0})
            logging_dict.update(acc_dict_test)
            for logger in self.loggers:
                logger.log(logging_dict)
        self.finish_logging()


# ======================================
# train_net.py 最后一步调用的入口：设种子 -> (多卡才)初始化分布式 -> 造 PDiscoTrainer -> 训练或评估 -> 收尾
# ======================================
def launch_ivpt_trainer(model: torch.nn.Module,
                          train_dataset: torch.utils.data.Dataset,
                          test_dataset: torch.utils.data.Dataset,
                          batch_size: int,
                          optimizer: torch.optim.Optimizer,
                          scheduler: torch.optim.lr_scheduler.LRScheduler,
                          loss_fn: List[torch.nn.Module],
                          epochs: int,
                          save_every: int,
                          loggers: List,
                          log_freq: int,
                          use_amp: bool = False,
                          snapshot_path: str = "snapshot.pt",
                          grad_norm_clip: float = 1.0,
                          num_workers: int = 0,
                          mixup_fn: Optional[Mixup] = None,
                          seed: int = 42,
                          eval_only: bool = False,
                          loss_hyperparams: Optional[Dict] = None,
                          eq_affine_transform_params: Optional[Dict] = None,
                          use_ddp: bool = False,
                          sub_path_test: str = "",
                          dataset_name: str = "",
                          amap_saving_prob: float = 0.05,
                          class_balanced_sampling: bool = False,
                          num_samples_per_class: int = 100,
                          n_pro: str = "",
                          enable_hierarchy_vis: bool = False,
                          epoch_fraction: float = 1.0,
                          eval_every_n_epochs: int = 1,
                          eval_fraction: float = 1.0,
                          ) -> None:
    """Train and evaluate an IVPT model.

    Instantiates a :class:`PDiscoTrainer`, then either trains for the
    requested number of epochs (with periodic evaluation) or runs
    evaluation only.

    Args:
        model: The IVPT model to train / evaluate.
        train_dataset: Training dataset.
        test_dataset: Test / validation dataset.
        batch_size: Per-GPU batch size.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        loss_fn: List of loss functions ``[train_loss, eval_loss]``.
        epochs: Total number of training epochs.
        save_every: Save a checkpoint every *N* epochs.
        loggers: List of logger objects (e.g. W&B run).
        log_freq: Logging frequency (iterations).
        use_amp: Enable mixed-precision training.
        snapshot_path: Directory (or file) to save/load checkpoints.
        grad_norm_clip: Max gradient norm for clipping.
        num_workers: DataLoader workers.
        mixup_fn: Optional Mixup / CutMix transform.
        seed: Random seed.
        eval_only: If ``True``, skip training and run evaluation only.
        loss_hyperparams: Dict of loss weights and settings.
        eq_affine_transform_params: Affine-transform params for equivariance.
        use_ddp: Use DistributedDataParallel.
        sub_path_test: Sub-path of the test image directory.
        dataset_name: Name of the dataset (e.g. ``"cub"``).
        amap_saving_prob: Probability of saving attention-map images.
        class_balanced_sampling: Use class-balanced sampling.
        num_samples_per_class: Samples per class when using balanced sampling.
        n_pro: Comma-separated prototype counts per layer (e.g. ``"17,14,11,8,5"``).
        enable_hierarchy_vis: Enable hierarchical prototype visualisation
            during eval-only runs.
        eval_every_n_epochs: Run evaluation every N epochs during training.
        eval_fraction: Fraction of test data for periodic eval (full test set
            is always used for the final evaluation after training).
    """

    # 固定随机种子(注：这里也会把 cudnn 设成确定性模式，盖过 train_net.py 的 benchmark=True)
    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    # 多卡才初始化分布式进程组(本次单卡不调)
    if use_ddp:
        ddp_setup()

    # 用 train_net.py 传来的所有零件实例化训练器(构造时就把 3 个 loader、损失、指标、可视化器等都备齐)
    model_trainer = PDiscoTrainer(model=model, train_dataset=train_dataset, test_dataset=test_dataset,
                                  batch_size=batch_size, optimizer=optimizer, scheduler=scheduler,
                                  loss_fn=loss_fn,
                                  save_every=save_every, snapshot_path=snapshot_path, loggers=loggers,
                                  log_freq=log_freq,
                                  use_amp=use_amp,
                                  grad_norm_clip=grad_norm_clip, max_epochs=epochs, num_workers=num_workers,
                                  mixup_fn=mixup_fn, eval_only=eval_only, loss_hyperparams=loss_hyperparams,
                                  eq_affine_transform_params=eq_affine_transform_params, use_ddp=use_ddp,
                                  sub_path_test=sub_path_test, dataset_name=dataset_name,
                                  amap_saving_prob=amap_saving_prob,
                                  class_balanced_sampling=class_balanced_sampling,
                                  num_samples_per_class=num_samples_per_class,
                                  n_pro=n_pro,
                                  enable_hierarchy_vis=enable_hierarchy_vis,
                                  epoch_fraction=epoch_fraction,
                                  eval_every_n_epochs=eval_every_n_epochs,
                                  eval_fraction=eval_fraction)
    # eval_only=True 走纯评估；本次=False，走完整训练循环 train()
    if eval_only:
        model_trainer.test_only()
    else:
        model_trainer.train()
    # 多卡训练结束后销毁进程组(本次单卡不调)
    if use_ddp:
        destroy_process_group()
