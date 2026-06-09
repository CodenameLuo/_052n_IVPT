"""
Distributed trainer for IVPT.

Implements ``PDiscoTrainer`` which handles multi-GPU (DDP) training,
evaluation, checkpointing, and attention-map / hierarchical prototype
visualization.
"""

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

from utils.data_utils.class_balanced_distributed_sampler import ClassBalancedDistributedSampler
from utils.data_utils.class_balanced_sampler import ClassBalancedRandomSampler
from utils.data_utils.epoch_fraction_sampler import EpochFractionDistributedSampler, EpochFractionSampler
from utils.data_utils.reversible_affine_transform import generate_affine_trans_params
from utils.training_utils.ddp_utils import ddp_setup, set_seeds
from utils.training_utils.engine_utils import AverageMeter, load_state_dict_ivpt
from utils.training_utils.snapshot_class import Snapshot
from utils.visualize_att_maps import VisualizeAttentionMaps
from utils.wandb_params import init_wandb

from .losses import *

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

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
        self.n_pro = [int(n) for n in n_pro.split(',')]
        self.enable_hierarchy_vis = enable_hierarchy_vis
        self.epoch_fraction = epoch_fraction
        self.eval_every_n_epochs = eval_every_n_epochs
        self.eval_fraction = eval_fraction
        self._init_ddp(use_ddp)
        self.num_landmarks = model.num_landmarks
        self.num_classes = model.num_classes
        # Top-k accuracy metrics for evaluation
        self._init_accuracy_metrics()
        self.model = model.to(self.local_rank)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.name_dataset = dataset_name
        self.sub_path_test = sub_path_test
        self.batch_size = batch_size
        self.eval_only = eval_only
        # Number of samples per class for class balanced sampling
        self.num_samples_per_class = num_samples_per_class
        self.train_loader = self._prepare_dataloader(train_dataset, num_workers=num_workers,
                                                     class_balanced_sampling=class_balanced_sampling,
                                                     is_train=True)
        # Partial eval loader (used during training for periodic eval)
        if self.eval_fraction < 1.0 and test_dataset is not None:
            self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers,
                                                        drop_last=False, is_eval_fraction=True)
        else:
            self.test_loader = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
        # Full eval loader (used for final evaluation after training completes)
        self.test_loader_full = self._prepare_dataloader(test_dataset, num_workers=num_workers, drop_last=False)
        if len(loss_fn) == 1:
            self.loss_fn_train = self.loss_fn_eval = loss_fn[0]
        else:
            self.loss_fn_train = loss_fn[0]
            self.loss_fn_eval = loss_fn[1]

        self.save_every = save_every
        self.amap_saving_prob = amap_saving_prob
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.isdir(snapshot_path):
            self.is_snapshot_dir = True
        else:
            self.is_snapshot_dir = False
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
        self.accum_steps = 1

        # Equivariance affine transform parameters
        self._init_affine_transform_params(eq_affine_transform_params)

        # Loss hyperparameters
        self._init_losses(loss_hyperparams)

        # Loss dictionary
        self._init_loss_dict()

        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

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
        self.vis_att_maps = VisualizeAttentionMaps(snapshot_dir=self.snapshot_path, sub_path_test=self.sub_path_test,
                                                   dataset_name=self.name_dataset, bg_label=self.num_landmarks,
                                                   batch_size=self.batch_size, num_parts=self.num_landmarks + 1, n_pro=self.n_pro)
        if self.use_ddp:
            if self.local_rank == 0 and self.global_rank == 0:
                print(f"Using {self.world_size} GPUs, Broadcast Buffers")
            self.model = DDP(self.model, device_ids=[self.local_rank], broadcast_buffers=True)
        else:
            print("Using single GPU")

        self.epoch_test_accuracies = []
        if self.local_rank == 0 and self.global_rank == 0:
            for logger in self.loggers:
                logger.watch(model, log="all", log_freq=self.log_freq)

    def _init_ddp(self, use_ddp) -> None:
        self.is_slurm_job = "SLURM_NODEID" in os.environ
        self.use_ddp = use_ddp
        if self.is_slurm_job:
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
            self.local_rank = int(os.environ['SLURM_LOCALID'])
            self.global_rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.local_world_size = self.world_size // n_nodes
            self.use_ddp = True
        else:
            if not self.use_ddp:
                self.local_rank = 0
                self.global_rank = 0
                self.world_size = 1
                self.local_world_size = 1
            else:
                self.local_rank = int(os.environ["LOCAL_RANK"])
                self.global_rank = int(os.environ["RANK"])
                self.world_size = int(os.environ["WORLD_SIZE"])
                self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    def _init_losses(self, loss_hyperparams: dict) -> None:
        # Loss hyperparameters
        self.l_classification = loss_hyperparams['l_class_att']
        self.l_presence = loss_hyperparams['l_presence']
        self.l_presence_beta = loss_hyperparams['l_presence_beta']
        self.l_presence_type = loss_hyperparams['l_presence_type']
        self.l_orth = loss_hyperparams['l_orth']
        self.l_equiv = loss_hyperparams['l_equiv']
        self.l_tv = loss_hyperparams['l_tv']
        self.l_enforced_presence = loss_hyperparams['l_enforced_presence']
        self.l_enforced_presence_loss_type = loss_hyperparams['l_enforced_presence_loss_type']
        self.l_pixel_wise_entropy = loss_hyperparams['l_pixel_wise_entropy']
        self.enforced_presence_loss = EnforcedPresenceLoss(loss_type=self.l_enforced_presence_loss_type).to(
            self.local_rank,
            non_blocking=True)
        self.total_variation_loss = TotalVariationLoss(reduction="mean").to(self.local_rank, non_blocking=True)
        self.presence_loss = PresenceLoss(beta=self.l_presence_beta,
                                          loss_type=self.l_presence_type).to(self.local_rank, non_blocking=True)
        self.loss_fn_eval = self.loss_fn_eval.to(self.local_rank, non_blocking=True)
        self.loss_fn_train = self.loss_fn_train.to(self.local_rank, non_blocking=True)

    def _init_affine_transform_params(self, eq_affine_transform_params: dict) -> None:
        # Equivariance affine transform parameters
        self.eq_degrees = eq_affine_transform_params['degrees']
        self.eq_translate = eq_affine_transform_params['translate']
        self.eq_scale_ranges = eq_affine_transform_params['scale_ranges']
        self.eq_shear = eq_affine_transform_params['shear']

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
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
        )

    def _prepare_dataloader(self, dataset: torch.utils.data.Dataset, num_workers: int = 4,
                            class_balanced_sampling: bool = False, drop_last: bool = True,
                            is_train: bool = False, is_eval_fraction: bool = False):
        if self.use_ddp:
            return self._prepare_dataloader_ddp(dataset, num_workers, class_balanced_sampling,
                                                is_train=is_train, is_eval_fraction=is_eval_fraction,
                                                drop_last=drop_last)

        if class_balanced_sampling:
            sampler = ClassBalancedRandomSampler(dataset, num_samples_per_class=self.num_samples_per_class)
        elif is_train and self.epoch_fraction < 1.0:
            sampler = EpochFractionSampler(dataset, fraction=self.epoch_fraction)
        elif is_eval_fraction and self.eval_fraction < 1.0:
            sampler = EpochFractionSampler(dataset, fraction=self.eval_fraction)
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=(sampler is None),
            num_workers=num_workers,
            drop_last=drop_last,
            sampler=sampler,
        )

    def _load_snapshot(self) -> None:
        loc = f"cuda:{self.local_rank}"
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

        snapshot, state_dict = load_state_dict_ivpt(snapshot_data)
        self.model.load_state_dict(state_dict)
        if self.eval_only:
            return
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.scheduler.step(snapshot.finished_epoch)
        if snapshot.epoch_test_accuracies is not None:
            self.epoch_test_accuracies = copy.deepcopy(snapshot.epoch_test_accuracies)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(
        self, 
        source, 
        targets, 
        train: bool = True, 
        vis_att_maps: bool = False, 
        curr_iter: int = 0, 
        vis_flag = None
    ) -> Tuple[Any, Any]:
        # torch.set_grad_enabled(train)：
        # 训练时 True → 建反向图、之后能 .backward()
        # 评估时 False → 不存中间量、省显存、也禁止误更新
        # 一个开关同时管「训练/评估」两条路
        # 
        # torch.amp.autocast(fp16, enabled=use_amp)：
        # 混合精度
        # 开了之后，里面的矩阵乘/卷积自动用 float16 算（快、省显存）
        # 数值敏感的（如 softmax/Norm 累加）仍留 float32
        # enabled=use_amp → 配置不开就是普通 fp32
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            all_features, dis_sim_maps, scores, maps_loss, (m_buffer, qm_buffer) = self.model(source) # (B, N+1, L) (B, P+1, N)

            # ★可解释：outputs 是平均后的「集体决定」，但 scores 保留了每个部件投了谁（头投麻雀、翅投知更鸟）
            outputs = scores.mean(dim=-1)  # (batch_size, num_classes)

            if train:
                # target = m_buffer[-1].detach()：
                # 取 m_buffer 最后一条＝最深/最粗那层的粗空间图，detach 当固定老师
                # 下面 consistency 让前面几层的粗图都去对齐它
                target = m_buffer[-1].detach()
                
                # loss_consistency = 
                
                # Forward pass of transformed images
                # 
                # generate_affine_trans_params(...)：
                # 随机生成一组仿射参数（转角 / 平移 / 缩放 / 切变），范围来自配置 eq_degrees 等
                angle, translate, scale, shear = generate_affine_trans_params(
                    degrees=self.eq_degrees, 
                    translate=self.eq_translate, 
                    scale_ranges=self.eq_scale_ranges, 
                    shears=self.eq_shear, 
                    img_size=[source.shape[2], 
                    source.shape[3]]
                )

                # Apply the affine transform to the source image
                # 
                # 把原图做这套仿射 → source_transformed（转一下/缩一下的同一张图）
                # 注意 shear=0.0 是写死的（生成了 shear 但应用时没用，细节）
                source_transformed = rigid_transform(
                    img=source, 
                    angle=angle, 
                    translate=translate, 
                    scale=scale, 
                    shear=0.0, 
                    invert=False
                )

                # equiv_maps = self.model(source_transformed)[3]：
                # 对变换后的图再跑一遍整个模型，只取第 4 个返回值 maps_loss（变换图的细分割图）
                equiv_maps = self.model(source_transformed)[3] # TODO

                # ============================================================

                # ★这是 _run_batch 训练时模型跑两次的原因：
                # 原图 1 次（拿分类 + 6 条正则）、变换图 1 次（只拿 equiv_maps 给 equivariance）
                # 训练算力≈评估的 2 倍

                # ============================================================

                # ===========================
                # 8 个 loss
                # 
                # 每个 loss 的讲法固定为：①管什么 → ②代码 → ③玩具手算 → ④把模型往哪推
                # 权重 l_* 用仓库真实默认值
                # （argument_parser_train.py）：
                # 分类/presence/equiv/orth/tv/pixel-entropy 都 1.0，enforced_presence 2.0，consistency 无权重(=1.0)
                # ===========================

                # ===
                # 分类 loss loss_classification —— 「类别猜得对不对」
                # 
                # ①管什么：标准多分类交叉熵，唯一直接盯「分类准确」的 loss，其余 7 条都在管部件图本身
                # ②代码：self.loss_fn_train
                # ③玩具手算：outputs=[1.060, 1.295]，真标签：知更鸟=1
                #            softmax: p(知更鸟) = 1/(1+0.791) = 0.559
                #            CE = -ln(0.559) = 0.582 × l_classification(1.0) = 0.582
                # ④往哪推：把知更鸟那一类的概率往 1 推（拉大正确类 logit）
                # ===

                # Classification loss
                loss_classification = self.loss_fn_train(outputs, targets) * self.l_classification

                # ===
                # 一致性 loss loss_consistency —— 「浅层粗图要像最深层粗图」（跨层 KL）
                # 
                # ①管什么：m_buffer[:-1]＝前几层（除最后一层）的粗空间图，
                #          逐个对 target（最后一层、已 detach）算 KL 散度并平均
                #          让不同深度对「同一个 patch 属于哪个粗部件」的判断一致
                #          -> 部件语义跨层稳定（这是 IVPT「层级原型」的黏合剂）
                #          注意它在总和里没有权重（隐含 1.0）
                # ②代码：consistency_loss(pred,target) 内部：
                #         Σ_通道 target·ln(target/pred)，
                #         再对位置取平均（ KL(target‖pred)，把 pred 拉向 target ）
                # ③玩具手算：m_buffer[:-1] = [注入f_maps] (pred)，
                #             target = 收尾f_maps。给收尾层一个更尖的版本（深层更自信）：
                # 
                #          pred(注入, 浅)              target(收尾, 深, detach)
                #         a     b     c     d            a     b     c     d
                # 粗0   .777  .117  .016  .450         .90   .08   .01   .48
                # 粗1   .205  .865  .020  .539         .08   .90   .01   .50
                # bg    .018  .018  .965  .009         .02   .02   .98   .02
                # 
                # 逐位置 KL = Σ_通道 target·ln(target/pred):
                # a: .90ln(.90/.777)+.08ln(.08/.205)+.02ln(.02/.018)= .1323-.0753+.0021=.0591
                # b: .08ln(.08/.117)+.90ln(.90/.865)+.02ln(.02/.018)=-.0304+.0357+.0021=.0074
                # c: .01ln(.01/.016)+.01ln(.01/.020)+.98ln(.98/.965)=-.0047-.0069+.0151=.0035
                # d: .48ln(.48/.450)+.50ln(.50/.539)+.02ln(.02/.009)= .0310-.0376+.0160=.0094
                # 
                # 位置平均 = (.0591+.0074+.0035+.0094)/4 = 0.0199 ≈ 0.020
                # 
                # toy：len(m_buffer[:-1]) = 1
                # 
                # ④往哪推：把浅层那张图改得更像深层那张（a 处差最大、贡献 0.059）
                #           KL(target‖pred) 每位置 ≥ 0
                # ===

                # Consistency loss
                loss_consistency = sum(
                    [
                        consistency_loss(maps_train, target) for maps_train in m_buffer[:-1]
                    ]
                ) / len(m_buffer[:-1])

                # ===
                # 全变差 loss loss_tv —— 「分割图空间上要平滑（别噪点）」
                # 
                # ①管什么：对每张细分割图算 Total Variation＝相邻像素差的绝对值之和（含背景通道）
                #           惩罚「东一块西一块」的碎片，逼部件成连通团块
                # ②代码：TotalVariationLoss(reduction="mean")：纵向差 + 横向差，绝对值求和成 score，再 / (B·H·W)
                # ③玩具手算：（对 3 通道全算，grid 2×2）：
                # 
                # 头 [[.982,    0],[.018, .495]]:  纵 |.018-.982|+|.495-0| + 横|0-.982|+|.495-.018| = 1.459 + 1.459
                # 翅 [[   0, .982],[.018, .495]]:  纵    .018    +  .487   + 横  .982  +    .477    = 0.505 + 1.459
                # bg [[.018, .018],[.965, .009]]:  纵    .947    +  .009   + 横    0   +    .956    = 0.956 + 0.956
                # 
                # score = (1.459+0.505+0.956) + (1.459+1.459+0.956) = 2.920 + 3.874 = 6.794
                # 
                # TV = 6.794 / (B·H·W=4) = 1.70 × l_tv(1.0) = 1.70
                # 
                # toy：len(maps_loss) = 1
                # 
                # ④往哪推：把跳变（如头通道 0→0.98 的硬边）磨平 → 部件区域更整片。
                #           ⚠️ toy 例子值偏大正是因为 2×2 太尖
                # ===

                # Total variation loss
                loss_tv = sum(
                    [
                        self.total_variation_loss(maps) * self.l_tv for maps in maps_loss
                    ]
                ) / len(maps_loss)

                # ===
                # 存在 loss loss_presence —— 「每个前景部件至少要在某处强烈出现」
                # 
                # ①管什么：maps[:, :-1] 丢掉背景、只留前景通道
                #           presence_loss(默认 original) ＝ 1 − mean_部件( max_空间(本部件激活) )
                #           逼每个前景部件在图里有个强响应点，防某部件「全程哑火」（退化坍缩）
                # ②代码：original 内部：avg_pool2d(maps,3) 先 3×3 平滑 
                #         → adaptive_max_pool2d(·,1) 取空间最大 
                #         → 跨 batch 取 max → 部件间求均值 → 1 − 它
                # ③玩具手算：⚠️ 3×3 平滑在 toy 2×2 上跑不动，当恒等处理
                # 
                # 头 空间 max = max(.982, 0, .018, .495) = .982
                # 翅 空间 max = .982
                # 部件均值 = (.982 + .982) / 2 = .982 → presence = 1 - .982 = 0.018 × l_presence(1.0) = 0.018
                # 
                # ④往哪推：值已很小＝头、翅各自都有强响应点（满足）
                #           若某部件最大才 0.1，presence→0.9 重罚
                #           与 enforced-presence 是一对：
                #           presence 要前景「各有强峰」，enforced 要背景「占住边角」，合起来 = 部件又尖又不占满
                # ===

                # Presence loss (fg) for landmarks
                loss_presence = sum(
                    [
                        self.presence_loss(maps=maps[:, :-1, :, :]) * self.l_presence for maps in maps_loss
                    ]
                ) / len(maps_loss)

                # ===
                # 正交 loss loss_orth —— 「不同部件的特征向量别趋同」
                # 
                # ①管什么：对 all_features [B,D,N+1]（含背景）按特征维归一化后两两算余弦，
                #           减单位阵（自己对自己=1 不罚），平方求均值。
                #           逼 N+1 个部件特征彼此正交（语义各管一摊），防多个部件学成同一个东西
                # ②代码：normed = normalize(feat, dim=1)
                #         G = normedᵀ·normed（余弦矩阵）
                #         mean((G − I)²)
                # ③玩具手算：（all_features 3 列：头=[1.74,.84] 翅=[.38,1.75] bg=[1,1]，bg 是占位）：
                #              归一化: 头 → [.901,.435]  翅 → [.212,.977]  bg → [.707,.707]
                #              余弦:   头·翅=.616   头·bg=.944   翅·bg=.841   (对角线减 1 → 0)
                #              mean((G-I)²) = 2·(.616²+.944²+.841²)/9 = 2·(.379+.891+.707)/9 = 3.956/9 = 0.44 × l_orth(1.0) = 0.44
                # 
                # ④往哪推：头·bg 余弦 0.94 太高（头部件和背景太像）→ 正交 loss 主要在把这俩推开
                #           头·翅 0.62 也偏高、一并压
                # ===

                # Orthogonality loss
                loss_orth = orthogonality_loss(all_features) * self.l_orth

                # ===
                # 等变 loss loss_equiv —— 「图转了，找到的部件也跟着转」
                # 
                # ①管什么：用那张变换图的部件图 equiv_maps，
                #           在 equivariance_loss 里把它反向变换回去（rigid_transform(invert=True)），
                #           再和原图部件图 maps_loss 比余弦。
                #           逼「先转图再找部件」≈「先找部件再转」 → 部件随物体几何一致地移动（不是死记图里固定坐标）
                # ②代码：rot_back = 反变换(equiv_maps)
                #         把前景通道拍平成向量；cos = cosine_similarity(orig, rot_back)；loss = 1 − mean(cos)
                # ③玩具手算：（⚠️ toy 2×2 没法真旋转、模型也没训，数字仅示意）：
                #              设头通道反变换回来 头'≈[.95,.02,.03,.45] vs 原 头=[.982,0,.018,.495]
                # 
                #              cos(头,头') = (.982·.95 + 0 + .018·.03 + .495·.45) / (‖头‖·‖头'‖) = 1.156 / (1.100·1.052) ≈ 0.999
                #              两前景平均 cos ≈ 0.998 -> loss_equiv = 1 - 0.998 = 0.002 × l_equiv(1.0) = 0.002
                # 
                # ④往哪推：值小＝部件对几何变换稳定（满足）
                #           不稳就大，逼模型学几何等变
                #           这是唯一需要第 2 次前向的 loss
                # ===

                # Equivariance loss: calculate rotated landmarks distance
                loss_equiv = sum(
                    [
                        equivariance_loss(maps_loss[i], equiv_maps[i], source, self.num_landmarks, translate, angle, scale, shear=0.0) * self.l_equiv for i in range(len(maps_loss))
                    ]
                ) / len(maps_loss)

                # ===
                # 强制存在 loss loss_enforced_presence —— 「背景部件要占住图像边角」
                # 
                # ①管什么：默认类型 enforced_presence（径向 mask 版）
                #           造一张「中心=0、四角=1」的径向权重图，乘到背景通道上，取空间最大，BCE 推向 1。
                #           逼背景部件在图像外围强烈存在 → 反过来把前景部件挤到中间、且不让前景吞掉边角。
                #           权重 2.0（8 个里最重）
                # ②代码：avg_pool2d(maps,3) → 乘径向 mask（中心 0 角落 1）
                #         → 取背景通道 → adaptive_max_pool2d 空间最大 → BCE(·, 1)
                # ③玩具手算：（⚠️ toy 2×2 上：3×3 平滑当恒等；径向 mask 在 2×2 退化＝四格全 1，因四格离中心等距）：
                # 
                #             背景通道 = [[.018,.018],[.965,.009]]，mask=1 → 空间 max = .965
                #             BCE(.965, 目标1) = -ln(.965) = 0.0356 × l_enforced_presence(2.0) = 0.071
                # 
                # ④往哪推：背景在 c 处已达 .965（近满足）
                #          真实 37×37 上 mask 把中心压 0、只认四角 
                #           → 「背景必须长在画面边缘」才算数
                # ===

                # Enforced presence loss
                loss_enforced_presence = sum(
                    [
                        self.enforced_presence_loss(maps) * self.l_enforced_presence for maps in maps_loss
                    ]
                ) / len(maps_loss)

                # ===
                # 逐像素熵 loss loss_pixel_wise_entropy —— 「每个 patch 要果断归一个部件」
                # 
                # ①管什么：把每个 patch 在「各部件通道」上的分布当类别分布，算熵，全图平均
                #          熵低=分配尖锐（一个 patch 主要归一个部件）
                #          逼软分割图别处处五五开、要果断
                # ②代码：clamp + 重归一 -> Categorical(probs).entropy() 逐像素 -> mean
                # ③玩具手算：（3 通道逐位置熵 −Σ p·ln p）：
                # 
                # a [.982,    0, .018]:  .982ln.982 + .018ln.018 = .0178 + .0723 = 0.090
                # b [   0, .982, .018]:  对称                                    = 0.090
                # c [.018, .018, .965]: 2·.0723 + .0344                          = 0.179
                # d [.495, .495, .009]: 2·.348  + .0424                          = 0.738   ← 最高(就是那个五五开的 d!)
                # 
                # 平均 = (.090+.090+.179+.738)/4 = 0.274 × l_pixel_wise_entropy(1.0) = 0.274
                # 
                # ④往哪推：patch d（49.5/49.5 平手那个）熵最高 0.738，正是这条 loss 重点惩罚的对象
                #          -> 逼 d 在头/翅里选一个
                # ===

                # Pixel-wise entropy loss
                loss_pixel_wise_entropy = sum(
                    [
                        pixel_wise_entropy_loss(maps) * self.l_pixel_wise_entropy for maps in maps_loss
                    ]
                ) / len(maps_loss)

                # loss = 八项相加（已各自乘过权重）→ 一个标量
                # 玩具：0.020+0.018+0.582+0.44+0.002+1.70+0.071+0.274 = 3.11
                loss = loss_consistency + loss_presence + loss_classification    + loss_orth \
                           + loss_equiv + loss_tv       + loss_enforced_presence + loss_pixel_wise_entropy

                # zero_grad(set_to_none=True)：
                # 清上一 batch 的梯度（置 None 比置 0 省显存/略快）
                # 必须在 backward 前清，否则梯度会累加
                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    # scaler.scale(loss).backward()：
                    # 先把 loss 放大再反向（防 fp16 小梯度下溢成 0）
                    self.scaler.scale(loss).backward()
                    if self.grad_norm_clip:
                        # clip_grad_norm_ 
                        # 把总梯度范数截到 grad_norm_clip（默认 1.0，防爆炸）
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 反向
                    loss.backward()
                    if self.grad_norm_clip:
                        # 裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                    # 更新
                    self.optimizer.step()

                # 给 _run_epoch 打日志/求平均
                # .item()：从 GPU 张量取出 Python float（脱离计算图，纯日志用）
                # 8 个分项 + 1 个 loss_total_train 一起塞进字典
                # 回到 _run_epoch 后用 AverageMeter 按样本数加权平均、每 log_freq 步打印
                # （就是训练时刷屏的 Total/Classification/Consistency/...）
                losses_dict = {
                    'loss_consistency': loss_consistency.item(),
                    'loss_classification_train': loss_classification.item(),
                    'loss_presence_train': loss_presence.item(),
                    'loss_orth_train': loss_orth.item(),
                    'loss_equiv_train': loss_equiv.item(),
                    'loss_total_train': loss.item(), 'loss_tv': loss_tv.item(),
                    'loss_enforced_presence': loss_enforced_presence.item(),
                    'loss_pixel_wise_entropy': loss_pixel_wise_entropy.item()
                }
            # （评估分支 train=False）只算 CE + 把 qm_buffer 整理成层级路由图
            else:
                # loss = loss_fn_eval(outputs, targets)：
                # 评估只算朴素交叉熵，那 7 条部件正则全不算、也不 backward（第 1 步 set_grad_enabled(False) 已禁梯度）
                # 玩具：loss_total_val = 0.582
                loss = self.loss_fn_eval(outputs, targets)
                losses_dict = {'loss_total_val': loss.item()}

                # ===
                # 之后整理 qm_buffer（每层的细→粗路由 [B,P,N]）成 maps_multi，给「层级原型可视化」
                # _run_epoch 里 relation_proto 那段用它统计「哪些细原型并进哪个粗部件」
                # 
                # maps[maps==0.1]=0：清掉「中性填充值」
                # ⚠️ 注意 forward 里空原型填的是 1/N＝真实 N=4 时是 0.25、不是 0.1，
                # 所以这行在当前配置几乎不触发 —— 像一处遗留常量（疑似只对 N=10 的旧配置生效），记一笔别依赖它
                # 
                # maps.mean(0)：跨 batch 平均 → [P,N]
                # / sum(-1)：每个细原型那一行重归一成 1
                # maps[isnan]=0：除零兜底
                # 
                # toy：
                # qm_buffer=[[[.79,.21],[.12,.88]]] 
                # → 无 0.1 → mean(batch=1) 不变 
                # → 行已和 1 → maps_multi=[[[.79,.21],[.12,.88]]]
                # （头主要进粗0、翅主要进粗1，正是层级结构）
                # ===

                maps_multi = []
                for lay, maps in enumerate(qm_buffer):
                    maps[maps==0.1] = 0
                    maps = maps.mean(0)
                    maps = maps / maps.sum(-1, keepdim=True)
                    maps[torch.isnan(maps)] = 0
                    maps_multi.append(maps)

                # vis_att_maps：仅每 10 个 batch、且主进程，把分割图叠到原图上存盘
                if vis_att_maps:
                    if vis_flag == None:
                        if curr_iter % 10 == 0 and is_main_process():
                            for lay, maps in enumerate(maps_loss):
                                self.vis_att_maps.show_maps(ims=source, maps=maps, epoch=self.current_epoch, # TODO
                                                            curr_iter=curr_iter, extra_info=lay)
                    else:
                        if curr_iter % 10 == 0 and is_main_process():
                            self.vis_att_maps.show_maps_hie(ims=source, maps_loss=maps_loss, epoch=self.current_epoch, # TODO
                                                        curr_iter=curr_iter, extra_info=lay, vis_flag=vis_flag)

        if train:
            # 训练：outputs（给 _run_epoch 更新 top1/top5 准确率）+ losses_dict（日志）
            return outputs, losses_dict
        else:
            # 评估（vis_flag 非 None，层级可视化的「第二趟重跑」）：
            # 函数走到底隐式返回 None —— 那趟只为生成叠加图、不要返回值（_run_epoch 里那次调用确实没接收返回）
            # 是有意的，不是漏写

            if vis_flag == None:
                # 评估（vis_flag=None，正常评估）：多还一个 maps_multi（层级可视化用）
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

        if self.use_ddp:
            dataloader.sampler.set_epoch(epoch)

        updates_per_epoch = (len(dataloader) + self.accum_steps - 1) // self.accum_steps
        num_updates = (epoch - 1) * updates_per_epoch

        vis_att_maps = True if epoch % self.save_every == 0 else False
        vis_att_maps = True if epoch == self.max_epochs else vis_att_maps
        vis_att_maps = True if epoch == 1 else vis_att_maps
        losses_dict = {}
        for key in self.loss_dict_train.keys():
            self.loss_dict_train[key].reset()
        for key in self.loss_dict_val.keys():
            self.loss_dict_val[key].reset()

        accuracies_dict = {}

        for key in self.acc_dict_train.keys():
            self.acc_dict_train[key].reset()
        for key in self.acc_dict_test.keys():
            self.acc_dict_test[key].reset()

        maps_buffer = []
        for it, mini_batch in enumerate(dataloader):
            source = mini_batch[0]
            targets = mini_batch[1]
            step_type = "Train" if train else "Eval"
            source = source.to(self.local_rank, non_blocking=True)
            targets = targets.to(self.local_rank, non_blocking=True)
            if train and self.mixup_fn is not None:
                source, targets = self.mixup_fn(source, targets)
            
            if train:
                batch_preds, losses_dict = self._run_batch(source, targets, train,
                                                        vis_att_maps=vis_att_maps, curr_iter=it)
            else:
                batch_preds, losses_dict, maps_multi = self._run_batch(source, targets, train,
                                                        vis_att_maps=vis_att_maps, curr_iter=it)
                maps_buffer.append(maps_multi)

            if train:
                for key in losses_dict.keys():
                    self.loss_dict_train[key].update(losses_dict[key], source.size(0))
                if self.mixup_fn is None:
                    for key in self.acc_dict_train.keys():
                        self.acc_dict_train[key].update(batch_preds, targets)
                num_updates += 1
                self.scheduler.step_update(num_updates=num_updates)
            else:
                for key in losses_dict.keys():
                    self.loss_dict_val[key].update(losses_dict[key], source.size(0))
                for key in self.acc_dict_test.keys():
                    self.acc_dict_test[key].update(batch_preds, targets)
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

        if train:
            for key in self.loss_dict_train.keys():
                losses_dict[key] = self.loss_dict_train[key].avg
            if self.mixup_fn is None:
                for key in self.acc_dict_train.keys():
                    accuracies_dict[key] = self.acc_dict_train[key].compute().item() * 100
            self.scheduler.step(epoch)
        else:
            for key in self.loss_dict_val.keys():
                losses_dict[key] = self.loss_dict_val[key].avg
            for key in self.acc_dict_test.keys():
                accuracies_dict[key] = self.acc_dict_test[key].compute().item() * 100
        return losses_dict, accuracies_dict

    def _save_snapshot(self, epoch, save_best: bool = False):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            epoch_test_accuracies=self.epoch_test_accuracies,
        )
        # save snapshot
        snapshot = asdict(snapshot)
        if self.is_snapshot_dir:
            save_path_base = self.snapshot_path
        else:
            save_path_base = os.path.dirname(self.snapshot_path)
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

    def train(self):
        for epoch in range(self.epochs_run, self.max_epochs):
            epoch += 1
            self.current_epoch = epoch
            self.model.train()
            loss_dict_train, acc_dict_train = self._run_epoch(epoch, self.train_loader, train=True)

            logging_dict = {'epoch': epoch,
                            'base_lr': self.optimizer.param_groups[0]['lr'],
                            'scratch_lr': self.optimizer.param_groups[-1]['lr'],
                            'modulation_lr': self.optimizer.param_groups[-2]['lr'],
                            'finer_lr': self.optimizer.param_groups[-3]['lr']}
            if self.local_rank == 0 and self.global_rank == 0:
                logging_dict.update(loss_dict_train)
                logging_dict.update(acc_dict_train)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            elif self.local_rank == 0 and epoch == self.max_epochs:
                self._save_snapshot(epoch)

            # Periodic eval (partial test set, every eval_every_n_epochs)
            should_eval = (self.test_loader and
                           epoch % self.eval_every_n_epochs == 0 and
                           epoch < self.max_epochs)
            if should_eval:
                self.model.eval()
                with torch.inference_mode():
                    loss_dict_val, acc_dict_test = self._run_epoch(epoch, self.test_loader, train=False)
                if self.local_rank == 0 and self.global_rank == 0:
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
            elif self.local_rank == 0 and self.global_rank == 0:
                # Log training-only metrics when we skip eval
                for logger in self.loggers:
                    logger.log(logging_dict)

        # ---- Final evaluation on full test set ----
        if self.test_loader_full:
            if self.local_rank == 0 and self.global_rank == 0:
                print("\n" + "=" * 60)
                print("Final evaluation on FULL test set")
                print("=" * 60)
            self.model.eval()
            with torch.inference_mode():
                loss_dict_val, acc_dict_test = self._run_epoch(self.max_epochs, self.test_loader_full, train=False)
            if self.local_rank == 0 and self.global_rank == 0:
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

    set_seeds(seed)
    # Loop through training and testing steps for a number of epochs
    if use_ddp:
        ddp_setup()

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
    if eval_only:
        model_trainer.test_only()
    else:
        model_trainer.train()
    if use_ddp:
        destroy_process_group()
