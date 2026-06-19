"""
Main entry point for IVPT training and classification evaluation.

Usage:
    # Multi-GPU training
    torchrun --nproc_per_node=4 train_net.py --data_path <path> ...

    # Single-GPU training
    python train_net.py --data_path <path> ...

    # Evaluation only
    python train_net.py --data_path <path> --eval_only --snapshot_dir <ckpt_dir>
"""

# ======================================
#
# 这个文件是整个训练流程的“总入口 + 总调度”：
# scripts/run_train.sh 里的 torchrun ... train_net.py ... 最终就是执行到这里的 ivpt_train_eval()。
#
# 它本身不做任何具体的训练计算，只负责按顺序把各个零件准备好：
#   参数 → 日志 → 检查点目录 → 数据增强 → 数据集 → 模型 → 损失 → 优化器 → 调度器
# 最后把这些零件一起交给真正干活的训练器 launch_ivpt_trainer()，由它跑训练循环、评估、存档。
#
# ======================================

import torch
# timer：一个高精度计时器，用来统计整个训练总共跑了多少秒
from timeit import default_timer as timer

# ======================================
# 下面这一批 import 就是上面说的“各个零件”的来源，按用途分组：

# parse_args：解析命令行参数(--n_pro / --lr / --freeze_backbone 等)，返回一个 args 对象
from argument_parser_train import parse_args
# load_transforms：构造数据增强流水线(训练用强增强、测试用弱增强)
from utils.data_utils.transform_utils import load_transforms
# build_optimizer：建优化器(本次是 Adam)；layer_group_matcher_ivpt：把模型参数按用途分组、各给不同学习率
from utils.training_utils.optimizer_params import build_optimizer, layer_group_matcher_ivpt
# build_scheduler：建学习率调度器(本次是 StepLR：每隔若干 epoch 把学习率乘以 gamma)
from utils.training_utils.scheduler_params import build_scheduler
# sync_bn_conversion：多卡时把 BN 换成跨卡同步的 SyncBN；check_snapshot：建好存检查点的目录
from utils.misc_utils import sync_bn_conversion, check_snapshot
# multi_gpu_check：检测当前是不是多 GPU 环境(决定要不要走 DDP 分布式训练)
from utils.training_utils.ddp_utils import multi_gpu_check
# get_train_loggers：准备日志记录器(开了 --wandb 才有内容，否则返回空列表)
from utils.wandb_params import get_train_loggers
# launch_ivpt_trainer：真正的训练器入口，里面是数据加载、前向、算损失、反向、评估的完整循环
from engine.distributed_trainer_ivpt import launch_ivpt_trainer
# get_dataset：构造训练集和测试集(本次是 CUB-200 鸟类细粒度数据集)
from data_sets.builder import get_dataset
# load_model_ivpt：搭出 IVPT 模型(冻结的 DINOv2 ViT-Base 主干 + 部件发现层)
from models.builder import load_model_ivpt
# load_classification_loss：分类损失(本次是普通交叉熵)；load_loss_hyper_params：读取各项辅助损失的权重和超参
from engine.losses.builder import load_classification_loss, load_loss_hyper_params

# ======================================

# 打开 cuDNN 的自动调优：让 cuDNN 针对固定的输入尺寸自动挑最快的卷积/矩阵算法
# 注：输入尺寸固定(本次图像统一 518×518)时能加速；若输入尺寸频繁变化反而会变慢
torch.backends.cudnn.benchmark = True

# ======================================

def ivpt_train_eval():
    # === 第 1 步：解析命令行参数 ===
    # 把 run_train.sh 里传进来的所有 --xxx 解析成 args 对象，后面所有零件都从 args 取配置
    args = parse_args()

    # === 第 2 步：准备日志记录器 ===
    # 没开 --wandb 时返回空列表 []，相当于不往 wandb 记日志
    train_loggers = get_train_loggers(args)

    # === 第 3 步：准备检查点目录 ===
    # 若 snapshot_dir 不存在就新建；若目录里已有旧检查点，后续训练器会自动接着它续训
    # Create directory to save training checkpoints, otherwise load the existing checkpoint
    check_snapshot(args)

    # === 第 4 步：构造数据增强流水线 ===
    # 本次 augmentations_to_use=cub_original：
    #   训练增强较强(Resize/翻转/颜色抖动/随机仿射/随机裁剪)，测试增强很弱(只 Resize + 中心裁剪)
    # Get the transforms and load the dataset
    train_transforms, test_transforms = load_transforms(args)

    # === 第 5 步：加载数据集 ===
    # 返回训练集、测试集，以及类别总数 num_cls(CUB 是 200 类)
    # num_cls 后面会传给模型(决定分类头输出维度)和损失函数
    # Load the dataset
    dataset_train, dataset_test, num_cls = get_dataset(args, train_transforms, test_transforms)

    # === 第 6 步：搭建模型 ===
    # 加载预训练的 DINOv2 ViT-Base 主干，并在最后几个 block 插入“部件发现”结构；
    # 因为 --freeze_backbone，主干绝大部分参数被冻结，只训练新增的部件相关层 + 分类头
    # Load the model
    model = load_model_ivpt(args, num_cls)

    # === 第 7 步：判断是否多卡 ===
    # Check if there are multiple GPUs
    use_ddp = multi_gpu_check()
    # 多卡时才需要把普通 BatchNorm 换成跨卡同步的 SyncBatchNorm(让各卡统计量一致)
    # 本次单卡(NUM_GPUS=1)，use_ddp 为 False，所以这一步会跳过
    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if use_ddp:
        model = sync_bn_conversion(model)

    # === 第 8 步：分类损失 ===
    # 本次没开 mixup/cutmix 且 smoothing=0，所以 loss_fn 就是普通的 nn.CrossEntropyLoss；mixup_fn 为 None
    # Load the loss function
    loss_fn, mixup_fn = load_classification_loss(args, dataset_train, num_cls)

    # === 第 9 步：各项辅助损失的权重与超参 ===
    # IVPT 除分类损失外还有一堆部件相关的正则损失(presence/等变/全变差/像素熵 等)，
    # 这里把它们的权重打包成 loss_hyperparams；
    # eq_affine_transform_params 是“等变损失”要用的随机仿射变换范围(旋转/平移/缩放/错切)
    # Load the loss hyperparameters
    loss_hyperparams, eq_affine_transform_params = load_loss_hyper_params(args)

    # === 第 10 步：优化器与学习率调度器 ===
    # layer_group_matcher_ivpt：把参数分成几组(冻结主干 1x、不加 weight decay 的、finer 层、调制层、从头训练层)，
    #   不同组在基准 lr 上配不同倍率(scratch/modulation ×1e4、finer ×2e2)
    # Define the optimizer and scheduler
    param_groups = layer_group_matcher_ivpt(args, model)
    # build_optimizer：本次用 Adam(并按归一化公式把 weight decay 换算到各参数组)
    optimizer = build_optimizer(args, param_groups, dataset_train)
    # build_scheduler：本次用 StepLR(每 scheduler_step_size=4 个 epoch 把学习率乘以 scheduler_gamma=0.5)
    scheduler = build_scheduler(args, optimizer)

    # === 第 11 步：开始计时 ===
    # Start the timer
    start_time = timer()

    # === 第 12 步：交给训练器，正式开训 ===
    # 前面准备好的所有零件(模型/数据/优化器/调度器/损失/各种超参)一次性传进去；
    # 训练循环、定期评估、存检查点、可视化部件注意力图，全部在 launch_ivpt_trainer 内部完成
    # Setup training and save the results
    launch_ivpt_trainer(model=model,
                          train_dataset=dataset_train,
                          test_dataset=dataset_test,
                          batch_size=args.batch_size,                 # 每张卡的 batch size(本次 12)
                          optimizer=optimizer,
                          scheduler=scheduler,
                          loss_fn=loss_fn,                            # 分类损失(交叉熵)
                          epochs=args.epochs,                         # 总训练轮数(本次 25)
                          save_every=args.save_every_n_epochs,        # 每多少个 epoch 存一次检查点
                          loggers=train_loggers,
                          log_freq=args.log_interval,                 # 每多少个 iter 打一次日志
                          use_amp=args.use_amp,                       # 是否用混合精度(本次未开)
                          snapshot_path=args.snapshot_dir,
                          grad_norm_clip=args.grad_norm_clip,         # 梯度裁剪上限(本次 2.0)
                          num_workers=args.num_workers,
                          mixup_fn=mixup_fn,                          # None(本次没开 mixup)
                          seed=args.seed,
                          eval_only=args.eval_only,                   # False：走训练流程，而不是纯评估
                          loss_hyperparams=loss_hyperparams,          # 各辅助损失的权重
                          eq_affine_transform_params=eq_affine_transform_params,  # 等变损失用的随机仿射变换范围
                          use_ddp=use_ddp,
                          sub_path_test=args.image_sub_path_test,
                          dataset_name=args.dataset,
                          amap_saving_prob=args.amap_saving_prob,     # 训练时按这个概率保存“部件分配图”可视化
                          class_balanced_sampling=args.use_class_balanced_sampling,  # 类平衡采样(本次 False)
                          num_samples_per_class=args.num_samples_per_class,
                          n_pro=args.n_pro,                           # 各注入层的原型数量字符串 "17,14,11,8,5"
                          enable_hierarchy_vis=args.enable_hierarchy_vis,  # 层级原型可视化(仅 eval-only 用，本次 False)
                          epoch_fraction=args.epoch_fraction,         # 每个 epoch 用多少比例的训练数据(本次 1.0=全量)
                          eval_every_n_epochs=args.eval_every_n_epochs,    # 每多少个 epoch 评估一次(本次 5)
                          eval_fraction=args.eval_fraction,           # 训练中途评估用多少比例的测试集(本次 0.1)
                          )

    # === 第 13 步：结束计时并打印总耗时 ===
    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


# ======================================

if __name__ == "__main__":
    ivpt_train_eval()
