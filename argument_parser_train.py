"""
Argument parser for IVPT training and classification evaluation.

Defines all command-line arguments for model architecture, data paths,
training hyper-parameters, loss weights, scheduler, optimizer, logging,
and visualization options.
"""

# ======================================
#
# 这个文件只做一件事：用 argparse 把命令行里的 --xxx 全部登记下来，解析成一个 args 对象返回。
# run_train.sh 里 torchrun ... train_net.py 后面跟的所有 flag，都必须先在这里 add_argument 登记过，
# 才能被识别、才有默认值。train_net.py 拿到 args 后，各个零件再按需从 args 取自己要的配置。
#
# 阅读提示：
#   - 参数非常多，但本次 run_train.sh 真正传了的只是其中一部分；凡注释里带【本次】的，
#     就是这次训练实际用到 / 被命令行覆盖了的；其余大多是别的数据集 / 消融实验 / 默认关闭的功能。
#   - default 是“命令行不传时的默认值”；命令行传了就以命令行为准(覆盖 default)。
#   - action='store_true' 的参数是“开关”：命令行写了它就为 True，不写就是 False。
#
# ======================================

import argparse


def parse_args():
    # 创建解析器；description 会在执行 python train_net.py --help 时显示在帮助开头
    parser = argparse.ArgumentParser(
        description='IVPT: Interpretable Visual Prompt Tuning – Training & Evaluation'
    )
    # 主干网络结构名(timm 模型名)【本次=vit_base_patch14_reg4_dinov2.lvd142m】
    parser.add_argument('--model_arch', default='vit_base_patch14_dinov2.lvd142m', type=str,
                        help='pick model architecture')
    # 用 torchvision 版 ResNet 实现(本次用 ViT，不涉及)
    parser.add_argument('--use_torchvision_resnet_model', default=False, action='store_true')

    # === 数据相关 ===
    # Data
    # 数据集根目录【本次=datasets/CUB_200_2011】(required=True：必须传，不传会报错)
    parser.add_argument('--data_path',
                        help='directory that contains cub files', required=True)
    # 训练图所在的子目录【本次=images】
    parser.add_argument('--image_sub_path_train', default='images',
                        help='subdirectory that contains training images')
    # 测试图所在的子目录【本次=images】
    parser.add_argument('--image_sub_path_test', default='images',
                        help='subdirectory that contains test images')
    # 数据集名【本次=cub】(决定用哪个 Dataset 类)
    parser.add_argument('--dataset', default='cub', type=str)
    # 用多大比例的官方训练集来训练，剩下的留作验证集【本次=1，即全部拿来训练、不留 val】
    parser.add_argument('--train_split', default=0.9, type=float, help='fraction of training data to use')
    # 用哪个划分来评估【本次=test】
    parser.add_argument('--eval_mode', default='val', choices=['train', 'val', 'test'], type=str,
                        help='which split to use for evaluation')
    # 下面这几个标注 / 元数据路径是 PartImageNet / PlantNet 等数据集才用的，CUB 不涉及
    parser.add_argument('--anno_path_train', default='', type=str, required=False)
    parser.add_argument('--anno_path_test', default='', type=str, required=False)
    parser.add_argument('--metadata_path', default='', type=str, required=False)
    parser.add_argument('--species_id_to_name_file', default='', type=str, required=False)

    # === 训练相关 ===
    # Training
    # 检查点(模型权重)保存目录【本次=./snapshot】
    parser.add_argument('--snapshot_dir', type=str)
    # 每多少个 epoch 存一次检查点【本次=10】
    parser.add_argument('--save_every_n_epochs', default=10, type=int)
    # 每张卡的 batch size【本次=12】
    parser.add_argument('--batch_size', type=int, default=16)
    # 总训练轮数【本次=25】
    parser.add_argument('--epochs', type=int, default=28)
    # DataLoader 取数据用的子进程数【本次=4】
    parser.add_argument('--num_workers', type=int, default=4)
    # 随机种子(默认 42，保证可复现)
    parser.add_argument('--seed', default=42, type=int)

    # === 类平衡采样(本次未用) ===
    # Class balanced training sampling
    # 是否在 mini-batch 内平衡各类别(本次 False)；注：该采样器依赖数据集里并未实现的方法，开了会直接崩
    parser.add_argument('--use_class_balanced_sampling', default=False, action='store_true')
    # 类平衡采样时每类取多少张(仅上面开关为 True 时才用到)
    parser.add_argument('--num_samples_per_class', default=100, type=int)

    # === epoch_fraction：每个 epoch 只用一部分训练数据 ===
    # Epoch fraction: use only a fraction of training data per epoch
    # Data is split into ceil(1/fraction) shards and cycled across epochs,
    # guaranteeing full coverage over multiple epochs.
    # 把数据切成 ceil(1/fraction) 份、各 epoch 轮换用其中一份，多轮后正好覆盖全量【本次=1.0，每轮用全量】
    parser.add_argument('--epoch_fraction', default=1.0, type=float,
                        help='Fraction of training data per epoch (0,1]. '
                             'e.g. 0.1 = 10%% data per epoch, full coverage every 10 epochs.')

    # === 评估频率与评估比例 ===
    # Eval frequency and eval fraction
    # 训练过程中每多少个 epoch 评估一次【本次=5】
    parser.add_argument('--eval_every_n_epochs', default=1, type=int,
                        help='Run evaluation every N epochs during training (default: 1 = every epoch).')
    # 训练中途评估只用多少比例的测试集【本次=0.1】(训练结束后的最终评估始终用全量测试集)
    parser.add_argument('--eval_fraction', default=1.0, type=float,
                        help='Fraction of test data to use for periodic eval during training (0,1]. '
                             'Full test set is always used for the final evaluation after training.')

    # === 注意力图保存概率 ===
    # Attention map saving probability
    # 训练时按这个概率把“部件分配图”存成图片(默认 0.2；本次未在命令行覆盖，用默认)
    parser.add_argument('--amap_saving_prob', default=0.2, type=float) # TODO

    # === 其它训练杂项 ===
    # * Misc training params
    # 梯度裁剪上限：反向后把梯度范数裁到不超过该值，防梯度爆炸【本次=2.0】
    parser.add_argument('--grad_norm_clip', default=2.0, type=float)
    # 是否用混合精度训练(AMP，fp16)(本次未开)
    parser.add_argument('--use_amp', action='store_true', default=False)

    # === 评估相关 ===
    # Evaluation params
    # 只评估、不训练(本次 False，走训练流程)
    parser.add_argument('--eval_only', default=False, action='store_true',
                        help='Whether to only eval the model')
    # 评估 resize 时的裁剪比例(仅 timm 弱增强分支用到；cub_original 不读它)
    parser.add_argument('--crop_pct', type=float, default=None)

    # === Mixup / CutMix(本次全部未用) ===
    # * Mixup params
    # 是否开启 mixup/cutmix(本次未开 --turn_on_mixup_or_cutmix，所以下面这些都不生效)
    parser.add_argument('--turn_on_mixup_or_cutmix', action='store_true')
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # === 数据增强参数 ===
    # Augmentation parameters
    # 用哪一套增强【本次=cub_original】：timm=timm 自动增强；cub_original=细粒度分类常用的标准增强
    parser.add_argument('--augmentations_to_use', type=str, default='cub_original',
                        choices=['timm', 'torchvision', 'cub_original'])
    # 输入图像尺寸【本次=518】
    parser.add_argument('--image_size', default=448, type=int)
    # 下面 color_jitter / aa / train_interpolation / reprob 等只在 timm 增强分支生效；cub_original 不读它们
    # 颜色抖动强度(仅 timm 分支)
    parser.add_argument('--color_jitter', type=float, default=0.1, metavar='PCT',
                        help='Color jitter factor (default: 0.1)')
    # AutoAugment 策略(仅 timm 分支)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    # 标签平滑【本次=0，等价于普通交叉熵】
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.)')
    # 训练插值方式(仅 timm 分支)
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # 用 ImageNet 默认均值方差(store_false：默认 True，写了才变 False)
    parser.add_argument('--imagenet_default_mean_and_std', action='store_false', default=True)
    # 水平翻转概率(cub_original 增强会用到，默认 0.5)
    parser.add_argument('--hflip', type=float, default=0.5, help='Horizontal flip probability')
    # 垂直翻转概率(默认 0.0，相当于不做垂直翻转)
    parser.add_argument('--vflip', type=float, default=0., help='Vertical flip probability')

    # === Random Erase(随机擦除，仅 timm 增强分支生效，本次不用) ===
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # === 模型参数 ===
    # Model params
    # 各注入层的原型数量字符串【本次=17,14,11,8,5】(在最后 5 个 block 分别注入这么多原型)
    parser.add_argument('--n_pro', help='number of prototypes to generate prompts',
                        default="17,14,11,8,5", type=str) # TODO
    # 主干加载预训练权重【本次=开】(需联网下载)
    parser.add_argument('--pretrained_start_weights', default=False, action='store_true')
    # drop path 概率【本次=0.0】(主干冻结时本就用不到)
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    # 模型 stride(仅用 timm 的 CNN 模型时才用)
    parser.add_argument('--output_stride', type=int, default=32, help='stride of the model')
    # 冻结主干【本次=开】(只训练新增的部件发现层 + 分类头)
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    # 在冻结主干基础上进一步冻结(ViT 全冻、只留部件发现层)(本次未开)
    parser.add_argument('--freeze_params', default=False, action='store_true')

    # parser.add_argument('--n_pro', type=int, default=4) # TODO

    # === 优化器参数 ===
    # * Optimizer params
    # 优化器类型【本次=adam】
    parser.add_argument('--optimizer_type', default='adam', type=str)
    # 权重衰减(归一化形式)【本次=0】
    parser.add_argument('--weight_decay', default=0, type=float, help='normalized weight decay')
    # 下面 momentum / betas / dampening / trust_coeff 等是各类优化器的专属超参；Adam 只用到 betas1/betas2
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--betas1', default=0.9, type=float)
    parser.add_argument('--betas2', default=0.999, type=float)
    parser.add_argument('--dampening', default=0.0, type=float)
    parser.add_argument('--trust_coeff', default=0.001, type=float)
    parser.add_argument('--always_adapt', action='store_true', default=False)
    parser.add_argument('--turn_off_grad_averaging', action='store_true', default=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)

    # === 调度器参数 ===
    # * Scheduler params
    # 学习率调度器类型【本次=steplr】
    parser.add_argument('--scheduler_type', default='cosine',
                        choices=['cosine', 'linearlr', 'steplr'],
                        type=str)
    # warmup 轮数(本次默认 0，无 warmup)
    parser.add_argument('--scheduler_warmup_epochs', default=0, type=int)
    # warmup 起始学习率
    parser.add_argument('--warmup_lr', type=float, default=1e-6)
    # cosine 重启因子(仅 cosine 调度器用)
    parser.add_argument('--scheduler_restart_factor', default=1, type=int)
    # StepLR 衰减系数：每到步长就把学习率乘以它【本次=0.5】
    parser.add_argument('--scheduler_gamma', default=0.1, type=float)
    # StepLR 步长：每多少个 epoch 衰减一次【本次=4】
    parser.add_argument('--scheduler_step_size', default=10, type=int)
    # cosine 调度器的学习率下界
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--cosine_cycle_limit', default=1, type=int)

    # === 各参数组的学习率与倍率 ===
    # * LR params for each param group
    # 基准学习率【本次=0.866e-6】(= 1e-6 × √(12/16)，按 batch 做了平方根缩放)
    parser.add_argument('--lr', default=1e-6, type=float)
    # scratch 组(从头训练的层)在基准 lr 上的倍率【本次=1e4】
    parser.add_argument('--scratch_lr_factor', default=1e4, type=float)
    # finer 组在基准 lr 上的倍率【本次=2e2】
    parser.add_argument('--finer_lr_factor', default=1e3, type=float)
    # modulation(调制)组在基准 lr 上的倍率【本次=1e4】
    parser.add_argument('--modulation_lr_factor', default=1e4, type=float)

    # === wandb 日志(本次未开 --wandb，下面都不生效) ===
    # Wandb params
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--job_type', default='', type=str)
    # 每多少个 iter 打一次日志(本次 wandb 没开，但这个间隔仍用于普通打印)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--group', default='vit_base', type=str)
    parser.add_argument('--wandb_entity', default='', type=str)
    parser.add_argument('--wandb_mode', default='online', type=str, choices=['online', 'offline'])

    # === 断点续训(本次未用) ===
    # * Resume training params
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--wandb_resume_id', default=None, type=str)

    # === 各项损失的权重与超参 ===
    # Loss hyperparameters
    # 分类损失(交叉熵)权重【本次=1】
    parser.add_argument('--classification_loss', default=1.0, type=float)
    # presence 损失权重【本次=1】(约束部件“该出现就出现”)
    parser.add_argument('--presence_loss', default=1.0, type=float)
    # presence 损失里的 beta 超参
    parser.add_argument('--presence_loss_beta', default=0.1, type=float)
    # presence 损失的具体形式【本次=original】
    parser.add_argument('--presence_loss_type', default="original",
                        choices=["original", "soft_constraint", "tanh", "soft_tanh"], type=str)
    # concentration(集中度)损失权重(本次=0；且训练器代码里根本没读它，属装载但未使用)
    parser.add_argument('--concentration_loss', default=0, type=float)
    # 等变损失权重【本次=1】(图像仿射变换后，部件注意力图应同步变换)
    parser.add_argument('--equivariance_loss', default=1.0, type=float)
    # 部件正交损失权重(默认 1，让不同部件的表示彼此区分)
    parser.add_argument('--orthogonality_loss_landmarks', default=1.0, type=float)
    # 全变差损失权重【本次=1】(让注意力图空间上平滑、连片)
    parser.add_argument('--total_variation_loss', default=1.0, type=float)
    # enforced presence 损失权重【本次=1】(进一步强制部件在前景有响应)
    parser.add_argument('--enforced_presence_loss', default=2.0, type=float)
    # enforced presence 损失的具体形式【本次=enforced_presence】
    parser.add_argument('--enforced_presence_loss_type', default="enforced_presence", choices=["linear", "log", "mse", "enforced_presence"],
                        type=str)
    # 像素级熵损失权重【本次=1】(让每个像素尽量只归属一个部件)
    parser.add_argument('--pixel_wise_entropy_loss', default=1.0, type=float)
    # 下面这一组 imbalanced_noised_topk 是另一种分类损失分支(本次未开，相关超参不生效)
    parser.add_argument('--use_imbalanced_noised_topk', default=False, action='store_true')
    parser.add_argument('--topk_k', default=5, type=int)
    parser.add_argument('--topk_epsilon', default=0.01, type=float)
    parser.add_argument('--max_m', default=0.2, type=float)
    parser.add_argument('--topk_scale', default=60, type=int)
    parser.add_argument('--topk_n_sample', default=5, type=int)

    # === 等变损失用的随机仿射变换范围 ===
    # Equivariance affine transform params
    # 旋转角度上限(度)：等变时图像随机旋转 ±degrees
    parser.add_argument('--degrees', default=90, type=float)
    # 平移比例上限 x / y(占图宽高的比例)
    parser.add_argument('--translate_x', default=0.11, type=float)
    parser.add_argument('--translate_y', default=0.11, type=float)
    # 缩放下界 / 上界
    parser.add_argument('--scale_l', default=0.8, type=float)
    parser.add_argument('--scale_u', default=1.4, type=float)
    # 错切角度 x / y
    parser.add_argument('--shear_x', default=0.0, type=float)
    parser.add_argument('--shear_y', default=0.0, type=float)

    # === Part Dropout ===
    # Part Dropout
    # 训练时按这个概率随机丢弃部件(默认 0.3，正则化、防止过度依赖个别部件)
    parser.add_argument('--part_dropout', default=0.3, type=float)

    # === 给 ViT 输出特征加高斯噪声(本次 0，即不加) ===
    # Add noise to vit output features
    parser.add_argument('--noise_variance', default=0.0, type=float)

    # === Gumbel-Softmax ===
    # Gumbel Softmax
    # 对部件注意力图用 Gumbel-Softmax【本次=开】(让“像素→部件”的分配更接近硬分配，又保持可导)
    parser.add_argument('--gumbel_softmax', default=False, action='store_true')
    # Gumbel-Softmax 温度(默认 1.0)
    parser.add_argument('--gumbel_softmax_temperature', default=1.0, type=float)
    # 是否用 hard 直通估计(本次未开)
    parser.add_argument('--gumbel_softmax_hard', default=False, action='store_true')

    # === 原型调制类型 ===
    # Modulation
    # 调制方式【本次=layer_norm】
    parser.add_argument('--modulation_type', default="original",
                        choices=["original", "layer_norm", "parallel_mlp", "parallel_mlp_no_bias",
                                 "parallel_mlp_no_act", "parallel_mlp_no_act_no_bias", "none"],
                        type=str)

    # === 分类器类型 ===
    # Classifier type
    # 类别预测用的分类器【本次=linear(默认)】(independent_mlp 分支本次不用)
    parser.add_argument('--classifier_type', default="linear",
                        choices=["linear", "independent_mlp"], type=str)

    # === 数组作业：同一设置下用多个随机种子批量训练(本次未用) ===
    # Array training job
    parser.add_argument('--array_training_job', default=False, action='store_true',
                        help='Whether to run as an array job (i.e. training with multiple random seeds on the same settings)')

    # === 层级原型可视化(仅 eval-only 模式用)(本次未用) ===
    # Hierarchical prototype visualization
    parser.add_argument('--enable_hierarchy_vis', default=False, action='store_true',
                        help='Enable hierarchical prototype visualization during eval-only runs')

    # 真正去解析命令行，得到 args 对象并返回
    args = parser.parse_args()
    return args
