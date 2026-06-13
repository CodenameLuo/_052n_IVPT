"""Optimizer builder and IVPT-specific parameter group matching."""

# ======================================
#
# 这个文件对应 train_net.py 第 10 步，干两件事：
#   layer_group_matcher_ivpt：把模型参数分成 5 组，每组配不同的学习率倍率和 weight decay；
#                             冻结主干就在这里把主干参数的 requires_grad 关掉。
#   build_optimizer          ：按 optimizer_type 造优化器(本次 Adam)，weight decay 走归一化公式。
#
# 为什么要分组、为什么倍率差这么大(主干 1× vs 新层 1e4×)：
#   主干是预训练好的、且本次被冻结，基准 lr 本身就极小(0.866e-6)；而新增的部件层是从零学的，
#   得用大得多的有效学习率才学得动，所以给它们乘上 2e2 ~ 1e4 的倍率。
#
# ======================================

import math

import torch
# Lars / Lamb 是别的优化器(本次用 Adam，不走)
from timm.optim.lars import Lars
from timm.optim.lamb import Lamb

from utils.training_utils.ddp_utils import calculate_effective_batch_size


# ======================================

# 按 optimizer_type 造优化器；params_groups 是上面分好的 5 组(每组自带 lr / weight_decay)
def build_optimizer(args, params_groups, dataset_train):
    """
    Function to build the optimizer
    :param args: arguments from the command line
    :param params_groups: parameters to be optimized
    :param dataset_train: training dataset
    :return: optimizer
    """
    grad_averaging = not args.turn_off_grad_averaging
    # 算出(归一化后的)weight decay，传给优化器；本次 weight_decay=0，所以结果也是 0
    weight_decay = calculate_weight_decay(args, dataset_train)
    if args.optimizer_type == 'adamw':
        return torch.optim.AdamW(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay)
    elif args.optimizer_type == 'sgd':
        return torch.optim.SGD(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                               nesterov=True)
    # 【本次走这条】Adam：betas=(0.9,0.999)，lr 取每组各自的(这里的 args.lr 只是默认值)
    elif args.optimizer_type == 'adam':
        return torch.optim.Adam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                weight_decay=weight_decay)
    elif args.optimizer_type == 'nadam':
        return torch.optim.NAdam(params=params_groups, betas=(args.betas1, args.betas2), lr=args.lr,
                                 weight_decay=weight_decay)
    # —— 以下 Lars / Lamb 一系列变体本次都不走 ——
    elif args.optimizer_type == 'lars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlars':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'larc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'nlarc':
        return Lars(params=params_groups, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay,
                    dampening=args.dampening, nesterov=True, trust_coeff=args.trust_coeff, trust_clip=True,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lamb':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=False,
                    always_adapt=args.always_adapt)
    elif args.optimizer_type == 'lambc':
        return Lamb(params=params_groups, lr=args.lr, betas=(args.betas1, args.betas2), weight_decay=weight_decay,
                    grad_averaging=grad_averaging, max_grad_norm=args.max_grad_norm, trust_clip=True,
                    always_adapt=args.always_adapt)
    else:
        raise NotImplementedError(f'Optimizer {args.optimizer_type} not implemented.')


# ======================================

# 归一化 weight decay(出自 AdamW 论文)：把命令行给的 weight_decay 按训练总步数缩放一下
# 公式：wd = norm_wd × sqrt(1 / (每轮迭代数 × 轮数))；本次 norm_wd=0，所以结果恒为 0
def calculate_weight_decay(args, dataset_train):
    """
    Function to calculate the weight decay
    Implementation of normalized weight decay as per the paper "Decoupled Weight Decay Regularization": https://arxiv.org/pdf/1711.05101.pdf
    :param args: Arguments from the command line
    :param dataset_train: Training dataset
    :return: weight_decay: Weight decay
    """
    # 有效 batch size(单卡=batch_size；多卡=batch_size×卡数)
    batch_size = calculate_effective_batch_size(args)
    # 每个 epoch 的迭代数(drop_last=True 所以用整除)
    num_iterations = len(dataset_train) // batch_size  # Since we set drop_last=True
    norm_weight_decay = args.weight_decay
    weight_decay = norm_weight_decay * math.sqrt(1 / (num_iterations * args.epochs))
    return weight_decay


# ======================================

# 把模型参数分成 5 组，每组配不同学习率倍率/weight decay；并在这里按 freeze_backbone 关掉主干梯度
def layer_group_matcher_ivpt(args, model):
    """
    Function to group the parameters of the model into different groups
    :param args: Arguments from the command line
    :param model: Model to be trained
    :return: param_groups: Parameters grouped into different groups
    """
    # 按参数名里是否含这些关键字来归组：
    # scratch：从零训练的分类头(给最大学习率，且保留默认 weight decay)
    scratch_layers = ["fc_class_landmarks", 'fc_class_landmarks_dist']
    # modulation：调制层 + p_linear + p_classifier(从零训练，给最大学习率、wd=0)
    modulation_layers = ["modulation", "p_linear", "p_classifier"]
    # finer：要微调的少量主干相关参数 + 原型/偏置(cls/pos/reg token、p_bias、p_token)，给中等学习率倍率
    finer_layers = ["cls_token", "pos_embed", "reg_token", "p_bias", "p_token"]
    # unfrozen：留空 -> 下面对应的 elif 分支其实永不触发(死分支)
    unfrozen_layers = []
    scratch_parameters = []
    modulation_parameters = []
    backbone_parameters_wd = []
    no_weight_decay_params = []
    finer_parameters = []

    # 遍历每个参数，按名字落到对应的组里
    for name, p in model.named_parameters():
        # 分类头 -> scratch 组(始终训练)
        if any(x in name for x in scratch_layers):
            scratch_parameters.append(p)
            p.requires_grad = True

        # 调制/投影/软分配头 -> modulation 组(始终训练)
        elif any(x in name for x in modulation_layers):
            modulation_parameters.append(p)
            p.requires_grad = True

        # 原型/偏置/各 token -> finer 组(始终训练)
        elif any(x in name for x in finer_layers):
            finer_parameters.append(p)
            p.requires_grad = True

        # 死分支(unfrozen_layers 为空，永不进来)
        elif any(x in name for x in unfrozen_layers):
            no_weight_decay_params.append(p)
            if args.freeze_params:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # 其余 = 主干本体
        else:
            # 本次 freeze_backbone=True -> 关掉主干梯度(不更新；但仍会作为参数组留在优化器里)
            if args.freeze_backbone:
                p.requires_grad = False
            else:
                p.requires_grad = True

            # 一维参数(bias、norm 的权重)不加 weight decay；其余(权重矩阵)加 weight decay
            if p.ndim == 1:
                no_weight_decay_params.append(p)
            else:
                backbone_parameters_wd.append(p)

    # 组装 5 个参数组，每组指定 lr(用倍率)和 weight_decay：
    #   1) 主干带 wd     : 基准 lr ×1
    #   2) 主干不带 wd   : 基准 lr ×1，wd=0
    #   3) finer 组      : 基准 lr ×finer_lr_factor(本次 2e2)，wd=0
    #   4) modulation 组 : 基准 lr ×modulation_lr_factor(本次 1e4)，wd=0
    #   5) scratch 组    : 基准 lr ×scratch_lr_factor(本次 1e4)，wd 用默认
    param_groups = [{'params': backbone_parameters_wd, 'lr': args.lr},
                    {'params': no_weight_decay_params, 'lr': args.lr, 'weight_decay': 0.0},
                    {'params': finer_parameters, 'lr': args.lr * args.finer_lr_factor, 'weight_decay': 0.0},
                    {'params': modulation_parameters, 'lr': args.lr * args.modulation_lr_factor, 'weight_decay': 0.0},
                    {'params': scratch_parameters, 'lr': args.lr * args.scratch_lr_factor}]

    return param_groups
