"""
Loss function builder for IVPT.

Provides utilities to construct classification losses (including mixup,
label smoothing, and imbalanced top-k) and to assemble loss hyperparameters
and equivariance affine-transform parameters.
"""

# ======================================
#
# 这个文件负责“损失这一摊”的装配，对应 train_net.py 第 8、9 步两次调用：
#   load_classification_loss：返回分类损失(训练用 + 评估用)和 mixup 函数；本次走最普通的交叉熵、mixup_fn=None。
#   load_loss_hyper_params  ：把各项辅助损失的“权重”打包成一个字典，外加“等变损失”要用的随机仿射变换范围。
# 注意：这里只是“准备好系数和参数”，真正按这些系数把 8 个损失加起来是在训练器的 _run_batch 里(Part 7)。
#
# ======================================

import torch
# 下面三个是“别的分类损失分支”用的(mixup / 标签平滑 / 不平衡 TopK)，本次都不走
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from pytopk import ImbalNoisedTopK


# ======================================

# 选分类损失：按命令行开关在 mixup / TopK / 标签平滑 / 普通交叉熵 之间挑一个；评估恒用普通交叉熵
def load_classification_loss(args, dataset_train, num_cls):
    """
    Load the loss function for classification
    :param args: Arguments from the argument parser
    :param dataset_train: Training dataset
    :param num_cls: Number of classes in the dataset
    :return:
    loss_fn: List of loss functions for training and evaluation
    """
    # Mixup/Cutmix
    # mixup：把两张图按比例混合、标签也混合的增强；本次未开(turn_on_mixup_or_cutmix=False)
    mixup_fn = None
    mixup_active = args.turn_on_mixup_or_cutmix
    if mixup_active:
        # 提醒：mixup 会和等变损失冲突(本次没开，所以这段不执行)
        print("Mixup is activated! Please note that this may not work with the equivariance loss")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=num_cls)

    # 分支一：不平衡 TopK 损失(本次未开 use_imbalanced_noised_topk)
    if args.use_imbalanced_noised_topk:
        loss_fn_train = ImbalNoisedTopK(k=args.topk_k, epsilon=args.topk_epsilon, max_m=args.max_m,
                                        cls_num_list=dataset_train.cls_num_list, scale=args.topk_scale,
                                        n_sample=args.topk_n_sample)
        print(
            "Using Imbalanced Noised TopK loss, please note that label smoothing and mixup are not implemented in this case.")
        mixup_fn = None
    else:
        # Define loss and optimizer
        # 分支二：开了 mixup -> 用软标签交叉熵(本次不走)
        if mixup_fn is not None:
            # smoothing is handled with mix-up label transform
            loss_fn_train = SoftTargetCrossEntropy()
        # 分支三：标签平滑 > 0 -> 标签平滑交叉熵(本次 smoothing=0，不走)
        elif args.smoothing > 0.:
            loss_fn_train = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        # 分支四【本次走这条】：最普通的交叉熵
        else:
            loss_fn_train = torch.nn.CrossEntropyLoss()

    # 评估始终用普通交叉熵
    loss_fn_eval = torch.nn.CrossEntropyLoss()
    # 打包成 [训练损失, 评估损失] 返回(训练器里按这个顺序取用)
    loss_fn = [loss_fn_train, loss_fn_eval]
    return loss_fn, mixup_fn


# ======================================

# 打包：各辅助损失的权重(字典) + 等变损失用的随机仿射变换范围(字典)
def load_loss_hyper_params(args):
    """
    Load the hyperparameters for the loss functions and affine transform parameters for equivariance
    :param args: Arguments from the argument parser
    :return:
    loss_hyperparams: Dictionary of loss hyperparameters
    eq_affine_transform_params: Dictionary of affine transform parameters for equivariance
    """
    # 各损失的权重系数(本次除个别外大多=1)；注：l_conc(concentration)装进来了但训练器其实从不读它
    loss_hyperparams = {'l_class_att': args.classification_loss, 'l_presence': args.presence_loss,
                        'l_presence_beta': args.presence_loss_beta, 'l_presence_type': args.presence_loss_type,
                        'l_equiv': args.equivariance_loss, 'l_conc': args.concentration_loss,
                        'l_orth': args.orthogonality_loss_landmarks, 'l_tv': args.total_variation_loss,
                        'l_enforced_presence': args.enforced_presence_loss, 'l_pixel_wise_entropy': args.pixel_wise_entropy_loss,
                        'l_enforced_presence_loss_type': args.enforced_presence_loss_type}

    # === 等变损失用的随机仿射变换范围 ===
    # 注意：这套变换是“等变损失”专用的(把图变换后再过模型，看部件图是否跟着同步变换)，
    #       和前面数据增强里的 RandomAffine 是两码事、互不影响。
    # Affine transform parameters for equivariance
    # 旋转范围 [-90, 90] 度
    degrees = [-args.degrees, args.degrees]
    # 平移比例 [x, y] = [0.11, 0.11]
    translate = [args.translate_x, args.translate_y]
    # 缩放范围 [下界, 上界] = [0.8, 1.4]
    scale = [args.scale_l, args.scale_u]
    # 错切 x / y
    shear_x = args.shear_x
    shear_y = args.shear_y
    shear = [shear_x, shear_y]
    # 两个错切都为 0 时直接置 None(本次 shear=None，不做错切)
    if shear_x == 0.0 and shear_y == 0.0:
        shear = None

    eq_affine_transform_params = {'degrees': degrees, 'translate': translate, 'scale_ranges': scale, 'shear': shear}

    return loss_hyperparams, eq_affine_transform_params
