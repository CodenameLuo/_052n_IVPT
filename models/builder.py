"""
Model builder utilities for IVPT.

Provides functions to load backbone architectures (ResNet, ConvNeXt, ViT),
initialize the IVPT model, and load pre-trained checkpoints.
"""

# ======================================
#
# 这个文件是“模型工厂”，train_net.py 第 6 步调用的 load_model_ivpt 就在这里。它分两步：
#   1) load_model_arch：从 timm 把主干网络(本次=DINOv2 ViT-Base)建出来、加载预训练权重；
#   2) init_ivpt_model：把这个主干用 IndividualLandmarkViT 包一层，在 ViT 里加上“部件发现”结构。
# 包完返回的就是最终要训练的 IVPT 模型。
#
# 注：本文件后半段 ivpt_vit / ivptnet_vit / ivptnet_resnet101 是“从网上拉训练好的权重做推理”用的加载器，
#     不在训练路径上(训练不会调它们)，下面有横幅标注。
#
# ======================================

import copy
import os
from pathlib import Path

import torch
# timm 的建模入口(按模型名造网络、可选加载预训练权重)
from timm.models import create_model
# torchvision 的建模入口(仅 ResNet 用 torchvision 实现时才用)
from torchvision.models import get_model

# ======================================

# IndividualLandmarkViT：IVPT 的核心模型类；ivpt_vit_bb/ivptnet_vit_bb 是下面 hub 加载器用的(且为坏死代码)
from models.individual_landmark_vit import IndividualLandmarkViT, ivpt_vit_bb, ivptnet_vit_bb
from utils.training_utils.engine_utils import load_state_dict_ivpt

# ======================================

# 第 1 步：把主干网络建出来。按 model_arch 名字分派到 ResNet / ConvNeXt / ViT 三类；本次走 ViT('patch')分支
def load_model_arch(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """

    # —— ResNet 分支(本次不走)：先从模型名里抠出层数，拼出对应的 timm 权重标签 ——
    if 'resnet' in args.model_arch:
        num_layers_split = [int(s) for s in args.model_arch if s.isdigit()]
        num_layers = int(''.join(map(str, num_layers_split)))
        if num_layers >= 100:
            timm_model_arch = args.model_arch + ".a1h_in1k"
        else:
            timm_model_arch = args.model_arch + ".a1_in1k"

    # ResNet + torchvision 实现(本次不走)
    if "resnet" in args.model_arch and args.use_torchvision_resnet_model:
        weights = "DEFAULT" if args.pretrained_start_weights else None
        base_model = get_model(args.model_arch, weights=weights)

    # ResNet + timm 实现(本次不走)；注意：只有非 eval_only(即训练)时才加 drop_path
    elif "resnet" in args.model_arch and not args.use_torchvision_resnet_model:
        if args.eval_only:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                timm_model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )

    # —— ConvNeXt 分支(本次不走) ——
    elif "convnext" in args.model_arch:
        if args.eval_only:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )
        else:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                num_classes=num_cls,
                output_stride=args.output_stride,
            )

    # —— ViT 分支【本次走这条】：模型名里含 'patch'(如 vit_base_patch14_...) ——
    elif "patch" in args.model_arch:
        # eval 时不加 drop_path
        if args.eval_only:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                img_size=args.image_size,
            )
        # 训练时加 drop_path(本次 drop_path=0，等于没加)；img_size=518 让 timm 重建匹配的位置编码
        else:
            base_model = create_model(
                args.model_arch,
                pretrained=args.pretrained_start_weights,
                drop_path_rate=args.drop_path,
                img_size=args.image_size,
            )

        # 取出 patch 大小(本次 14)，检查图像尺寸能被整除(518/14=37，整除 OK)，否则切不出整数个 patch
        vit_patch_size = base_model.patch_embed.proj.kernel_size[0]
        if (args.image_size % vit_patch_size) != 0:
            raise ValueError(f"Image size {args.image_size} must be divisible by patch size {vit_patch_size}")

    else:
        raise ValueError('Model not supported.')

    return base_model


# ======================================

# 第 2 步：把主干包成 IVPT 模型，把所有 IVPT 专属超参(调制类型/gumbel/原型数 n_pro 等)透传进去
def init_ivpt_model(base_model, args, num_cls):
    """
    Function to initialize the model
    :param base_model: Base model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    # Initialize the network
    # 本次是 ViT，用 IndividualLandmarkViT 包(把部件发现结构插进 ViT)
    if 'patch' in args.model_arch:
        model = IndividualLandmarkViT(
            base_model, 
            num_classes=num_cls, 
            part_dropout=args.part_dropout, 
            modulation_type=args.modulation_type, 
            gumbel_softmax=args.gumbel_softmax, 
            gumbel_softmax_temperature=args.gumbel_softmax_temperature, 
            gumbel_softmax_hard=args.gumbel_softmax_hard, 
            classifier_type=args.classifier_type, 
            noise_variance=args.noise_variance, 
            n_pro=args.n_pro
        )

    else:
        raise ValueError('Model not supported.')

    return model


# ======================================

# 训练入口(train_net.py 第 6 步调用)：建主干 -> 包成 IVPT 模型 -> 返回
def load_model_ivpt(args, num_cls):
    """
    Function to load the model
    :param args: Arguments from the command line
    :param num_cls: Number of classes in the dataset
    :return:
    """
    base_model = load_model_arch(args, num_cls)
    model = init_ivpt_model(base_model, args, num_cls)

    return model


# ============================================================================
# ↓↓↓ 以下 ivpt_vit / ivptnet_vit / ivptnet_resnet101 都【不在本次训练路径上】↓↓↓
# 它们是“从 torch.hub URL 下载训练好的检查点、组装好模型做推理/复现”的加载器，训练时不会被调用。
# 而且它们依赖的 ivpt_vit_bb / ivptnet_vit_bb 是坏死代码(传了 IndividualLandmarkViT 并不接收的
# num_landmarks / modulation_orth 参数)，ivptnet_resnet101 还引用了本文件没 import 的函数——真调用会报错。
# 这里保持原样、不逐行展开。
# ============================================================================

def ivpt_vit(pretrained=True, model_dataset="cub", k=8, model_url="", img_size=224, num_cls=200):
    """
    Function to load the PDiscoFormer model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param img_size: Image size
    :param num_cls: Number of classes in the dataset
    :return: PDiscoFormer model with ViT backbone
    """
    model = ivpt_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "ivpt_checkpoints", f"ivpt_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_ivpt(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def ivptnet_vit(pretrained=True, model_dataset="nabirds", k=8, model_url="", img_size=224, num_cls=555):
    """
    Function to load the PDiscoNet model with ViT backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param img_size: Image size
    :param num_cls: Number of classes in the dataset
    :return: PDiscoNet model with ViT backbone
    """
    model = ivptnet_vit_bb("vit_base_patch14_reg4_dinov2.lvd142m", num_cls=num_cls, k=k, img_size=img_size)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "ivpt_checkpoints", f"ivptnet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_ivpt(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model


def ivptnet_resnet101(pretrained=True, model_dataset="nabirds", k=8, model_url="", num_cls=555):
    """
    Function to load the PDiscoNet model with ResNet-101 backbone
    :param pretrained: Boolean flag to load the pretrained weights
    :param model_dataset: Dataset for which the model is trained
    :param k: Number of unsupervised landmarks the model is trained on
    :param model_url: URL to load the model weights from
    :param num_cls: Number of classes in the dataset
    :return: PDiscoNet model with ResNet-101 backbone
    """
    model = ivptnet_resnet_torchvision_bb("resnet101", num_cls=num_cls, k=k)
    if pretrained:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, "ivpt_checkpoints", f"ivptnet_{model_dataset}")

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        url_path = model_url + str(k) + "_parts_snapshot_best.pt"
        snapshot_data = torch.hub.load_state_dict_from_url(url_path, model_dir=model_dir, map_location='cpu')
        if 'model_state' in snapshot_data:
            _, state_dict = load_state_dict_ivpt(snapshot_data)
        else:
            state_dict = copy.deepcopy(snapshot_data)
        model.load_state_dict(state_dict, strict=True)
    return model
