"""Image transform builders for training and evaluation."""

# ======================================
#
# 这个文件负责“数据增强流水线”：定义一组 transform，DataLoader 每取一张图就按顺序过一遍。
# train_net.py 第 4 步调用的 load_transforms(args) 就是这里的总入口。
#
# 本次 augmentations_to_use=cub_original，所以训练走 make_train_transforms、测试走 make_test_transforms；
# build_transform_timm(timm 自动增强)这条分支本次不走，下面只做概述。
# 文件末尾的 inverse_normalize* 系列是“反归一化”，给后面可视化注意力图时把图还原成正常颜色用的(训练本身不用)。
#
# ======================================

import torch
from torchvision import transforms as transforms
from torchvision.transforms import Compose

# timm 自带的 ImageNet 归一化常数(均值/方差)；本仓库的图都按 ImageNet 统计量做标准化
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

# ======================================

# 训练用的增强流水线(cub_original)：强增强，每个 epoch、每张图都重新随机一次
def make_train_transforms(args):
    train_transforms = transforms.Compose(
        [
            # 把短边缩放到 image_size(本次 518)，长边按比例缩放(保持长宽比)
            transforms.Resize(size=args.image_size, antialias=True),
            # 以 hflip 概率(默认 0.5)做水平翻转
            transforms.RandomHorizontalFlip(p=args.hflip),
            # 以 vflip 概率(默认 0.0，相当于不翻)做垂直翻转
            transforms.RandomVerticalFlip(p=args.vflip),
            # 颜色抖动(亮度/对比度/饱和度，用 torchvision 默认幅度)
            transforms.ColorJitter(),
            # 随机仿射：旋转 ±90°、平移 ±20%、缩放 0.8~1.2 倍(空出来的区域默认填黑)
            # 注：这里的 90 / 0.2 / 0.8,1.2 是写死的，不读 args；旋转范围相当大，对细粒度任务算激进
            transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            # 从上面结果里随机裁出 image_size×image_size(本次 518×518)的方图，作为最终输入尺寸
            transforms.RandomCrop(args.image_size),
            # PIL 图 -> [0,1] 的 [C,H,W] 张量
            transforms.ToTensor(),
            # 按 ImageNet 均值/方差做标准化((x-mean)/std)
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]
    )
    return train_transforms

# ======================================

# 测试/评估用的增强流水线：弱增强、无随机性(保证评估结果可复现、不丢判别性区域)
def make_test_transforms(args):
    test_transforms: Compose = transforms.Compose(
        [
            # 短边缩放到 image_size(本次 518)
            transforms.Resize(size=args.image_size, antialias=True),
            # 从中心裁出 image_size×image_size 的方图(没有随机裁剪)
            transforms.CenterCrop(args.image_size),
            # PIL 图 -> [0,1] 张量
            transforms.ToTensor(),
            # ImageNet 标准化
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ]
    )

    return test_transforms

# ======================================

# 另一套增强：用 timm 的 create_transform(支持 AutoAugment/RandAugment/RandomErasing 等)
# 注：仅当 augmentations_to_use='timm' 时才走这里；本次用的是 cub_original，所以这个函数本次不会被调用
def build_transform_timm(args, is_train=True):
    # 图够大(>32)才做 resize；否则当作 CIFAR 这类小图处理
    resize_im = args.image_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    # 选用 ImageNet 默认 还是 Inception 风格的均值/方差
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # 训练：交给 timm 自动拼一套训练增强(含 AutoAugment、随机擦除等)
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            hflip=args.hflip,
            vflip=args.vflip,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        # 小图情形：把第一个 transform 换成带 padding 的随机裁剪
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.image_size, padding=4)
        return transform

    # 测试：手动拼一套确定性的 resize + 中心裁剪
    t = []
    if resize_im:
        # 图 >=384 时直接拉伸成正方形(warping，不裁剪)
        # warping (no cropping) when evaluated at 384 or larger
        if args.image_size >= 384:
            t.append(
                transforms.Resize((args.image_size, args.image_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            print(f"Warping {args.image_size} size input images...")
        else:
            # 图较小时：先按 crop_pct 放大一点，再中心裁剪到目标尺寸(沿用 224/256 的惯例比例)
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.image_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            )
            t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# ======================================
#
# 下面三个是“反归一化”工具：标准化是 (x-mean)/std，反过来就是 x*std+mean。
# torchvision 的 Normalize 只会算 (x-m)/s，所以这里把参数凑成 mean'=-mean/std、std'=1/std，
# 代进公式 (x - (-mean/std)) / (1/std) = x*std + mean，正好实现还原。
# 这几个函数训练本身不用，是后面把注意力图叠到原图上可视化时，把图还原成正常颜色用的。
#
# ======================================

# 反归一化：把标准化后的张量还原回 [0,1] 区间的原图(用于可视化)
def inverse_normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    un_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return un_normalize


# 只做正向标准化(单独拿出来给别处复用)
def normalize_only(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize


# 反归一化 + 再缩放到指定分辨率(可视化时把图统一缩到 256×256 之类)
def inverse_normalize_w_resize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
                               resize_resolution=(256, 256)):
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    resize_unnorm = transforms.Compose([
        transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
        transforms.Resize(size=resize_resolution, antialias=True)])
    return resize_unnorm


# ======================================

# 总入口(train_net.py 第 4 步调用)：根据 augmentations_to_use 选用哪套训练增强，返回(训练增强, 测试增强)
def load_transforms(args):
    # Get the transforms and load the dataset

    # timm：用 timm 自动增强
    if args.augmentations_to_use == 'timm':
        train_transforms = build_transform_timm(args, is_train=True)
    # cub_original：用上面手写的 make_train_transforms【本次走这条】
    elif args.augmentations_to_use == 'cub_original':
        train_transforms = make_train_transforms(args)
    else:
        raise ValueError('Augmentations not supported.')

    # 测试增强始终用确定性的 make_test_transforms
    test_transforms = make_test_transforms(args)

    return train_transforms, test_transforms
