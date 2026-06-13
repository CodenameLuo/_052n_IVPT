"""
Dataset builder for IVPT.

Provides the ``get_dataset`` function to construct training and test
datasets based on command-line arguments.
"""

# ======================================
#
# 这个文件是“数据集工厂”：train_net.py 第 5 步调用 get_dataset，它根据 args.dataset 选用合适的 Dataset 类，
# 分别建好训练集和测试集，再顺手把类别总数 num_cls 抠出来返回。本次 dataset=cub，走鸟类细粒度数据集。
#
# ======================================

# 真正的数据集实现(读 CUB 的几个 txt、按下标返回一张图+标签)
from data_sets.fg_bird_dataset import FineGrainedBirdClassificationDataset


def get_dataset(args, train_transforms, test_transforms):
    # CUB 和 NABirds 共用同一个 Dataset 类(都是鸟类细粒度数据集，目录结构一致)
    if args.dataset == 'cub' or args.dataset == 'nabirds':
        # 训练集：mode='train'，用 train_split 比例的官方训练集(本次 split=1 即全量)，配训练增强
        dataset_train = FineGrainedBirdClassificationDataset(args.data_path, split=args.train_split, mode='train',
                                                             transform=train_transforms,
                                                             image_sub_path=args.image_sub_path_train)
        # 测试集：mode=eval_mode(本次 'test')，配测试增强；不传 split(测试集固定就是官方测试划分)
        dataset_test = FineGrainedBirdClassificationDataset(args.data_path, mode=args.eval_mode,
                                                            transform=test_transforms,
                                                            image_sub_path=args.image_sub_path_test)
        # 类别总数从训练集里取(CUB=200)，后面用来定分类头输出维度和损失
        num_cls = dataset_train.num_classes
    else:
        # 只支持 cub / nabirds，其它名字直接报错
        raise ValueError('Dataset not supported.')
    return dataset_train, dataset_test, num_cls
