"""Fine-grained bird classification datasets (CUB-200, NABirds).

Provides :class:`FineGrainedBirdClassificationDataset` for training/testing
and :class:`FineGrainedBirdClassificationParts` for part-annotation-based
interpretability evaluation.

Reference:
    https://github.com/zxhuang1698/interpretability-by-parts/
"""

# ======================================
#
# 这个文件定义“怎么把 CUB 数据集喂给模型”。本次训练只用到第一个类
# FineGrainedBirdClassificationDataset：__init__ 读几个标注 txt 整理成一张表，
# __getitem__ 按下标返回(增强后的图, 标签)。
#
# 第二个类 FineGrainedBirdClassificationParts 是带“部件关键点/检测框”的版本，
# 只在“可解释性评估”里用，训练流程不碰它(下面有专门的提示横幅)。
#
# CUB 目录里几个关键 txt(空格分隔)：
#   train_test_split.txt : <图id> <是否训练集(1/0)>
#   images.txt           : <图id> <相对文件名>
#   image_class_labels.txt: <图id> <类别标签>
#   parts/part_locs.txt  : <图id> <部件id> <x> <y> <是否可见>
#
# ======================================

import os
from collections import defaultdict
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.utils.data

# ======================================

# pil_loader：读图并转 RGB；center_crop_boxes_kps：带关键点/框的中心裁剪(仅 Parts 类用)
from utils.data_utils.dataset_utils import pil_loader, center_crop_boxes_kps
# file_line_count：数文件行数(仅 Parts 类用来数关键点种类数)
from utils.misc_utils import file_line_count

# ======================================

class FineGrainedBirdClassificationDataset(torch.utils.data.Dataset):
    """
    A general class for fine-grained bird classification datasets. Tested for CUB200-2011 and NABirds.
    Variables
    ----------
        data_path, str: Root directory of the dataset.
        split, int: Percentage of training samples to use for training.
        mode, str: Current data split.
            "train": Training split
            "val": Validation split
            "test": Testing split
        transform, callable: A function/transform that takes in a PIL.Image and transforms it.
        image_sub_path, str: Path to the folder containing the images.
    """

    def __init__(
        self, 
        data_path, 
        split=1, 
        mode='train', 
        transform=None, 
        image_sub_path="images"
    ):
        # 记下根目录、当前划分、增强、图片子目录
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        # 读图函数(打开 -> 转 RGB)
        self.loader = pil_loader

        # === 读入四张标注表(都是空格分隔的 txt，用 pandas 读成 DataFrame) ===
        # <图id, 是否训练集>
        train_test = pd.read_csv(
            os.path.join(data_path, 'train_test_split.txt'), 
            sep='\s+', 
            names=['id', 'train']
        )
        # <图id, 文件名>
        image_names = pd.read_csv(
            os.path.join(data_path, 'images.txt'), 
            sep='\s+', 
            names=['id', 'filename']
        )
        # <图id, 类别标签>
        labels = pd.read_csv(
            os.path.join(data_path, 'image_class_labels.txt'), 
            sep='\s+', 
            names=['id', 'label']
        )
        # <图id, 部件id, x, y, 是否可见>
        image_parts = pd.read_csv(
            os.path.join(data_path, 'parts', 'part_locs.txt'), 
            sep='\s+', 
            names=['id', 'part_id', 'x', 'y', 'visible']
        )

        # 按图id 把“是否训练集 + 文件名 + 标签”拼成一张大表
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        # ======================================
        # 根据 mode，从上面的完整样本表中选出当前 Dataset 对象实际使用的样本
        # mode='train'：从官方训练集里取前 split 比例
        # mode='val'  ：从官方训练集里取剩余的后 (1-split) 比例
        # mode='test' ：直接使用完整的官方测试集
        # ======================================
        if mode == 'train':
            # dataset['train'] == 1：逐行判断是否属于官方训练集，得到一列 True / False 布尔值
            # dataset.loc[...]：只保留结果为 True 的行
            # CUB 完整数据有 11788 张图；筛选后只剩 5994 张官方训练图
            dataset = dataset.loc[dataset['train'] == 1]

            # 为筛选后的官方训练集生成从 0 开始的连续“行位置编号”
            # 注意：这些数字不是图片 id，也不是 pandas 原索引，只表示当前 dataset 中第几行
            # 例：len(dataset)=5994 -> samples_train=array([0, 1, 2, ..., 5993])
            samples_train = np.arange(len(dataset))

            # 根据 split 计算实际训练样本数量，并取前 split 比例的行位置
            # int() 会直接舍弃小数部分
            # 例：split=1   -> int(5994 * 1)=5994，使用全部官方训练图
            # 例：split=0.8 -> int(5994 * 0.8)=4795，使用前 4795 张图
            self.train_samples = samples_train[:int(len(samples_train) * split)]

            # iloc 按“整数位置”选行，用上面的位置编号截出实际训练集
            # 本次 split=1，所以 dataset 仍包含全部 5994 张官方训练图
            # 注意：这里没有随机打乱或按类别分层；split<1 时只是按当前顺序取前一部分样本
            dataset = dataset.iloc[self.train_samples]
        elif mode == 'test':
            # 测试集：官方测试集(train==0)，固定不切分
            dataset = dataset.loc[dataset['train'] == 0]
        elif mode == 'val':
            # 验证集：从官方训练集里取“后 (1-split) 比例”那部分(与 train 互补)
            # 注：若 split=1，这里会切出空集——所以 split=1 时不要用 val 模式
            dataset = dataset.loc[dataset['train'] == 1]
            samples_val = np.arange(len(dataset))
            self.val_samples = samples_val[int(len(samples_val) * split):]
            dataset = dataset.iloc[self.val_samples]


        # training images are labelled 1, test images labelled 0. Add these
        # images to the list of image IDs
        # 当前划分里所有图的 id 和 文件名(后面 __getitem__ 按下标取)
        self.ids = dataset['id'].to_numpy()
        self.names = dataset['filename'].to_numpy()

        # ======================================
        # 把当前划分的原始类别标签，重映射成从 0 开始、没有缺口的连续整数
        # 分类模型通常要求标签能直接作为类别下标使用，所以最终标签应为 0, 1, ..., num_classes-1
        # ======================================

        # 从当前划分的 DataFrame 中取出 label 列，并转成一维 numpy 数组
        # CUB 原始类别标签从 1 开始；本次 train/test 划分各自都包含 200 类
        # 例：train 模式下 labels_to_array.shape=(5994,)，内容类似 [1, 1, ..., 200, 200]
        # Handle the case where the labels are not 0-indexed and there are gaps
        labels_to_array = dataset['label'].to_numpy()

        # np.unique：取出当前划分中不重复的原始标签，并按从小到大排序
        # enumerate：为排序后的每个原始标签分配一个从 0 开始的新类别下标
        # 例：原始标签集合 [1, 2, 5, 10] -> labels_to_index={1:0, 2:1, 5:2, 10:3}
        # CUB 当前划分：{1:0, 2:1, ..., 200:199}
        labels_to_index = {
            label: i for i, label in enumerate(np.unique(labels_to_array))
        }

        # 逐个查表，把每张图的原始标签转换成连续的新标签；结果顺序与 dataset 的图片行顺序完全一致
        # 例：labels_to_array=[1, 1, 5, 10] -> self.labels=array([0, 0, 2, 3])
        # self.labels 后面由 __len__、__getitem__ 和类别计数逻辑直接使用
        self.labels = np.array(
            [labels_to_index[label] for label in labels_to_array]
        )

        # 保存反向映射：新标签下标 -> 原始标签
        # 例：new_to_orig_label={0:1, 1:2, 2:5, 3:10}，需要恢复原始类别编号时可以查表
        # 注意：标签映射是在当前划分内独立建立的；若不同划分缺少不同类别，映射可能不一致
        # 当前 CUB train/test 各自都有完整 200 类，所以二者映射一致，都是 0~199 -> 1~200
        self.new_to_orig_label = {
            i: label for i, label in enumerate(np.unique(labels_to_array))
        }

        # ======================================
        # 整理当前划分对应的“可见部件”标注，仅供后续部件可解释性评估使用
        # ======================================

        # image_parts 是完整 part_locs.txt 表，每行是一条部件标注：<图片id, 部件id, x, y, visible>
        # image_parts['id'].isin(self.ids)：逐行判断该部件所属图片是否位于当前 train/val/test 划分
        # .loc[...]：只保留当前划分图片的部件记录
        image_parts = image_parts.loc[image_parts['id'].isin(self.ids)]

        # 再只保留 visible==1 的部件；被遮挡或未标出的部件不参与可见部件评估
        # self.parts 仍是 pandas DataFrame，列为 id / part_id / x / y / visible
        # CUB 完整 part_locs.txt 有 176820 条记录，其中 141407 条 visible==1
        self.parts = image_parts[image_parts['visible'] == 1]

        # ======================================
        # 统计当前划分的类别总数，以及每个类别包含的图片数量
        # ======================================

        # self.labels 已经是 0 起连续标签；取唯一值个数即可得到当前划分的类别数
        # 当前 CUB train/test 各自都包含 200 类，所以 self.num_classes=200
        self.num_classes = len(np.unique(self.labels))

        # defaultdict(int) 中不存在的键默认值为 0，适合逐样本累加类别数量
        # 例：首次执行 self.per_class_count[3] += 1 时，会从默认值 0 加到 1
        self.per_class_count = defaultdict(int)

        # 遍历当前划分每张图的新标签，统计每个类别有多少张图
        # 结果形式类似：{0: 30, 1: 30, ..., 199: 30}；实际每类数量可能不同
        for label in self.labels:
            self.per_class_count[label] += 1

        # 按类别下标 0, 1, ..., num_classes-1 的固定顺序，把字典整理成列表
        # 例：per_class_count={0:30, 1:28, 2:31} -> cls_num_list=[30, 28, 31]
        # 该列表可供类别平衡采样或损失加权使用；本次默认训练流程不使用它
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    # 数据集大小 = 样本数
    def __len__(self):
        return len(self.labels)

    # ======================================
    # 按位置下标读取一条样本；DataLoader 会反复调用该函数，再把多条样本拼成一个 batch
    # 返回值为 (图片, 标签)：配置 transform 时图片是 [3,H,W] Tensor，否则仍是 RGB PIL.Image
    # ======================================
    def __getitem__(self, idx):
        # idx 表示当前划分内部的“样本位置”，不是 CUB 图片 id
        # 例：idx=0 表示取 self.names[0] 和 self.labels[0]；DataLoader shuffle 只会改变传入 idx 的顺序

        # self.names[idx] 是图片相对于图片目录的路径，例如：
        # 001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg
        # os.path.join 把“数据集根目录 / 图片子目录 / 相对文件名”拼成可直接读取的完整路径
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])

        # 调用 pil_loader 打开图片并统一转成 RGB 三通道 PIL.Image
        # 转 RGB 可兼容灰度图或带透明通道的图片，并保证后续 ToTensor 得到 [3,H,W]
        im = self.loader(image_path)

        # 从与 self.names 相同位置取出该图片的新标签
        # 标签已经在 __init__ 中重映射为 0, 1, ..., num_classes-1；单样本阶段通常是 numpy 整数
        label = self.labels[idx]

        # 若构造 Dataset 时传入了 transform，就对 PIL 图片执行对应的数据变换流水线
        # train 模式使用随机强增强；test/val 模式使用确定性的弱增强
        # 本次 image_size=518，流水线最终会执行 ToTensor + Normalize，得到 [3,518,518] Tensor
        # 若 self.transform=None，则跳过该分支，im 保持为 RGB PIL.Image
        if self.transform:
            im = self.transform(im)

        # 返回一条样本；DataLoader 默认 collate 会把多条图片堆成 [B,3,H,W] Tensor，
        # 并把多个标签整理成形状 [B] 的整数 Tensor，再交给训练/评估循环
        return im, label

    # —— 仅评估用：返回第 idx 张图里所有“可见”的部件 id —— 训练流程不调用
    def get_visible_parts(self, idx):
        """
        Returns all parts that are visible in the current image
        Parameters
        ----------
        idx: int
            The index for which to retrieve the visible parts
        """
        dataset_id = self.ids[idx]
        parts = self.parts[self.parts['id'] == dataset_id].loc[:, ["part_id"]].to_numpy()
        return parts


# ============================================================================
# ↓↓↓ 以下 FineGrainedBirdClassificationParts 类【不在本次训练路径上】↓↓↓
# 它是“部件发现/可解释性评估”专用的数据集：除了图和标签，还返回 15 个关键点标注和检测框，
# 并在评估时按中心裁剪同步修正关键点/框坐标。bash scripts/run_train.sh 的训练流程完全不会用到它，
# 故这里不逐行展开注释(保持原样)，等讲到评估步骤时再细看。
# ============================================================================
class FineGrainedBirdClassificationParts(torch.utils.data.Dataset):
    """
    Class for evaluating part detection/discovery on CUB200-2011 dataset. Also tested on NABirds.
    Adapted from: https://github.com/zxhuang1698/interpretability-by-parts/
    Variables
    ----------
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _train_data, list of str: List of paths to the training images.
        _train_labels, np.array: List of labels for the training images.
        _train_parts, torch.FloatTensor: List of part annotations for the training images.
        _train_boxes, torch.FloatTensor: List of bounding box annotations for the training images.
        _test_data, list of str: List of paths to the testing images.
        _test_labels, np.array: List of labels for the testing images.
        _test_parts, torch.FloatTensor: List of part annotations for the testing images.
        _test_boxes, torch.FloatTensor: List of bounding box annotations for the testing images.
    """

    def __init__(self, root, train=True, transform=None, resize=448, center_crop=False, image_sub_path="images"):
        """
        Load the dataset.
        Args
        ----------
        root: str
            Root directory of the dataset.
        train: bool
            train/test data split.
        transform: callable
            A function/transform that takes in a PIL.Image and transforms it.
        resize: int
            Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.
        """
        self._root = root
        self._train = train
        self._transform = transform
        self.loader = pil_loader
        self.newsize = resize
        self.center_crop = center_crop
        self.image_sub_path = image_sub_path
        # 15 key points provided by CUB
        self.num_kps = file_line_count(os.path.join(root, 'parts', 'parts.txt'))

        if not os.path.isdir(root):
            os.mkdir(root)

        self.per_class_count = defaultdict(int)
        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            self.num_classes = len(np.unique(self._train_labels))
            for label in self._train_labels:
                self.per_class_count[label] += 1
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            self.num_classes = len(np.unique(self._test_labels))
            for label in self._test_labels:
                self.per_class_count[label] += 1

        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    def __getitem__(self, index):
        """
        Retrieve data samples.
        Args
        ----------
        index: int
            Index of the sample.
        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """
        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            parts = self._train_parts[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            parts = self._test_parts[index]
            boxes = self._test_boxes[index]

        # load the image
        image = self.loader(image_path)
        image = np.array(image)

        # calculate the resize factor
        # if original image height is larger than width, the real resize factor is based on width
        if image.shape[0] >= image.shape[1]:
            factor = self.newsize / image.shape[1]
        else:
            factor = self.newsize / image.shape[0]

        # transform 15 landmarks according to the new shape
        # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
        for j in range(self.num_kps):

            # step in only when the current landmark exists
            if abs(parts[j][-1]) > 1e-5:
                # calculate the new location according to the new shape
                parts[j][-3] = parts[j][-3] * factor
                parts[j][-2] = parts[j][-2] * factor

        # rescale the annotation of bounding boxes
        # the annotation format of the bounding boxes are <image_id, col of top-left corner, row of top-left corner, width, height>
        boxes[1:] *= factor

        # convert the image into a PIL image for transformation
        image = PIL.Image.fromarray(image)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)

        # center crop
        if self.center_crop:
            image, parts, boxes = center_crop_boxes_kps(image, self.newsize, parts, boxes, self.num_kps)
        return image, target, parts, boxes

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""

        # load the list into numpy arrays
        image_path = os.path.join(self._root, self.image_sub_path)
        train_test = pd.read_csv(os.path.join(self._root, 'train_test_split.txt'), sep='\s+',
                                 names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(self._root, 'images.txt'), sep='\s+',
                                  names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(self._root, 'image_class_labels.txt'), sep='\s+',
                             names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(self._root, 'parts', 'part_locs.txt'), sep='\s+',
                                  names=['id', 'part_id', 'x', 'y', 'visible'])
        image_boxes = pd.read_csv(os.path.join(self._root, 'bounding_boxes.txt'), sep='\s+',
                                  names=['id', 'x', 'y', 'width', 'height'])
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')
        dataset = dataset.merge(image_boxes, on='id')

        labels_to_array = dataset['label'].to_numpy()
        labels_to_index = {label: i for i, label in enumerate(np.unique(labels_to_array))}
        dataset['label'] = dataset['label'].apply(lambda x: labels_to_index[x])
        # Handle string-based image ids
        image_id_to_index = {image_id: i for i, image_id in enumerate(dataset['id'].to_numpy())}
        image_parts['id'] = image_parts['id'].apply(lambda x: image_id_to_index[x])
        dataset['id'] = dataset['id'].apply(lambda x: image_id_to_index[x])

        # return according to different splits
        if train:
            dataset_train = dataset.loc[dataset['train'] == 1]
            image_parts_train = image_parts.loc[image_parts['id'].isin(dataset_train['id'])]
            data = [os.path.join(image_path, name) for name in dataset_train['filename']]
            labels = dataset_train['label'].to_numpy()
            boxes = dataset_train[['id', 'x', 'y', 'width', 'height']].to_numpy()
            boxes = torch.from_numpy(boxes).float()
            parts = image_parts_train.loc[:, ["part_id", "x", "y", "visible"]].to_numpy().reshape(
                len(dataset_train), self.num_kps, 4)
            parts = torch.from_numpy(parts).float()
        else:
            dataset_test = dataset.loc[dataset['train'] == 0]
            image_parts_test = image_parts.loc[image_parts['id'].isin(dataset_test['id'])]
            data = [os.path.join(image_path, name) for name in dataset_test['filename']]
            labels = dataset_test['label'].to_numpy()
            boxes = dataset_test[['id', 'x', 'y', 'width', 'height']].to_numpy()
            boxes = torch.from_numpy(boxes).float()

            parts = image_parts_test.loc[:, ["part_id", "x", "y", "visible"]].to_numpy().reshape(len(dataset_test),
                                                                                                 self.num_kps, 4)
            parts = torch.from_numpy(parts).float()
        return data, labels, parts, boxes


if __name__ == '__main__':
    pass
