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
# pil_loader：读图并转 RGB；center_crop_boxes_kps：带关键点/框的中心裁剪(仅 Parts 类用)
from utils.data_utils.dataset_utils import pil_loader, center_crop_boxes_kps
# file_line_count：数文件行数(仅 Parts 类用来数关键点种类数)
from utils.misc_utils import file_line_count


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

    def __init__(self, data_path, split=1, mode='train', transform=None, image_sub_path="images"):
        # 记下根目录、当前划分、增强、图片子目录
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.image_sub_path = image_sub_path
        # 读图函数(打开 -> 转 RGB)
        self.loader = pil_loader
        # === 读入四张标注表(都是空格分隔的 txt，用 pandas 读成 DataFrame) ===
        # <图id, 是否训练集>
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), sep='\s+',
                                 names=['id', 'train'])
        # <图id, 文件名>
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), sep='\s+',
                                  names=['id', 'filename'])
        # <图id, 类别标签>
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), sep='\s+',
                             names=['id', 'label'])
        # <图id, 部件id, x, y, 是否可见>
        image_parts = pd.read_csv(os.path.join(data_path, 'parts', 'part_locs.txt'), sep='\s+',
                                  names=['id', 'part_id', 'x', 'y', 'visible'])
        # 按图id 把“是否训练集 + 文件名 + 标签”拼成一张大表
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        # === 按 mode 选出当前划分的样本 ===
        if mode == 'train':
            # 只留官方训练集(train==1)，再取前 split 比例当训练(本次 split=1 即全部)
            dataset = dataset.loc[dataset['train'] == 1]
            samples_train = np.arange(len(dataset))
            self.train_samples = samples_train[:int(len(samples_train) * split)]
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
        # === 标签重映射成 0 起、连续的整数 ===
        # 原始标签可能不是从 0 开始、还可能有缺口；这里把它们压成 0,1,2,... 连续整数
        # 例：原标签 [1, 2, 5, 10] -> labels_to_index={1:0, 2:1, 5:2, 10:3} -> self.labels 里存 0/1/2/3
        # Handle the case where the labels are not 0-indexed and there are gaps
        labels_to_array = dataset['label'].to_numpy()
        labels_to_index = {label: i for i, label in enumerate(np.unique(labels_to_array))}
        self.labels = np.array([labels_to_index[label] for label in labels_to_array])
        # 反向映射：新下标 -> 原始标签(评估时想报告原始类别号时用)
        self.new_to_orig_label = {i: label for i, label in enumerate(np.unique(labels_to_array))}
        # 部件标注：只留当前划分的图、且只保留“可见(visible==1)”的部件(仅评估时用到)
        image_parts = image_parts.loc[image_parts['id'].isin(self.ids)]
        self.parts = image_parts[image_parts['visible'] == 1]
        # 类别总数(CUB=200)
        self.num_classes = len(np.unique(self.labels))
        # 统计每个类有多少张图，存成 cls_num_list(类平衡采样/加权时会用到，本次默认不用)
        self.per_class_count = defaultdict(int)
        for label in self.labels:
            self.per_class_count[label] += 1
        self.cls_num_list = [self.per_class_count[idx] for idx in range(self.num_classes)]

    # 数据集大小 = 样本数
    def __len__(self):
        return len(self.labels)

    # 按下标取一条样本：返回(增强后的图张量, 标签整数)
    def __getitem__(self, idx):
        # 拼出图片完整路径：根目录 / 图片子目录 / 文件名
        image_path = os.path.join(self.data_path, self.image_sub_path, self.names[idx])
        # 读图(PIL，已转 RGB)
        im = self.loader(image_path)
        # 取标签(已是 0 起连续整数)
        label = self.labels[idx]

        # 过一遍增强流水线(训练=强增强，测试=弱增强) -> 得到 [3, H, W] 张量
        if self.transform:
            im = self.transform(im)

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
