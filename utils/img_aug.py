"""Data augmentation pipeline for cropped CUB images.

Uses the ``Augmentor`` library to apply rotation, skew, and shear
transforms to the training split of the cropped dataset.

Usage::

    python utils/img_aug.py --data_path datasets/cub200_cropped
"""

import argparse
import os

import Augmentor

# Augmentor 的 rotate 在极端宽扁图（横宽比约 >=3.8:1）抽到 13~15 度时，
# 旋转后裁"最大内接矩形"的公式会算出上下颠倒的裁剪框，PIL 抛 ValueError。
# 角度在 perform_operation 内部随机抽取，捕获后重试即换角度；
# 重试耗尽则放弃本次旋转返回原图（实际几乎不会走到）。
_orig_rotate = Augmentor.Operations.RotateRange.perform_operation

def _safe_rotate(self, images):
    for _ in range(20):
        try:
            return _orig_rotate(self, images)
        except ValueError:
            continue
    return images

Augmentor.Operations.RotateRange.perform_operation = _safe_rotate


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
args = parser.parse_args()

# datasets_root_dir = 'datasets/cub200_cropped/'
datasets_root_dir = args.data_path
dir = os.path.join(datasets_root_dir, 'train_cropped/')
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented/')

makedir(target_dir)
folders = [os.path.abspath(os.path.join(dir, folder)) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.abspath(os.path.join(target_dir, folder)) for folder in next(os.walk(dir))[1]]
# folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
# target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(10):
        p.process()
    del p
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for i in range(10):
    #    p.process()
    #del p