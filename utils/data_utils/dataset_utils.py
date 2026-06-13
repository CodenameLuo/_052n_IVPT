"""Dataset helper utilities (JSON I/O, image loading, bbox cropping)."""

# ======================================
#
# 这个文件是一组数据相关的小工具。本次训练路径上真正用到的只有 pil_loader(读图并转 RGB)，
# 它被上面的 Dataset.__getitem__ 调用。
#
# 其余函数(load_json/save_json、get_dimensions、center_crop_boxes_kps、_get_center_crop_params_)
# 都是“带关键点/检测框的评估”才用的，训练不碰，下面只做概述、保持原样。
#
# ======================================

import json
from typing import List, Optional

import numpy as np
import torchvision
from PIL import Image
from torch import Tensor


# —— 评估/日志用：读写 json —— 训练路径不用
def load_json(path: str):
    """
    Load json file from path and return the data
    :param path: Path to the json file
    :return:
    data: Data in the json file
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(data: dict, path: str):
    """
    Save data to a json file
    :param data: Data to be saved
    :param path: Path to save the data
    :return:
    """
    with open(path, "w") as f:
        json.dump(data, f)


# ======================================

# 【训练路径在用】读图：打开文件 -> 用 PIL 解码 -> 统一转成 RGB 三通道
# 转 RGB 是为了兼容灰度图/带透明通道的图(灰度会复制成三通道)，保证后续张量都是 [3,H,W]
def pil_loader(path):
    """
    Load image from path using PIL
    :param path: Path to the image
    :return:
    img: PIL Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# ============================================================================
# ↓↓↓ 以下函数【不在本次训练路径上】↓↓↓
# 都是“带部件关键点(parts)和检测框(boxes)的评估”才用的几何工具：
#   get_dimensions          : 从 Tensor/ndarray/PIL 三种格式里取出 (高, 宽)
#   center_crop_boxes_kps   : 对图做中心裁剪，并同步把关键点/框坐标平移、越界的标为不可见
#   _get_center_crop_params_: 算中心裁剪的左上角偏移(含目标尺寸大于原图时的居中补边情形)
# 训练时取图只过普通增强、不带关键点，所以这些都不会被调用，这里不逐行展开。
# ============================================================================

def get_dimensions(image: Tensor):
    """
    Get the dimensions of the image
    :param image: Tensor or PIL Image or np.ndarray
    :return:
    h: Height of the image
    w: Width of the image
    """
    if isinstance(image, Tensor):
        _, h, w = image.shape
    elif isinstance(image, np.ndarray):
        h, w, _ = image.shape
    elif isinstance(image, Image.Image):
        w, h = image.size
    else:
        raise ValueError(f"Invalid image type: {type(image)}")
    return h, w


def center_crop_boxes_kps(img: Tensor, output_size: Optional[List[int]] = 448, parts: Optional[Tensor] = None,
                          boxes: Optional[Tensor] = None, num_keypoints: int = 15):
    """
    Calculate the center crop parameters for the bounding boxes and landmarks and update them
    :param img: Image
    :param output_size: Output size of the cropped image
    :param parts: Locations of the landmarks of following format: <part_id> <x> <y> <visible>
    :param boxes: Bounding boxes of the landmarks of following format: <image_id> <x> <y> <width> <height>
    :param num_keypoints: Number of keypoints
    :return:
    cropped_img: Center cropped image
    parts: Updated locations of the landmarks
    boxes: Updated bounding boxes of the landmarks
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = output_size
    else:
        raise ValueError(f"Invalid output size: {output_size}")

    crop_height, crop_width = output_size
    image_height, image_width = get_dimensions(img)
    img = torchvision.transforms.functional.center_crop(img, output_size)

    crop_top, crop_left = _get_center_crop_params_(image_height, image_width, output_size)

    if parts is not None:
        for j in range(num_keypoints):
            # Skip if part is invisible
            if parts[j][-1] == 0:
                continue
            parts[j][1] -= crop_left
            parts[j][2] -= crop_top

            # Skip if part is outside the crop
            if parts[j][1] > crop_width or parts[j][2] > crop_height:
                parts[j][-1] = 0
            if parts[j][1] < 0 or parts[j][2] < 0:
                parts[j][-1] = 0

            parts[j][1] = min(crop_width, parts[j][1])
            parts[j][2] = min(crop_height, parts[j][2])
            parts[j][1] = max(0, parts[j][1])
            parts[j][2] = max(0, parts[j][2])

    if boxes is not None:
        boxes[1] -= crop_left
        boxes[2] -= crop_top
        boxes[1] = max(0, boxes[1])
        boxes[2] = max(0, boxes[2])
        boxes[1] = min(crop_width, boxes[1])
        boxes[2] = min(crop_height, boxes[2])

    return img, parts, boxes


def _get_center_crop_params_(image_height: int, image_width: int, output_size: Optional[List[int]] = 448):
    """
    Get the parameters for center cropping the image
    :param image_height: Height of the image
    :param image_width: Width of the image
    :param output_size: Output size of the cropped image
    :return:
    crop_top: Top coordinate of the cropped image
    crop_left: Left coordinate of the cropped image
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 2:
        output_size = output_size
    else:
        raise ValueError(f"Invalid output size: {output_size}")

    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        crop_top, crop_left = padding_ltrb[1], padding_ltrb[0]
        return crop_top, crop_left

    if crop_width == image_width and crop_height == image_height:
        crop_top = 0
        crop_left = 0
        return crop_top, crop_left

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))

    return crop_top, crop_left
