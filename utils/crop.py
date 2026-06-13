"""Crop CUB-200-2011 images by bounding box and split into train/test folders.

Run as a standalone script — produces a ``cub200_cropped/`` directory with
``train_cropped/`` and ``test_cropped/`` sub-directories.
"""

# ======================================
# 这个脚本做一件事：把 CUB 每张原图按官方标注的鸟身边界框裁剪出来，
# 再按官方的 train/test 划分，整理成一个新的数据集目录
#
# 输入：datasets/CUB_200_2011/（解压后的原始 CUB 数据集，200 类共 11788 张图）
# 输出：datasets/cub200_cropped/
#       ├── train_cropped/<类别文件夹>/<图片>    训练集裁剪图（5994 张）
#       ├── test_cropped/<类别文件夹>/<图片>     测试集裁剪图（5794 张）
#       └── 三个标注 txt（从原数据集原样复制，让新目录自成一个完整数据集）
#
# 流程分四段，前三段各建一个字典，第四段干活：
# 1. 读 images.txt           -> id 对应哪张图（相对路径）
# 2. 读 bounding_boxes.txt   -> id 对应哪个边界框
# 3. 读 train_test_split.txt -> id 属于训练集还是测试集
# 4. 主循环逐图：读原图 -> 按框切片 -> 按划分存到对应文件夹
#
# 运行方式：在仓库根目录下执行 python utils/crop.py
# 注意：下面的路径全是写死的相对路径，必须在仓库根目录下运行，
#       换了目录就会找不到 datasets/CUB_200_2011/
#
# 这一步在 README 里标为可选：训练快速开始用的是未裁剪的原始 CUB；
# 裁剪出的 cub200_cropped/ 之后被两处使用：
# utils/img_aug.py 在它上面做离线增强，scripts/run_eval.sh 的可解释性评估用 --data_path 指向它
# ======================================

# 拼路径、建输出文件夹
import os
# 复制标注 txt 文件
import shutil
# OpenCV：读图、写图（裁剪本身只是 numpy 切片）
import cv2
# 进度条，包住主循环显示处理进度
from tqdm import tqdm

# ======================================
# 路径配置：输入在哪、输出到哪、要读哪三个标注文件
# ======================================

# 原始数据集根目录（输入）
data_root = 'datasets/CUB_200_2011/'
# 裁剪后数据集根目录（输出，主循环里逐级创建）
out_dir = 'datasets/cub200_cropped/'

# CUB 官方的三个标注文件，格式统一：每行 = 图片id + 空格 + 标注内容
# 三个文件都是 11788 行，同一张图在三个文件里共用同一个 id，id 就是关联键

# id -> 图片相对路径（类别文件夹/文件名）
# 例：第一行是 "1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg"
img_txt = os.path.join(data_root, 'images.txt')

# id -> 鸟身边界框，四个数依次是 左上角x 左上角y 宽 高
# 例：第一行是 "1 60.0 27.0 325.0 304.0"
bbox_txt = os.path.join(data_root, 'bounding_boxes.txt')

# id -> 训练/测试划分，1 是训练集，0 是测试集
# 例：第一行是 "1 0"，表示 id=1 这张图属于测试集
train_txt = os.path.join(data_root, 'train_test_split.txt')

# ======================================
# 第 1 段：解析 images.txt，建立 id -> (类别文件夹, 文件名) 的字典
# ======================================

# Get the image path of each image id
id_to_path = {}

# readlines 一次性把 11788 行全读进列表，每行末尾都带着换行符 '\n'
with open(img_txt, 'r') as f:
    img_lines = f.readlines()

for img_line in img_lines:
    # 按空格切开：[0] 是图片 id，[1] 是相对路径；[:-1] 切掉行末的 '\n'
    # 例："1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg\n"
    #     -> img_id = 1，img_path = "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg"
    # 
    # img_id, img_path = int(img_line.split(' ')[0]), img_line.split(' ')[1][:-1]
    img_id   = int(img_line.split(' ')[0])
    img_path = img_line.split(' ')[1][:-1]

    # 再按 '/' 把相对路径切成两截：类别文件夹 和 文件名
    # 例：img_folder = "001.Black_footed_Albatross"，img_name = "Black_Footed_Albatross_0046_18.jpg"
    # 
    # img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    img_folder = img_path.split('/')[0]
    img_name   = img_path.split('/')[1]

    # 拆成两截存，是因为主循环里要分别用：
    # 类别文件夹 用来在输出目录下建同名子文件夹，文件名 用来命名输出图片
    id_to_path[img_id] = (img_folder, img_name)


# ======================================
# 第 2 段：解析 bounding_boxes.txt，建立 id -> (x1, y1, x2, y2) 的字典
# 原始标注是"左上角点坐标(x1, y1)，宽(x2)，高(y2)"，这里顺手换算成"左上角，右下角"，后面切片直接用
# ======================================

# Get the bounding box of each image id
id_to_bbox = {}

with open(bbox_txt, 'r') as f:
    bbox_lines = f.readlines()

for bbox_line in bbox_lines:
    # 按空格切成 5 截：id、x、y、宽、高
    # 例："1 60.0 27.0 325.0 304.0\n" -> cts = ['1', '60.0', '27.0', '325.0', '304.0\n']
    # cts : contents (内容)
    cts = bbox_line.split(' ')

    # 标注里的坐标是 "60.0" 这种浮点写法，但切片下标必须是整数
    # split('.')[0] 取小数点前面那截，相当于舍掉小数：'60.0' -> '60'，再 int 成 60
    # 这个写法还顺手解决了行末换行符：'304.0\n' 按 '.' 切开后取 [0] 得 '304'，'\n' 留在后半截里被丢掉
    # 
    # img_id, bbox_x, bbox_y, bbox_width, bbox_height = int(cts[0]), int(cts[1].split('.')[0]), int(cts[2].split('.')[0]), int(cts[3].split('.')[0]), int(cts[4].split('.')[0])
    img_id      = int(cts[0])
    bbox_x      = int(cts[1].split('.')[0])
    bbox_y      = int(cts[2].split('.')[0])
    bbox_width  = int(cts[3].split('.')[0])
    bbox_height = int(cts[4].split('.')[0])

    # 左上角 (x, y) 加上宽高，得到右下角 (x2, y2)
    # 例：左上角 (60, 27)，宽高 (325, 304) -> 右下角 (385, 331)
    # 
    # bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
    bbox_x2 = bbox_x + bbox_width
    bbox_y2 = bbox_y + bbox_height

    id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

# ======================================
# 第 3 段：解析 train_test_split.txt，建立 id -> 是否训练集 的字典
# 这份划分是 CUB 官方固定写死的，不是随机切的，所有用 CUB 的工作都按它对齐
# ======================================

# Get the train/test (1/0) label of each image id
id_to_train = {}

with open(train_txt, 'r') as f:
    train_lines = f.readlines()

for train_line in train_lines:
    # 和第 1 段同一个套路：[0] 是 id，[1] 是划分标记，[:-1] 切掉 '\n'
    # 例："1 0\n" -> img_id = 1，is_train = 0（测试集）
    #
    # img_id, is_train = int(train_line.split(' ')[0]), int(train_line.split(' ')[1][:-1])
    img_id   = int(train_line.split(' ')[0])
    is_train = int(train_line.split(' ')[1][:-1])

    id_to_train[img_id] = is_train

# 到这里三个字典都建好了：11788 个 id，每个 id 都能查到 路径、边界框、划分
# 按这份官方划分：训练集 5994 张，测试集 5794 张

# ======================================
# 第 4 段：主循环，逐张图完成 读原图 -> 按框切片 -> 存到对应文件夹
# ======================================

# 遍历全部 11788 个 id，tqdm 包一层显示进度条
for img_id in tqdm(id_to_path.keys()):
    # 查字典拿到这张图的 (类别文件夹, 文件名)
    root_path = id_to_path[img_id]

    # 拼出原图完整路径：原图都在 data_root 的 images/ 子目录下
    # 例：datasets/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
    pre_path = os.path.join(data_root, 'images', root_path[0], root_path[1])

    # 查划分，决定这张图归 train_cropped 还是 test_cropped
    is_train = id_to_train[img_id]

    # Save training images to 'train_cropped', save test images to 'test_cropped'
    # 
    # folder_name = 'train_cropped' if is_train == 1 else 'test_cropped'
    if is_train == 1:
        folder_name = 'train_cropped'
    else:
        folder_name = 'test_cropped'

    # 输出路径保持"类别文件夹/文件名"的原结构，只是根目录换成 cub200_cropped/<划分>/
    # 例：id=1 是测试集 -> datasets/cub200_cropped/test_cropped/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
    save_dir = os.path.join(out_dir, folder_name, root_path[0])
    out_path = os.path.join(save_dir, root_path[1])

    # 每个类别文件夹第一次遇到时创建，makedirs 会把中间缺的各级目录一并建出来
    # 其实 exist_ok=True 已经允许目录存在，外面这层 if 是冗余的，但无害
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir, exist_ok=True)

    # 读入原图，得到 (高, 宽, 3) 的 numpy 数组，OpenCV 的通道顺序是 BGR
    pre_img = cv2.imread(pre_path)

    # 裁剪就是 numpy 切片：第一维是行（对应 y），第二维是列（对应 x），所以 y 在前 x 在后
    # 例：pre_img[27:331, 60:385] 切出高 304、宽 325 的鸟身区域
    # 就算框略微超出图片边界，切片也只会自动截到图片边缘，不会报错
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = id_to_bbox[img_id]
    out_img = pre_img[bbox_y1:bbox_y2, bbox_x1:bbox_x2] # Crop the bird part

    # 写出裁剪图，文件名和原图同名；imwrite 也按 BGR 解释数组，和 imread 一读一写正好抵消，颜色不会错
    cv2.imwrite(out_path, out_img)  # Save the cropped image

# ======================================
# 收尾：把三个标注文件原样复制到输出目录
# 裁剪不改变图片的 id、相对路径、类别标签和 train/test 划分，所以这三个文件能直接复用，
# 复制过去之后 cub200_cropped/ 就自成一个完整数据集，
# 数据集类（data_sets/fg_bird_dataset.py）读它和读原始 CUB 用的是同一套逻辑
# 注意 bounding_boxes.txt 没复制：框已经"用掉"了，裁剪图整张就是框内内容
# ======================================

# Copy the required annotation files
required_files = [
    'images.txt', 
    'image_class_labels.txt', 
    'train_test_split.txt'
]
for file_name in required_files:
    shutil.copy(
        os.path.join(data_root, file_name), 
        os.path.join(out_dir, file_name)
    )