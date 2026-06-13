"""Data augmentation pipeline for cropped CUB images.

Uses the ``Augmentor`` library to apply rotation, skew, and shear
transforms to the training split of the cropped dataset.

Usage::

    python utils/img_aug.py --data_path datasets/cub200_cropped
"""

# ======================================
# 这个脚本做一件事：对 crop.py 裁出来的训练集做"离线数据增强"，
# 把每张训练图旋转 / 透视 / 错切出几十个变体，落盘成一个新的图片集
#
# 输入：datasets/cub200_cropped/train_cropped/<类别文件夹>/<图片>
#       （上一步 crop.py 的产物：5994 张训练裁剪图，按 200 个类别分文件夹放）
# 输出：datasets/cub200_cropped/train_cropped_augmented/<类别文件夹>/<增强图>
#       （理论上的目标位置：每张原图生成 30 个增强变体，5994 × 30 ≈ 18 万张）
#
# 名词：这里是"离线增强"——训练开始前就把增强图全部生成好、写到磁盘，训练时当普通图片直接读。
#       区别于"在线增强"（My_CIL 里 transforms.RandomResizedCrop 那种：每个 epoch 临时变换、不落盘）。
#       离线增强的变体固定、可复现、省训练时的 CPU；代价是吃磁盘、且增强的多样性被一次性写死。
#
# 干活全靠第三方库 Augmentor：玩法是建一个"流水线"(Pipeline) 对象，
# 往里加几个增强算子（旋转、翻转……），再调 process() 把整个目录的图挨个过一遍
#
# 运行方式：在仓库根目录下执行 python utils/img_aug.py --data_path datasets/cub200_cropped
#
# 注意（实测确认的坑，文件末尾会详细复盘）：
#   Augmentor 把 output_directory 当成"相对 source_directory"来解析。而 README 给的 --data_path
#   是相对路径，于是增强图并不会落在 train_cropped_augmented/ 下，而是被嵌套塞进
#   train_cropped/<类别>/ 内部的一长串子路径里，预期的 train_cropped_augmented/ 反而是空的。
#   想让它正确落盘，--data_path 需要传绝对路径（文件末尾给修复说明）
#
# 这一步在 README 里和 crop.py 一样标为可选
# ======================================

# 解析命令行参数（这里只有一个 --data_path）
import argparse
# 拼路径、建文件夹、遍历目录
import os
# 第三方数据增强库（需单独 pip install Augmentor）：用"流水线"方式对整个图片目录批量做几何变换
# 注意它不在本仓库 requirements.txt 的默认依赖里（那一行被注释掉了），跑这步前要先自己装上
import Augmentor


# ======================================
# 小工具：目录不存在就创建（os.makedirs 会把中间缺的各级目录一并建出来），已存在则什么都不做
# 跟 crop.py 里直接用的 os.makedirs(..., exist_ok=True) 是一回事，这里包成函数复用
# ======================================
def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


# ======================================
# 命令行参数：只接收一个 --data_path，指向 裁剪数据集 的根目录
# 例：python utils/img_aug.py --data_path datasets/cub200_cropped
# ======================================

parser = argparse.ArgumentParser()
# --data_path：裁剪数据集根目录，里面要有 crop.py 产出的 train_cropped/ 子目录
parser.add_argument('--data_path', type=str)
args = parser.parse_args()

# 这里注释掉的是原作者写死的默认值，现在改成从命令行 --data_path 读
# datasets_root_dir = 'datasets/cub200_cropped/'
datasets_root_dir = args.data_path

# 输入目录：训练集裁剪图所在处（下面每个类别一个子文件夹）
# 注意 dir 是 Python 内置函数名，这里被当普通变量名用了（局部覆盖了内置）——小瑕疵，不影响运行
dir = os.path.join(datasets_root_dir, 'train_cropped/')

# 输出目录：增强图"本应"去的地方——"本应"是因为实际落盘位置受 Augmentor 路径解析影响，见文件末尾复盘
target_dir = os.path.join(datasets_root_dir, 'train_cropped_augmented/')

# 先把输出根目录建出来
makedir(target_dir)

# next(os.walk(dir))[1]：取 dir 下所有"子文件夹名"
#   os.walk(dir) 是生成器，第一次 next (只调用 next 一次) 产出三元组 (当前目录, [子目录名...], [文件名...])，取 [1] 即 子目录名 列表
#   对 CUB 训练集就是 200 个类别文件夹名，例如 '001.Black_footed_Albatross'
#   next(os.walk(dir))[1] : [子目录名...] = ['001.Black_footed_Albatross', '002.Laysan_Albatross', ..., '200.Common_Yellowthroat']
class_folders = next(os.walk(dir))[1]

# folders：每个类别的 输入路径 
# 例：datasets/cub200_cropped/train_cropped/001.Black_footed_Albatross
folders = [os.path.join(dir, folder) for folder in class_folders]

# target_folders：每个类别的 输出路径 ，和 folders 按类别一一对应
# 例：datasets/cub200_cropped/train_cropped_augmented/001.Black_footed_Albatross
# 
# target_folders = [os.path.join(target_dir, folder) for folder in class_folders]
# 
# 由于 Augmentor 内部用 os.path.join(source_directory, output_directory) 拼输出目录
# 那么这里通过 ‘../’ 等效修复路径
target_folders = [os.path.join('../../train_cropped_augmented', folder) for folder in class_folders]

# ======================================
# 主循环：逐个类别文件夹做增强
# 对每个类别建三条独立的增强流水线——旋转、透视倾斜、错切——每条各跑 10 遍
# 三条流水线的算子不同，所以每条都新建一个 Pipeline、用完 del 掉，不复用
# ======================================

# 遍历 200 个类别（i 是类别下标）
for i in range(len(folders)):
    # 当前类别的 输入文件夹 / 输出文件夹
    fd = folders[i]
    tfd = target_folders[i]

    # ---- 流水线 1：随机旋转 ----
    # rotation

    # 新建流水线：源目录 = 当前类别输入，输出目录 = 当前类别输出
    # （注意 Augmentor 把 output_directory 当成"相对 source_directory"解析，落盘位置的坑见文末复盘）
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # 旋转算子：probability=1 → 每张图必转；方向(左/右)和角度都在 [-15°, +15°] 内随机
    # Augmentor 转完会自动从转正的图里裁出"最大的同宽高比矩形"再缩放回原尺寸，所以输出无黑边、尺寸不变
    # 隐患：Augmentor 原生 rotate 在极端宽扁图（横宽比 ≥3.8:1）抽到 13~15° 时，裁内接矩形会算出颠倒框、PIL 抛 ValueError；
    #       本原始版没有防护，跑真实 CUB 撞上这类图就会中途崩溃。参考版 IVPT_tmp 用猴补丁捕获重试修了（详见文末复盘第 4 点）
    # 
    # p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    # 角度修小，避免崩溃
    p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
    # 水平翻转：probability=0.5，大约一半的图会被左右镜像
    # （小细节：Augmentor 内部把随机数 round 到 1 位小数再跟概率比，所以 0.5 的实际命中率略高于一半，无伤大雅）
    p.flip_left_right(probability=0.5)

    # process() 把源目录每张图各过一遍流水线、各存 1 张输出；跑 10 遍 → 每张原图产出 10 个"旋转(+可能翻转)"变体
    # （这里的 i 复用了外层的类别下标 i，但无妨：外层 for 每轮会重新给 i 赋值，内层用完即弃）
    for i in range(10):
        p.process()

    # 销毁这条流水线，下面两条各自重建
    del p

    # ---- 流水线 2：随机透视倾斜（skew）----
    # skew

    # 结构和流水线 1 完全一样，只把"旋转"换成"透视倾斜"，翻转 / 跑 10 遍 / del 都照旧
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # skew 模拟"换个视角看物体"的透视形变；magnitude=0.2 控制最大倾斜量
    # 注意 magnitude 是"占图片边长的比例"(取值 0~1)、不是角度——行末原注释 "max 45 degrees" 其实有点误导
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)

    for i in range(10):
        p.process()

    del p

    # ---- 流水线 3：随机错切（shear）----
    # shear

    # 同样的结构，换成错切：沿 x 或 y 轴把图"推斜"成平行四边形
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # 错切角度在左右各 10° 内随机，错切的轴(x/y)也随机；同样裁出最大同比例矩形再缩放回原尺寸
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)

    for i in range(10):
        p.process()

    del p

    # ---- 流水线 4：随机弹性扭曲（random_distortion）—— 原作者整段注释掉了，默认不启用 ----
    # random_distortion

    # 弹性扭曲会在图上铺一个 10×10 网格、随机抖动网格交点，做出"局部揉皱"式的非刚性形变
    # 没启用大概是因为：细粒度鸟类分类靠的是羽毛 / 喙等局部细节，非刚性扭曲容易破坏这些判别性特征
    # 想用就取消下面几行注释（这样每张原图的变体数会从 30 变成 40）
    # p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    # p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    # p.flip_left_right(probability=0.5)
    # 
    # for i in range(10):
    #    p.process()
    # 
    # del p

# ======================================
# 复盘：这一步真正发生了什么 + 必须知道的事实
#
# 1) 增强总量：每个类别 3 条流水线（旋转 / 透视 / 错切）× 每条 process() 跑 10 遍 ×（每遍每张图各产 1 张）
#    = 每张原图 30 张增强图。全训练集 5994 × 30 ≈ 18 万张。
#    输出文件名形如 "<类别名>_original_<原文件名(含扩展名)>_<uuid>.jpg"，靠 uuid 避免重名覆盖。
#    （翻转是概率性的，但每遍 process 对每张图都必产 1 个输出文件，所以总量是确定的 30）
#
# 2) 输出路径的坑（已造假图实测确认）：Augmentor 内部用 os.path.join(source_directory, output_directory)
#    拼输出目录，等于把 output_directory 当成"相对 source_directory"解析。
#    本脚本传进去的 fd、tfd 都是相对路径（因为 README 给的 --data_path 是相对路径），于是实际输出目录
#    变成两者拼接后的怪路径，增强图被嵌套塞进了"输入类别文件夹的内部"：
#        datasets/cub200_cropped/train_cropped/<类别>/datasets/cub200_cropped/train_cropped_augmented/<类别>/
#    而我们以为的 train_cropped_augmented/ 反而是空的！
#    修复有两种等效思路（都是让路径变绝对，使 os.path.join 丢弃前半截、回到预期位置）：
#      · 不改代码：--data_path 传绝对路径，例 python utils/img_aug.py --data_path /绝对路径/到/datasets/cub200_cropped
#      · 改代码：把上面 folders / target_folders 那两行套上 os.path.abspath(...)——参考版 IVPT_tmp 正是这么修的
#
# 3) 要不要跑这一步：就当前这份仓库代码而言，没有任何地方读取 train_cropped_augmented/
#    （已 grep 全仓库：只有 utils/img_aug.py 和它的同款拷贝 eval/img_aug.py 提到这个名字，均为生成方、无消费方；
#     主训练走在线增强、不吃它，可解释性评估只读 test_cropped/）。
#    它是从 ProtoPNet 系管线沿用过来的产物，对本仓库的训练 / 评估都不是必需的——除非你自己的实验要用它。
#
# 4) 与参考版 IVPT_tmp 的差异（本原始版缺的两处修复；按你这次的要求只在注释里说明、不动代码）：
#    a. abspath 落盘修复：见第 2 点——IVPT_tmp 给 folders / target_folders 套了 os.path.abspath，绕开嵌套坑。
#    b. rotate 崩溃补丁：见流水线 1——Augmentor 原生 rotate 在极端宽扁图抽到大角度时会 PIL ValueError 崩溃，
#       IVPT_tmp 在文件开头给 RotateRange.perform_operation 打了猴补丁（捕获 ValueError、换角度重试、最多 20 次）。
#    小结：本原始版直接跑真实 CUB，可能既崩溃（宽扁图）又落错盘（相对路径）。要稳定复现，可参照 IVPT_tmp
#         补这两处，或临时用"绝对 --data_path + 留意宽扁图"先扛过去。
# ======================================