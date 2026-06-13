"""Training engine utilities: meters, progress logging, checkpoint loading."""

# ======================================
#
# 训练器(Part 6/7)会用到的几个小工具。本次训练真正用到的是：
#   AverageMeter        ：在线累计某个标量的平均值(用来累计每个 epoch 的各项 loss / 指标)；
#   load_state_dict_ivpt：读档时把 torch.load 出来的 dict 还原成 Snapshot、抠出模型权重。
# ProgressMeter / Summary 是打印进度的辅助；change_key / accuracy 是死代码(训练器用 torchmetrics 算准确率)。
#
# ======================================

from enum import Enum

import torch

from .snapshot_class import Snapshot


# 给 meter 标注汇总方式的枚举(配合 ProgressMeter 用)
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


# 在线平均器：每来一个(批)数值就更新一次累计平均，避免把所有值都存下来
class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    # 清零(每个 epoch 开始前调一次)
    def reset(self):
        self.val = 0      # 最近一次的值
        self.avg = 0      # 当前累计平均
        self.sum = 0      # 累计总和
        self.count = 0    # 累计样本数

    # 喂入一个值 val(代表 n 个样本)，更新总和/计数/平均
    # 例：先 update(0.8, 4) 再 update(0.6, 4) -> sum=5.6, count=8, avg=0.7
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# 把 batch 序号格式化成 "[ 12/100]" 这样的字符串(对齐位数)
def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 进度打印辅助：把一组 meter 的当前值拼成一行打印出来
class ProgressMeter(object):
    """
    Customized progress meter
    Ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    # 打印某个 batch 处各 meter 的当前值
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    # 打印汇总(注：依赖 meter.summary()，而 AverageMeter 没实现该方法，故此分支实际不被使用)
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))


# 读档：把 torch.load 出来的字典还原成 Snapshot 对象，并抠出模型权重 state_dict
def load_state_dict_ivpt(snapshot_data):
    """Load state dict of a snapshot.

    Args:
        snapshot_data (dict): dictionary containing the state dict of a snapshot
    """
    # **snapshot_data 按字段名解包进 Snapshot(要求 dict 的键和 Snapshot 字段一一对应)
    snapshot = Snapshot(**snapshot_data)
    state_dict = snapshot.model_state
    return snapshot, state_dict


# —— 以下两个是死代码，本次训练路径不调用 ——
# 改 OrderedDict 的某个键名(没被用到)
def change_key(ordered_dict_obj, old, new):
    for _ in range(len(ordered_dict_obj)):
        k, v = ordered_dict_obj.popitem(False)
        ordered_dict_obj[new if old == k else k] = v


# 手算 top-k 准确率(训练器改用 torchmetrics，所以这个没被用到)
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
