"""Learning-rate scheduler builder (cosine, step, linear)."""

# ======================================
#
# 这个文件对应 train_net.py 第 10 步的调度器部分：按 scheduler_type 造一个学习率调度器。
# 本次用 steplr(StepLR)：每隔 step_size 个 epoch 把学习率乘以 gamma。
# 例(本次 step_size=4, gamma=0.5, 基准 lr≈0.866e-6)：
#   epoch 0~3 用 0.866e-6 → epoch 4 起 ×0.5≈0.433e-6 → epoch 8 起 ≈0.217e-6 → ……(每组参数各自按这个比例衰减)
# 注：t_in_epochs=True 表示调度器“按 epoch”走，不是按 iteration。
#
# ======================================

# 三种调度器实现：cosine / steplr 来自 timm，linearlr 是本仓库自己写的(本次只用 steplr)
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

from .linear_lr_scheduler import LinearLRScheduler


def build_scheduler(args, optimizer):
    """
    Function to build the scheduler
    :param args: arguments from the command line
    :param optimizer: optimizer used for training
    :return: scheduler
    """
    # initialize scheduler hyperparameters
    total_steps = args.epochs                    # 总轮数(cosine/linear 用来定整个周期长度)
    type_lr_schedule = args.scheduler_type       # 调度器类型(本次 steplr)
    warmup_steps = args.scheduler_warmup_epochs  # warmup 轮数(本次 0，无 warmup)
    decay_steps = args.scheduler_step_size       # StepLR 的衰减步长(本次 4)
    warmup_lr_init = args.warmup_lr              # warmup 起始学习率

    restart_factor = args.scheduler_restart_factor
    gamma = args.scheduler_gamma                 # StepLR 衰减系数(本次 0.5)
    min_lr = args.min_lr
    # —— cosine 余弦退火(本次不走) ——
    if type_lr_schedule == 'cosine':
        return CosineLRScheduler(
            optimizer,
            t_initial=total_steps,
            cycle_decay=restart_factor,
            lr_min=min_lr,
            warmup_t=warmup_steps,
            cycle_limit=args.cosine_cycle_limit,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    # —— 【本次走这条】StepLR：每 decay_t 个 epoch 把 lr 乘以 decay_rate ——
    elif type_lr_schedule == 'steplr':
        return StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=gamma,
            warmup_t=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    # —— linearlr 线性衰减(本次不走) ——
    elif type_lr_schedule == 'linearlr':
        return LinearLRScheduler(
            optimizer,
            t_initial=total_steps,
            lr_min_rate=0.01,
            warmup_t=warmup_steps,
            warmup_lr_init=warmup_lr_init,
            t_in_epochs=True
        )
    else:
        raise NotImplementedError
