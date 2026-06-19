"""Weights & Biases (W&B) logging utilities for IVPT."""

# ======================================
#
# wandb(Weights & Biases)是个在线实验记录平台。这个文件负责初始化 wandb。
# train_net.py 第 2 步调用的 get_train_loggers 就在这里：本次没开 --wandb，所以它返回空列表 []，
# 相当于不记 wandb 日志(训练照常，只是不上传指标)。
#
# ======================================

import copy

import wandb


# 真正去连 wandb、开一个 run(只有开了 --wandb 时才会被调到)
def init_wandb(args):
    # 登录(用本地缓存的凭证；没有会提示认证)
    wandb.login()
    # 兼容 args 是 dict 或 argparse.Namespace 两种形式
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)
    # 续训：必须给 wandb_resume_id，才能接着原来的 run 记录
    if args["resume_training"]:
        if args_dict["wandb_resume_id"] is not None:
            run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                             job_type=args_dict["job_type"],
                             group=args_dict["group"], mode=args_dict["wandb_mode"],
                             config=args_dict, id=args_dict["wandb_resume_id"], resume="must")
        else:
            raise ValueError("wandb_resume_id is None")
    # 新训练：开一个全新的 run
    else:
        run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                         job_type=args_dict["job_type"],
                         group=args_dict["group"], mode=args_dict["wandb_mode"],
                         config=args_dict)
    return run


# train_net.py 第 2 步调用：开了 --wandb 才把一份(深拷贝的)配置塞进列表返回；否则返回空列表
def get_train_loggers(args):
    """Get the train loggers for the experiment"""

    train_loggers = []
    if args.wandb:
        wandb_logger_settings = copy.deepcopy(vars(args))
        train_loggers.append(wandb_logger_settings)

    return train_loggers
