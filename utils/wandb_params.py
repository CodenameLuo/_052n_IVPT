"""Weights & Biases (W&B) logging utilities for IVPT."""

import copy

import wandb


def init_wandb(args):
    wandb.login()
    if isinstance(args, dict):
        args_dict = args
    else:
        args_dict = vars(args)
    if args["resume_training"]:
        if args_dict["wandb_resume_id"] is not None:
            run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                             job_type=args_dict["job_type"],
                             group=args_dict["group"], mode=args_dict["wandb_mode"],
                             config=args_dict, id=args_dict["wandb_resume_id"], resume="must")
        else:
            raise ValueError("wandb_resume_id is None")
    else:
        run = wandb.init(project=args_dict["wandb_project"], entity=args_dict["wandb_entity"],
                         job_type=args_dict["job_type"],
                         group=args_dict["group"], mode=args_dict["wandb_mode"],
                         config=args_dict)
    return run


def get_train_loggers(args):
    """Get the train loggers for the experiment"""
    
    # 初始 list 为空
    train_loggers = []

    if args.wandb:
        # 若有参数 wandb

        # ars(args)：把 Namespace 转成普通 dict；deepcopy：拷一份独立快照
        # 例：{ 'lr':1e-6, 'epochs':25, 'freeze_backbone':True, 'wandb_project':'', ...}
        # 
        # 为什么 deepcopy：
        # ① 怕后续别处改了 args 污染这份要上报的配置 
        # ② 这份要被 wandb.init(config=...) 记下来，留独立副本最干净
        wandb_logger_settings = copy.deepcopy(vars(args))

        # 把这份配置 dict 塞进 list（注意：塞的是 dict，不是 wandb 对象）
        train_loggers.append(wandb_logger_settings)

    return train_loggers
