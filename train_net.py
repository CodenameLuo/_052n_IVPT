"""
Main entry point for IVPT training and classification evaluation.

Usage:
    # Multi-GPU training
    torchrun --nproc_per_node=4 train_net.py --data_path <path> ...

    # Single-GPU training
    python train_net.py --data_path <path> ...

    # Evaluation only
    python train_net.py --data_path <path> --eval_only --snapshot_dir <ckpt_dir>
"""

# ======================================

import torch
from timeit import default_timer as timer

# ======================================

from argument_parser_train import parse_args
from utils.data_utils.transform_utils import load_transforms
from utils.training_utils.optimizer_params import build_optimizer, layer_group_matcher_ivpt
from utils.training_utils.scheduler_params import build_scheduler
from utils.misc_utils import sync_bn_conversion, check_snapshot
from utils.training_utils.ddp_utils import multi_gpu_check
from utils.wandb_params import get_train_loggers
from engine.distributed_trainer_ivpt import launch_ivpt_trainer
from data_sets.builder import get_dataset
from models.builder import load_model_ivpt
from engine.losses.builder import load_classification_loss, load_loss_hyper_params

# ======================================

# 输入尺寸固定时 (恒 518×518)，让 cuDNN 头几个 batch 试跑多种卷积算法、记下最快的复用
# 例：开头 1~2 步略慢(在挑算法)、之后每步提速；尺寸若频繁变反而拖累
torch.backends.cudnn.benchmark = True


def ivpt_train_eval():
    # 参数解析，拿到参数 args
    args = parse_args()

    # 默认没有 wandb 参数，train_loggers 是空 list
    train_loggers = get_train_loggers(args)

    # Create directory to save training checkpoints, otherwise load the existing checkpoint
    check_snapshot(args)

    # Get the transforms and load the dataset
    train_transforms, test_transforms = load_transforms(args)

    # Load the dataset
    dataset_train, dataset_test, num_cls = get_dataset(
        args, 
        train_transforms, 
        test_transforms
    )

    # Load the model
    model = load_model_ivpt(args, num_cls)

    # Check if there are multiple GPUs
    use_ddp = multi_gpu_check()
    # Convert BatchNorm to SyncBatchNorm if there is more than 1 GPU
    if use_ddp:
        model = sync_bn_conversion(model)

    # Load the loss function
    loss_fn, mixup_fn = load_classification_loss(
        args, 
        dataset_train, 
        num_cls
    )

    # Load the loss hyperparameters
    loss_hyperparams, eq_affine_transform_params = load_loss_hyper_params(args)

    # Define the optimizer and scheduler
    param_groups = layer_group_matcher_ivpt(args, model)
    optimizer = build_optimizer(args, param_groups, dataset_train)
    scheduler = build_scheduler(args, optimizer)

    # Start the timer
    start_time = timer()
    # Setup training and save the results
    launch_ivpt_trainer(
        model=model, 
        train_dataset=dataset_train, 
        test_dataset=dataset_test, 
        batch_size=args.batch_size, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        loss_fn=loss_fn, 
        epochs=args.epochs, 
        save_every=args.save_every_n_epochs, 
        loggers=train_loggers, 
        log_freq=args.log_interval, 
        use_amp=args.use_amp, 
        snapshot_path=args.snapshot_dir, 
        grad_norm_clip=args.grad_norm_clip, 
        num_workers=args.num_workers, 
        mixup_fn=mixup_fn, 
        seed=args.seed, 
        eval_only=args.eval_only, 
        loss_hyperparams=loss_hyperparams, 
        eq_affine_transform_params=eq_affine_transform_params, 
        use_ddp=use_ddp, 
        sub_path_test=args.image_sub_path_test, 
        dataset_name=args.dataset, 
        amap_saving_prob=args.amap_saving_prob, 
        class_balanced_sampling=args.use_class_balanced_sampling, 
        num_samples_per_class=args.num_samples_per_class, 
        n_pro=args.n_pro, 
        enable_hierarchy_vis=args.enable_hierarchy_vis, 
        epoch_fraction=args.epoch_fraction, 
        eval_every_n_epochs=args.eval_every_n_epochs, 
        eval_fraction=args.eval_fraction, 
    )

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    ivpt_train_eval()