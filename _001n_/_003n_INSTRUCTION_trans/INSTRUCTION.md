# 训练说明

本文档包含训练论文中各实验所用模型的说明。

本代码设计为同时支持单 GPU 和多 GPU 训练（包括多节点训练），使用 PyTorch 的分布式数据并行（Distributed Data Parallel, DDP）以及来自 [torchelastic](https://pytorch.org/docs/stable/elastic/run.html) 的 `torchrun` 工具。它还设计为能自动检测 slurm 环境，并为 DDP 多 GPU 训练设置相应的环境变量。

## Batch Size 与学习率
`--batch_size` 和 `--lr` 参数可用于设置训练的 batch size 和学习率。
batch size 是每张 GPU 上的值，因此总 batch size 为 `batch_size * num_gpus`。

如果你想修改 batch size，请按照[平方根缩放规则（square root scaling rule）](https://arxiv.org/abs/1404.5997)相应地调整学习率。
我们使用基准 batch size 16，对应的起始学习率为 1e-6（backbone）。
因此，如果你想使用 batch size 32，应使用学习率 1e-6 * sqrt(32/16) = 1e-6 * sqrt(2) = 1.414e-6。
该缩放并未在训练脚本中实现，所以你需要手动调整学习率。

### 推荐的 Batch Size
- 对于在 CUB/NABirds 上训练的模型，我们推荐使用 batch size 32（或更高）。
- 对于在 PartImageNet OOD、PartImageNet Seg 和 Flowers102 上训练的模型，我们推荐使用 batch size 128（或更高）。对于更低的 batch size，你可能需要开启 weight decay 来稳定训练（推荐值为 0.05）。

## 训练命令
论文中各实验的主要训练命令如下所示。请阅读[数据集相关参数](#数据集相关参数)和[模型相关参数](#模型相关参数)两节，按你的实验需要调整参数。

例如，要复现在 CUB 数据集上 K=4 个前景部件（foreground parts）的模型训练，你可以使用以下命令在单节点 4 张 GPU 上训练模型：
```
torchrun \
--nnodes=1 \
--nproc_per_node=4 \
<base path to the code>/train_net.py \
--model_arch vit_base_patch14_reg4_dinov2.lvd142m \
--pretrained_start_weights \
--data_path <base path to the dataset>/CUB_200_2011 \
--batch_size 4 \
--wandb \
--epochs 28 \
--dataset cub \
--save_every_n_epochs 16 \
--num_workers 2 \
--image_sub_path_train images \
--image_sub_path_test images \
--train_split 1 \
--eval_mode test \
--wandb_project <project name> \
--job_type <job name> \
--group <group name> \
--snapshot_dir <path to save directory> \
--lr 1e-6 \
--optimizer_type adam \
--scheduler_type steplr \
--scheduler_gamma 0.5 \
--scheduler_step_size 4 \
--scratch_lr_factor 1e4 \
--modulation_lr_factor 1e4 \
--finer_lr_factor 1e3 \
--drop_path 0.0 \
--smoothing 0 \
--augmentations_to_use cub_original \
--image_size 518 \
--num_parts 4 \
--weight_decay 0 \
--total_variation_loss 1.0 \
--concentration_loss 0.0 \
--enforced_presence_loss 2 \
--enforced_presence_loss_type enforced_presence \
--pixel_wise_entropy_loss 1.0 \
--gumbel_softmax \
--freeze_backbone \
--presence_loss_type original \
--modulation_type layer_norm \
--modulation_orth \
--grad_norm_clip 2.0
```

### 数据集相关参数
- `--dataset`：数据集的名称。对于 CUB/NABirds，使用 `cub` 或 `nabirds`。对于 PartImageNet OOD，使用 `part_imagenet_ood`。对于 Oxford Flowers，使用 `flowers102`。对于 PartImageNet Seg，使用 `part_imagenet`。还支持更多数据集，详情请参阅 [load_dataset](load_dataset.py) 文件。
- `--data_path`：数据集的路径。文件夹结构应如 [README](README.md) 文件中所述。
- `--image_sub_path_train`：数据集中训练图像的子路径。例如，在 CUB 数据集中，图像位于 `images` 文件夹下。
- `--image_sub_path_test`：数据集中测试图像的子路径。
- `--train_split`：用于训练的数据划分（split）。仅当你希望在 CUB 数据集的子集上训练时适用。在论文中我们始终使用完整数据集进行训练，因此默认值为 1。
- `--eval_mode`：评估模式。使用 `test` 在测试集上评估，使用 `val` 在验证集上评估。默认值为 `test`。论文中的所有实验均在测试集上评估。
- `--augmentations_to_use`：训练所用的数据增强。增强方式定义在 [transform_utils](utils/data_utils/transform_utils.py) 文件中。默认值为 `cub_original`，它使用细粒度分类文献中的标准增强。我们还支持一种更复杂的自动增强（auto-augmentation）策略，可通过将该值设为 `timm` 来使用。它使用 [ConvNeXt 论文](https://arxiv.org/abs/2201.03545)中所用的 auto-augment 策略。在我们论文的所有实验中，我们使用 `cub_original` 增强。
- `--image_size`：输入图像的尺寸。对于 ViT 模型，在 CUB 和 NABirds 上该值设为 518（DinoV2 timm 模型的默认值）。对于 CUB 和 NABirds 上的 ResNet 模型，默认值为 448（与相关工作相同）。对于其他数据集，无论 ViT 还是 ResNet 模型，图像尺寸均设为常数 224。
- `--anno_path_train`：PartImageNet OOD / PartImageNet Seg 数据集的训练标注文件路径。对于 PartImageNet OOD，标注文件由 [README](README.md) 文件中提到的预处理脚本生成。不适用于其他数据集。
- `--anno_path_test`：PartImageNet OOD / PartImageNet Seg 数据集的测试标注文件路径。对于 PartImageNet OOD，该文件同样由 [README](README.md) 文件中提到的预处理脚本生成。不适用于其他数据集。
- `--metadata_path`：[PlantNet300K](https://zenodo.org/records/5645731) 数据集的元数据文件路径。不适用于其他数据集。由于缺少部件标注，我们在论文中未使用该数据集。不过，代码支持该数据集以备将来使用。
- `--species_id_to_name_file`：[PlantNet300K](https://zenodo.org/records/5645731) 数据集中物种 ID 到物种名称映射文件的路径。不适用于其他数据集。
- `--turn_on_mixup_or_cutmix`：开启 mixup 或 cutmix 进行训练。论文中的任何实验我们都没有使用它，因此默认值为 `False`。

### 模型相关参数
- `--model_arch`：模型的架构。对于论文中的实验，我们使用 [ViT-Base DinoV2](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m) 模型。理论上，timm 库中任何被 [VisionTransformer 类](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py)支持的模型都可以使用。此外，我们还支持所有 torchvision 和 timm 的 ResNet 模型以及 timm 的 ConvNeXt 模型。
- `--num_parts`：要发现的前景部件数量。请按数据集需要调整该值。建议使用论文中指定的值。
- `--pretrained_start_weights`：为 backbone 使用预训练权重。这需要有可用的网络连接。如果你想从头训练，可以去掉该 flag。
- `--use_torchvision_resnet_model`：使用 torchvision 实现的 ResNet 模型。这用于 ResNet 模型。如果你想使用 timm 的实现，可以去掉该 flag。
- `--freeze_backbone`：冻结 backbone 权重。这会冻结 ViT backbone 中的所有层，但我们为部件发现（part discovery）引入的层、class token、register token 和位置编码（position embeddings）除外。论文中的实验使用了该设置。对于 ResNet 和 ConvNeXt 模型，该 flag 会冻结除部件发现层之外的整个 backbone。
- `--freeze_params`：仅适用于 ViT 模型。与 `--freeze_backbone` flag 组合使用，可以冻结除部件发现层之外的整个模型。可用它来复现我们完全冻结 ViT 模型的结果。
- `--modulation_type`：部件发现层所用调制（modulation）的类型。默认值为 `layer_norm`。论文中的所有实验我们都使用该值。
- `--modulation_orth`：对调制后的特征施加正交性损失（orthogonality loss）。论文中的实验使用了该设置。
- `--gumbel_softmax`：在部件注意力图（part attention maps）上使用 Gumbel-Softmax 技巧。论文中的实验使用了该设置。
- `--gumbel_softmax_temperature`：Gumbel-Softmax 技巧的温度。默认值为 1.0。论文中的所有实验我们都使用该值。
- `--gumbel_softmax_hard`：对 Gumbel-Softmax 使用直通估计器（straight-through estimator）。论文中的实验未使用该设置。
- `--classifier_type`：类别预测所用分类器的类型。论文中的所有实验我们都使用默认值 `linear`。
- `--part_dropout`：part dropout 的丢弃概率。默认值为 0.3。论文中的所有实验我们都使用该值。
- `--drop_path`：ViT 模型的 drop path 概率。默认值为 0.0。论文中的所有实验我们都使用该值。除非你对模型进行完全微调，否则不会用到它。
- `--noise_variance`：添加到部件注意力图上的高斯噪声的方差。论文中我们没有使用它，因此默认值为 0.0。
- `--grad_norm_clip`：梯度的最大范数。默认值为 2.0。论文中的所有实验我们都使用该值。
- `--output_stride`：仅当你使用来自 timm 的 CNN 模型时适用。

### 检查点与日志参数
- `--snapshot_dir`：保存检查点的目录。可按需自由修改该值。
- `--save_every_n_epochs`：保存检查点以及（可选地）部件分配图（part assignment maps）的间隔。默认值为 16。可按需自由修改该值。默认情况下，会保存验证准确率最高的检查点和最后一个检查点。论文中我们使用验证准确率最高的模型进行评估。
- `--amap_saving_prob`：保存部件分配图的概率。它会在第一个 epoch、每 save_every_n_epochs 个 epoch 以及最后一个 epoch 被触发。设为 0 可将其关闭，设为 1 则每次迭代都保存。我们建议训练时使用 0.05，评估时使用更高的值（如 0.8）。由于这些图会以图像形式保存，训练时可能会造成明显的速度下降。

### 优化器与调度器参数
- `--optimizer_type`：所用优化器的类型。默认值为 `adam`。论文中的所有实验我们都使用该值。
- `--scheduler_type`：所用调度器的类型。默认值为 `steplr`。论文中的所有实验我们都使用该值。
- `--scheduler_gamma`：调度器的 gamma 值。默认值为 0.5。论文中的所有实验我们都使用该值。
- `--scheduler_step_size`：调度器的步长。默认值为 4。论文中的所有实验我们都使用该值。
- `--lr`：训练的学习率。详情请参阅 [Batch Size 与学习率](#batch-size-与学习率)一节。
- `--weight_decay`：优化器的 weight decay。默认值为 0。论文中的所有实验我们都使用该值。我们在代码中实现了 [AdamW 论文](https://arxiv.org/abs/1711.05101)中的归一化 weight decay 公式。
- `--scratch_lr_factor`：从头训练层（scratch layers）的学习率因子。默认值为 1e4。论文中的所有实验我们都使用该值。
- `--modulation_lr_factor`：调制层的学习率因子。默认值为 1e4。论文中的所有实验我们都使用该值。
- `--finer_lr_factor`：finer 层的学习率因子。默认值为 1e3。论文中的所有实验我们都使用该值。

### 损失超参数
损失超参数已设为论文中所用的值。如果你的实验需要，可以自由修改这些值。

### 额外说明
- 如果你想在单张 GPU 上训练，可以去掉 `torchrun` 命令以及 `--nnodes` 和 `--nproc_per_node` flag。然后以 `python <base path to the code>/train_net.py <arguments>` 的方式运行。
- 请注意，代码的编写假设所有可见的 GPU 都将用于训练。如果你只想使用部分 GPU，需要在运行训练脚本前手动设置 `CUDA_VISIBLE_DEVICES` 环境变量。在 slurm 及其他作业调度器环境中这是自动完成的，所以在那些情况下你无需担心。
- `--pretrained_start_weights` flag 用于为 backbone 加载预训练权重。这需要有可用的网络连接。
- 权重会保存在 `~/.cache/torch/hub/checkpoints` 目录中，若该目录已存在，我们的代码会自动检测到它。
- 对于 ResNet 模型，我们同时支持 timm 和 torchvision 的实现。不过请注意，据我们所知，文献中所有关于部件发现的工作使用的都是 torchvision 版本。
- （可选）如果你没有可用的网络连接，也可以单独运行[这里](https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m)的 `create_model()` 函数来下载 vit-base 模型的权重。它会自动检测 `~/.cache/torch/hub/checkpoints` 目录并将权重保存在那里。类似地，对于 resnet 模型，使用 torchvision 的 [get_model 函数](https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models)并设置 `weights="DEFAULT"`，或使用[这里](https://huggingface.co/timm/resnet101.a1h_in1k)的 `create_model()` 获取 timm 权重。
- [ClassBalancedDistributedSampler](utils/data_utils/class_balanced_distributed_sampler.py) 和 [ClassBalancedSampler](utils/data_utils/class_balanced_sampler.py) 类可用于在 mini-batch 中平衡各类别。论文中我们没有使用它们，因此相应的 flag 默认设为 `False`。我们没有对它们做过充分测试，请谨慎使用。
