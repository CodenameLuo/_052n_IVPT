# IVPT 环境安装说明（可复现）

> 记录 IVPT（ICLR 2026）官方代码 `_010n_codes/_019n_IVPT/IVPT` 的 Python 环境如何从零装好。
> 创建于 2026-06-03，下方"验证"一节的检查当时全部通过。

## 0. 环境一览

| 项 | 值 |
|---|---|
| conda env | `/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_004n_IVPT` |
| Python | 3.11.13 |
| 关键框架 | torch 2.2.2（cu121） / torchvision 0.17.2 |
| GPU / 驱动 | NVIDIA RTX 3090 / driver 570.124.04（支持 CUDA 12.1） |
| pip 镜像 | 阿里云 `http://mirrors.aliyun.com/pypi/simple` |
| 包总数 | 82（精确版本见 `requirements_lock.txt`） |

## 1. 前提

- 已装 conda（miniconda / anaconda）
- NVIDIA 驱动 ≥ 525（cu121 运行库的最低驱动要求；本机 570 满足）
- **不需要**单独装 CUDA Toolkit —— torch 的 pip wheel 自带 cu121 运行库（即 `nvidia-*-cu12` 那一堆包）

## 2. 一键复现

```bash
bash install.sh
```

脚本做三件事：没有 env 就建（Python 3.11）→ 装 `requirements_lock.txt` → 跑 `verify_env.py` 自检。换机器时改 `install.sh` 顶部的 `ENV_PREFIX`。

## 3. 手动复现（与一键等价，每步看得清）

```bash
# (1) 建 Python 3.11 环境（ENV_PREFIX 换成你的路径）
ENV_PREFIX=/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_004n_IVPT
conda create -p "$ENV_PREFIX" python=3.11 -y
PY="$ENV_PREFIX/bin/python"

# (2) 可选：配国内镜像，下载快
#     （PyPI 的 linux torch wheel 默认就是 cu121 GPU 版，镜像同步即可，无需 pytorch 官方源）
"$PY" -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
"$PY" -m pip config set global.trusted-host mirrors.aliyun.com

# (3a) 精确复现（推荐）：锁定全部 82 个包的版本
"$PY" -m pip install -r requirements_lock.txt

# (3b) 或语义复现：先钉 numpy + torch，再装项目 requirements
#     先单独装 numpy/torch/torchvision，能确保 CUDA 版本与 numpy<2 不被后续包扰动
"$PY" -m pip install numpy==1.26.4 torch==2.2.2 torchvision==0.17.2
"$PY" -m pip install -r ../../IVPT/requirements.txt numpy==1.26.4

# (4) 自检
"$PY" verify_env.py
```

## 4. 关键版本 & 为什么这么定（以后别再踩坑）

| 选择 | 原因 |
|---|---|
| **Python 3.11** | 对齐作者 `environment.yml`；torch 2.2 支持 3.8–3.11 |
| **torch 2.2.2 + cu121** | 代码**硬需** torch ≥ 2.0：前向用 `F.scaled_dot_product_attention`（2.0 新增，见 `models/layers/transformer_layers.py`）、训练用 `torch.amp.autocast(device_type=...)`（2.1+ 接口，见 `engine/distributed_trainer_ivpt.py`）。torch 1.13 会直接报错。版本对齐作者（2.2.0），cu121 匹配 3090 + driver 570 |
| **numpy 钉死 1.26.4（< 2）** | torch 2.2 编译期基于 numpy 1.x，numpy 2.x 改了 ABI，混用会崩。装 `-r` 时额外再传一次 `numpy==1.26.4`，防止被别的包顺手拉升到 2.x |
| **opencv 用 headless 版** | 服务器无显示器，`opencv-python-headless` 省掉 GUI 依赖；`import cv2` 功能等价 |
| **阿里云镜像** | 国内快。torch 全家桶约 5 GB（cudnn 1.2G / cublas 595M / …），用官方源在国内很慢 |

## 5. 验证（复现后应看到）

跑 `python verify_env.py`，三层都过即可：

1. torch 2.2.2+cu121，`cuda.is_available() = True`，能看到 RTX 3090
2. timm / torchmetrics / pytopk / wandb / cv2 / skimage / colorcet … 全部 import 成功
3. GPU 上 `torch.amp.autocast` + `scaled_dot_product_attention` 前向 + 反向跑通（fp16，梯度正常）

## 6. 注意事项（等真跑训练时）

- **pandas 装成了 3.0.3（很新，破坏性改动多）**：CUB 数据集用 pandas 读 `.txt`，若报 `delim_whitespace` 之类的 API 错，把 pandas 降到 `pandas<2.2` 即可。
- **单卡运行**：仓库 `scripts/run_train.sh` 默认 4 卡 `torchrun`；单张 3090 直接 `python train_net.py ...`（不要 torchrun）。
- **预训练权重**：`--pretrained_start_weights` 会让 timm 联网从 HuggingFace 下载 DINOv2 reg4；国内先 `export HF_ENDPOINT=https://hf-mirror.com`。
- **数据集**：CUB-200-2011 放到 `IVPT/datasets/CUB_200_2011/`。

## 7. 本目录文件

| 文件 | 用途 |
|---|---|
| `README.md` | 本说明 |
| `requirements_lock.txt` | 全部 82 个包的精确版本（`pip freeze` 导出），精确复现用 |
| `install.sh` | 一键复现脚本 |
| `verify_env.py` | 装完自检脚本 |
