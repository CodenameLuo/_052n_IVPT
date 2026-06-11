#!/usr/bin/env bash
# IVPT 环境一键复现 —— 2026-06-03 验证可用
# 前提：已装 conda；NVIDIA 驱动 ≥ 525（支持 CUDA 12.1）。无需单独装 CUDA Toolkit。
set -euo pipefail
cd "$(dirname "$0")"

# 环境路径（换机器时改这一行；或运行前 export ENV_PREFIX=... 覆盖）
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_004n_IVPT}"
PY="$ENV_PREFIX/bin/python"

# 1) 没有就建 Python 3.11 环境
if [ ! -x "$PY" ]; then
  echo "[install] 创建环境 $ENV_PREFIX (python 3.11)"
  conda create -p "$ENV_PREFIX" python=3.11 -y
fi

# 2) 配国内镜像，下载快（torch 的 cu121 wheel 在 PyPI 自带，镜像同步即有）
"$PY" -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
"$PY" -m pip config set global.trusted-host mirrors.aliyun.com

# 3) 精确复现：装锁定版本（torch 全家桶约 5GB，首次较慢）
echo "[install] 安装锁定依赖 requirements_lock.txt ..."
"$PY" -m pip install -r requirements_lock.txt

# 4) 自检
echo "[install] 自检 ..."
"$PY" verify_env.py

echo "[install] 完成。激活方式： conda activate \"$ENV_PREFIX\"  （或直接用 $PY）"
