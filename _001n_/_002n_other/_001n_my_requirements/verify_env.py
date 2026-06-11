# IVPT 环境自检：装完跑 `python verify_env.py`，三层都过就能跑代码
import importlib

# 第一层：torch 能用上 GPU
import torch, torchvision, numpy
print("torch       :", torch.__version__)
print("torchvision :", torchvision.__version__)
print("numpy       :", numpy.__version__)            # 应为 1.26.4，不能是 2.x
print("cuda build  :", torch.version.cuda)           # 应为 12.1
print("CUDA 可用   :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用：检查 NVIDIA 驱动，或 torch 装成了 CPU 版"
print("GPU         :", torch.cuda.get_device_name(0))

# 第二层：IVPT 依赖的第三方包都在
pkgs = "timm torchmetrics wandb pytopk cv2 skimage colorcet sklearn pandas scipy matplotlib yaml fsspec safetensors huggingface_hub PIL".split()
for m in pkgs:
    mod = importlib.import_module(m)
    print(f"  ok {m:16s} {getattr(mod, '__version__', '?')}")

# 第三层：IVPT 用到的两个 torch-2.x 关键 API 真能在 GPU 上跑
# 例：q/k/v 形状 (batch=2, heads=8, tokens=197, dim=64)，模拟一层注意力的前向+反向
import torch.nn.functional as F
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    q = torch.randn(2, 8, 197, 64, device="cuda", requires_grad=True)
    k = torch.randn(2, 8, 197, 64, device="cuda")
    v = torch.randn(2, 8, 197, 64, device="cuda")
    out = F.scaled_dot_product_attention(q, k, v)    # torch 2.0 才有
out.float().sum().backward()                         # autocast 下反向也要能走通
print("SDPA + amp 前向反向 OK :", tuple(out.shape), out.dtype, "| grad:", q.grad is not None)

print("\n环境自检通过 ✅")
