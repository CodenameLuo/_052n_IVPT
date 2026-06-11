# 演示 LayerNorm 的"操作过程"和"作用"，全部用真实数字 print 出来
import torch
import torch.nn as nn

print("="*64)
print("Demo 1: LayerNorm 到底算了什么（手算 vs nn.LayerNorm 对拍）")
print("="*64)
x = torch.tensor([10., 12., 14., 16.])
mu  = x.mean()
var = x.var(unbiased=False)            # LayerNorm 用"有偏"方差（除以 N，不是 N-1）
eps = 1e-5
x_hat = (x - mu) / torch.sqrt(var + eps)
print("输入 x           :", x.tolist())
print(f"① 求均值 μ        = {mu.item():.4f}")
print(f"② 求方差 σ²       = {var.item():.4f}   (σ = {var.sqrt().item():.4f})")
print("③ 归一化 (x-μ)/σ  =", [round(v, 4) for v in x_hat.tolist()])
y = nn.LayerNorm(4)(x)                 # 默认 γ=1, β=0
print("nn.LayerNorm(x)  =", [round(v, 4) for v in y.detach().tolist()], " ← 与手算一致")
print(f"输出 mean={y.mean().item():+.4f} std={y.std(unbiased=False).item():.4f}  → 作用1: 任意一行数被拉成 均值0/标准差1")

print("\n" + "="*64)
print("Demo 2: 作用2 —— 对输入的整体'缩放/平移'免疫（只留相对形状）")
print("="*64)
ln = nn.LayerNorm(4)
x2 = 100 * x + 50
print("x        =", x.tolist(),  "-> LN =", [round(v,4) for v in ln(x).detach().tolist()])
print("100*x+50 =", x2.tolist(), "-> LN =", [round(v,4) for v in ln(x2).detach().tolist()])
print("→ 放大100倍再加50，LN 输出几乎不变：丢掉绝对尺度/偏移，只保留'谁比谁大'的相对模式")

print("\n" + "="*64)
print("Demo 3: 作用3 —— 可学习 γ/β 能把尺度'调回来'（归一化不损失表达力）")
print("="*64)
ln2 = nn.LayerNorm(4)
with torch.no_grad():
    ln2.weight.fill_(3.0)              # γ = 3
    ln2.bias.fill_(5.0)               # β = 5
y2 = ln2(x)
print("γ=3, β=5 :", [round(v,4) for v in y2.detach().tolist()])
print(f"  输出 mean={y2.mean().item():.4f} (≈β=5)   std={y2.std(unbiased=False).item():.4f} (≈γ=3)")
print("→ 先归一化到 0/1，再用 γ,β 自由设定输出的 标准差≈γ、均值≈β（网络想要多大尺度自己学）")

print("\n" + "="*64)
print("Demo 4: 作用4 —— 跨深度稳住激活幅度（深层网络/transformer 为何离不开）")
print("="*64)
torch.manual_seed(0)
D = 256
h_no = torch.randn(D); h_ln = h_no.clone()
ln3 = nn.LayerNorm(D)
print(f"{'层':>3} | {'无LN  激活std':>16} | {'有LN  激活std':>14}")
for l in range(1, 9):
    W = torch.randn(D, D) * (1.5 / D**0.5)   # 每过一层把范数 ×~1.5 的随机线性层
    h_no = W @ h_no                          # 不加 LN：每层 ×1.5 → 指数漂移
    h_ln = W @ ln3(h_ln)                     # 加 LN：每层先拉回 std≈1 再 ×1.5 → 不漂
    print(f"{l:>3} | {h_no.std().item():>16.4e} | {h_ln.std().item():>14.4f}")
print("→ 无 LN：std 随层数指数爆炸(几层就 1e+?)；有 LN：每层先归一化，全程稳在 ~1.5")
