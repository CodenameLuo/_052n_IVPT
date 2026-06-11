#!/usr/bin/env python
# loss_detail.md 第 1 批：分类 CE + 一致性 KL 的逐元素中间量
# 直接 load _004n 存的 forward_dump.pt（m_buffer / maps_list / outputs 等），保证数字与 forward_loss.md 一致
# 跑法：cd .../_scripts && _004n_IVPT/bin/python _007n_lossdetail_1_2.py
import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

d = torch.load('forward_dump.pt')
m_buffer = d['m_buffer']        # 4 条 [6,4,36]，通道=[头,翅,身,bg]
outputs  = d['outputs']         # [6,10]
targets  = torch.tensor([2,2,2,2,2,2])
H = W = 6

# ============================================================
# 损失1：分类 CE
print("################ 损失1 分类 CE ################")
print("计算对象：outputs [6,10]（forward 的 scores.mean(-1)）+ targets [6]（=cls2）")
print("\noutputs 全 6 行：")
for b in range(6):
    print(f"  图{b}: " + " ".join(f"{outputs[b,j].item():6.3f}" for j in range(10)))

print("\n以 b=0 逐步：")
o0 = outputs[0]
print("  ① 取指数 exp(outputs[0])：")
print("     " + " ".join(f"{torch.exp(o0)[j].item():6.3f}" for j in range(10)))
Z = torch.exp(o0).sum()
print(f"  ② 求和 Z = Σ exp = {Z.item():.4f}")
sm = F.softmax(o0, 0)
print("  ③ softmax = exp/Z：")
print("     " + " ".join(f"{sm[j].item():6.4f}" for j in range(10)))
print(f"  ④ 取目标类 cls2 概率 = {sm[2].item():.4f}")
print(f"  ⑤ CE_0 = -ln(0.{int(sm[2].item()*10000):04d}) = {-torch.log(sm[2]).item():.4f}")

print("\n6 行各自 CE，再平均：")
ce = F.cross_entropy(outputs, targets, reduction='none')
for b in range(6):
    print(f"  图{b}: softmax[cls2]={F.softmax(outputs[b],0)[2].item():.4f}  CE={ce[b].item():.4f}")
print(f"  loss_classification = mean = {ce.mean().item():.4f}")

# ============================================================
# 损失2：一致性 KL
print("\n\n################ 损失2 一致性 KL ################")
print("计算对象：m_buffer 4 条 [6,4,36]。pred=m_buffer[0/1/2]（前3层粗图）, target=m_buffer[3]（收尾图,detach）")

def kl_per_pos(pred, target, eps=1e-10):
    pred = pred + eps; target = target + eps
    return (target * torch.log(target / pred)).sum(dim=1)   # [6,36] 逐位置 KL（已对 4 通道求和）

target = m_buffer[3]
pred0 = m_buffer[0]
klpp = kl_per_pos(pred0, target)        # [6,36]

print("\n【pred = m_buffer[0] (L0) vs target = m_buffer[3] (收尾)】")
print("逐位置 KL 热图 (b=0, 6×6)：每格 = Σ_4通道 target·ln(target/pred)")
kl0 = klpp[0].view(H, W)
for r in range(H):
    print("   " + " ".join(f"{kl0[r,c].item():6.4f}" for c in range(W)))
print(f"  → 36 位置平均 = {klpp[0].mean().item():.5f}（b=0）；6 张图再平均 = {klpp.mean().item():.5f}")

# 放大看 3 个位置的 4 通道向量
def show_pos(r, c):
    p = r*W + c
    pv = (pred0[0,:,p]); tv = (target[0,:,p])
    print(f"\n  位置 (r={r},c={c})  通道=[头,翅,身,bg]")
    print(f"    pred (L0)  = [{pv[0]:.3f}, {pv[1]:.3f}, {pv[2]:.3f}, {pv[3]:.3f}]")
    print(f"    target(收尾)= [{tv[0]:.3f}, {tv[1]:.3f}, {tv[2]:.3f}, {tv[3]:.3f}]")
    terms = []
    s = 0.0
    for k,nm in enumerate(['头','翅','身','bg']):
        t = (tv[k]+1e-10)*torch.log((tv[k]+1e-10)/(pv[k]+1e-10))
        terms.append(f"{nm}:{t.item():+.4f}")
        s += t.item()
    print(f"    逐通道 target·ln(target/pred) = " + "  ".join(terms))
    print(f"    该位置 KL = Σ = {s:.4f}")

show_pos(0, 2)   # 干净头 patch：KL≈0
show_pos(3, 2)   # 身体中段泄漏 patch：KL 高
show_pos(4, 2)   # 身体中段泄漏 patch

# 三层各自的平均 KL
print("\n三层 pred 各自对 target 的平均 KL（= forward_loss.md 的 0.081/0.036/0.036）：")
for i in range(3):
    print(f"  m_buffer[{i}] vs m_buffer[3] : {kl_per_pos(m_buffer[i], target).mean().item():.5f}")
print(f"  loss_consistency = 三层平均 = {sum(kl_per_pos(m_buffer[i],target).mean() for i in range(3)).item()/3:.5f}")
