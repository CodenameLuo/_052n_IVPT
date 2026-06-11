#!/usr/bin/env python
# loss_detail.md 第 2 批：全变差 tv + 存在 presence 的逐元素中间量
# load forward_dump.pt，数字与 forward_loss.md 一致
# 跑法：cd .../_scripts && _004n_IVPT/bin/python _008n_lossdetail_3_4.py
import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

d = torch.load('forward_dump.pt')
maps_list = d['maps_list']      # 4 张 [6,C,6,6]，通道 8/6/5/4
H = W = 6

def grid(t, fmt="{:5.2f}", title=""):
    if title: print(f"  {title}")
    for r in range(t.shape[0]):
        print("    " + " ".join(fmt.format(t[r,c].item()) for c in range(t.shape[1])))

# ============================================================
# 损失3：全变差 TV
print("################ 损失3 全变差 TV ################")
print("计算对象：maps_list 4 张 [6,C,6,6]，C=8/6/5/4（含 bg）")
m = maps_list[-1]                # 用最终图 [6,4,6,6]（最干净，近 0/1，好看差分）
print("\n以 maps_list[-1] 的【头】通道 (b=0, ch0) 为例，6×6：")
grid(m[0,0], "{:6.3f}")

vdiff = m[..., 1:, :] - m[..., :-1, :]    # [6,4,5,6] 纵向差：下一行 - 本行
hdiff = m[..., :, 1:] - m[..., :, :-1]    # [6,4,6,5] 横向差：右一列 - 本列
print("\n纵向差 |diff1| = |下一行 − 本行|  (5×6)：")
grid(vdiff[0,0].abs(), "{:6.3f}")
print("\n横向差 |diff2| = |右一列 − 本列|  (6×5)：")
grid(hdiff[0,0].abs(), "{:6.3f}")
v_sum = vdiff[0,0].abs().sum().item()
h_sum = hdiff[0,0].abs().sum().item()
print(f"\n该(头)通道：|纵向差|求和={v_sum:.3f}  |横向差|求和={h_sum:.3f}  小计={v_sum+h_sum:.3f}")

# 完整聚合
res1 = vdiff.abs().sum([1,2,3])          # [6] 每个样本：对(通道,H,W)求和
res2 = hdiff.abs().sum([1,2,3])
score = res1 + res2                       # [6]
num_elements = m.shape[0]*m.shape[2]*m.shape[3]   # 6*6*6=216
print(f"\n完整聚合（对全部 4 通道、再对样本）：")
print(f"  score 每个样本(对4通道+HW求和) [6] = " + " ".join(f"{score[b].item():.2f}" for b in range(6)))
print(f"  num_elements = B·H·W = {num_elements}")
print(f"  TV(maps_list[-1]) = score.sum()/num_elements = {score.sum().item():.3f}/{num_elements} = {(score.sum()/num_elements).item():.4f}")

print("\n4 张图各自 TV（通道越多边越多→TV 越大）：")
def tv_mean(img):
    d1=img[...,1:,:]-img[...,:-1,:]; d2=img[...,:,1:]-img[...,:,:-1]
    s=d1.abs().sum([1,2,3])+d2.abs().sum([1,2,3])
    return s.sum()/(img.shape[0]*img.shape[2]*img.shape[3])
tvs=[tv_mean(mm).item() for mm in maps_list]
for i,t in enumerate(tvs): print(f"  maps_list[{i}] (C={maps_list[i].shape[1]}) : TV={t:.4f}")
print(f"  loss_tv = 平均 = {sum(tvs)/4:.4f}")

# ============================================================
# 损失4：存在 presence
print("\n\n################ 损失4 存在 presence ################")
print("计算对象：maps_list[:, :-1]（去 bg 的前景通道），以 maps_list[0] 为例 [6,7,6,6]")
m0 = maps_list[0]
fg = m0[:, :-1]                  # [6,7,6,6]  7 个前景原型
names=['head-L','head-R','wing-L','wing-R','body-U','body-D','body-C']

print("\n步骤：avg_pool2d(·,3,stride=1) → adaptive_max_pool2d(·,1) → 跨batch max → 部件均值 → 1−它")
print("\n以【body-C】通道 (ch6, b=0) 为例：")
print("  原 6×6 软图：")
grid(fg[0,6], "{:5.2f}")
pooled = F.avg_pool2d(fg, 3, stride=1)    # [6,7,4,4]
print("  avg_pool2d(3×3,stride1) → 4×4（每格=3×3 窗口均值）：")
grid(pooled[0,6], "{:5.3f}")
print(f"  adaptive_max_pool2d→空间最大 = {pooled[0,6].max().item():.4f}  ← body-C 偏低（半哑火）")

maxed = F.adaptive_max_pool2d(pooled, 1).flatten(1)   # [6,7]
print("\n7 个前景部件 × 6 张图的「空间最大」[6,7]：")
print("        " + " ".join(f"{n:>7s}" for n in names))
for b in range(6):
    print(f"  图{b}: " + " ".join(f"{maxed[b,k].item():7.3f}" for k in range(7)))
maxbatch = maxed.max(0)[0]            # [7] 跨 batch 取 max
print("  跨batch max [7]: " + " ".join(f"{maxbatch[k].item():7.3f}" for k in range(7)))
mean = maxbatch.mean()
print(f"\n  部件均值 = {mean.item():.4f}   →   presence = 1 − {mean.item():.4f} = {(1-mean).item():.4f}")
print("  ↑ body-C 的 max≈0.3 把均值拖低 → 1−均值 升高")

print("\n4 张图各自 presence：")
def pres(maps):
    return 1-F.adaptive_max_pool2d(F.avg_pool2d(maps,3,stride=1),1).flatten(1).max(0)[0].mean()
prs=[pres(mm[:,:-1]).item() for mm in maps_list]
for i,p in enumerate(prs): print(f"  maps_list[{i}] : presence={p:.4f}")
print(f"  loss_presence = 平均 = {sum(prs)/4:.4f}")
