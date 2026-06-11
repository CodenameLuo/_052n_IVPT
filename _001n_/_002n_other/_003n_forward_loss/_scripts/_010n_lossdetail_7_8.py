#!/usr/bin/env python
# loss_detail.md 第 4 批（最后）：强制存在 enforced_presence + 逐像素熵 pixel_entropy
# load forward_dump.pt
# 跑法：cd .../_scripts && _004n_IVPT/bin/python _010n_lossdetail_7_8.py
import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

d = torch.load('forward_dump.pt')
maps_list = d['maps_list']      # 4 张 [6,C,6,6]
H = W = 6
def grid(t, fmt="{:6.3f}"):
    for r in range(t.shape[0]):
        print("    " + " ".join(fmt.format(t[r,c].item()) for c in range(t.shape[1])))

# ============================================================
# 损失7：强制存在 enforced_presence
print("################ 损失7 强制存在 enforced_presence ################")
print("计算对象：maps_list（含 bg），只用背景通道 maps[:, -1]。权重 2.0")
m = maps_list[-1]               # [6,4,6,6] 最终图
print("\n以 maps_list[-1] 背景通道 (b=0, ch3=bg) 6×6：")
grid(m[0,-1], "{:5.2f}")

ap = F.avg_pool2d(m, 3, stride=1)    # [6,4,4,4]
print("\n① avg_pool2d(3×3,stride1) 背景通道 → 4×4：")
grid(ap[0,-1], "{:5.3f}")

# 径向 mask（中心0、四角1），逐字按 enforced_presence_loss.py
gx, gy = torch.meshgrid(torch.arange(ap.shape[2]), torch.arange(ap.shape[3]), indexing='ij')
gx = (gx.float()/gx.max())*2-1; gy = (gy.float()/gy.max())*2-1
mask = gx**2 + gy**2; mask = mask/mask.max()
print("\n② 径向 mask（中心≈0、四角=1）4×4：")
grid(mask, "{:5.3f}")

masked = ap[0,-1]*mask
print("\n③ 背景通道 × mask → masked 4×4（强调四角、压住中心）：")
grid(masked, "{:5.3f}")
mx = masked.max()
print(f"\n④ 空间最大 = {mx.item():.4f}")
print(f"⑤ BCE(max, 目标1) = -ln({mx.item():.4f}) = {-torch.log(mx).item():.4f}")

print("\n4 张图各自 BCE，再平均、×2：")
_cache={}
def enf(maps):
    ap=F.avg_pool2d(maps,3,stride=1); k=ap.shape[2]
    if k not in _cache:
        a,b=torch.meshgrid(torch.arange(k),torch.arange(k),indexing='ij')
        a=(a.float()/a.max())*2-1;b=(b.float()/b.max())*2-1;mm=a**2+b**2;_cache[k]=mm/mm.max()
    mbg=(ap*_cache[k])[:,-1,:,:]
    mp=F.adaptive_max_pool2d(mbg,1).flatten(start_dim=0)
    return F.binary_cross_entropy(mp,torch.ones_like(mp))
bces=[enf(mm).item() for mm in maps_list]
for i,b in enumerate(bces): print(f"  maps_list[{i}] : BCE={b:.4f}")
print(f"  loss_enforced_presence = 平均({sum(bces)/4:.4f}) × 2.0 = {sum(bces)/4*2:.4f}")

# ============================================================
# 损失8：逐像素熵 pixel_entropy
print("\n\n################ 损失8 逐像素熵 pixel_entropy ################")
print("计算对象：maps_list（含 bg），以 maps_list[0] [6,8,6,6] 为例")
m0 = maps_list[0]
mc = m0.float().clamp(min=1e-6, max=1.0); mc = mc/mc.sum(dim=1, keepdim=True)   # clamp+重归一
ent = torch.distributions.categorical.Categorical(probs=mc.permute(0,2,3,1).contiguous()).entropy()  # [6,6,6]
print("\n逐 patch 熵 −Σ p·ln p 热图 (b=0, 6×6)：")
grid(ent[0], "{:6.4f}")
print(f"  → 全图平均(b=0) = {ent[0].mean().item():.4f}；6 张图平均 = {ent.mean().item():.4f}")

def show_patch(r,c):
    p = mc[0,:,r,c]
    names=['head-L','head-R','wing-L','wing-R','body-U','body-D','body-C','bg']
    nz = [(names[k],p[k].item()) for k in range(8) if p[k].item()>0.005]
    e = -(p*torch.log(p)).sum().item()
    print(f"\n  patch (r={r},c={c})：8 通道分布（只列 >0.005）")
    print("    " + "  ".join(f"{n}={v:.3f}" for n,v in nz))
    print("    熵 = −Σ p·ln p = " + " + ".join(f"({v:.2f}·ln{v:.2f})" for _,v in nz).replace("(","−(") + f" = {e:.4f}")

show_patch(0,0)   # bg 角：近 one-hot，熵≈0
show_patch(0,2)   # 头：head-L 0.95 + head-R 0.05，熵中等
show_patch(3,2)   # 身体中段：body-U/C 平分，熵最高

print("\n4 张图各自平均熵（越深越尖→越低）：")
def pix(maps):
    mc=maps.float().clamp(1e-6,1.0); mc=mc/mc.sum(1,keepdim=True)
    return torch.distributions.categorical.Categorical(probs=mc.permute(0,2,3,1).contiguous()).entropy().mean()
pes=[pix(mm).item() for mm in maps_list]
for i,p in enumerate(pes): print(f"  maps_list[{i}] (C={maps_list[i].shape[1]}) : entropy={p:.4f}")
print(f"  loss_pixel_wise_entropy = 平均 = {sum(pes)/4:.4f}")
