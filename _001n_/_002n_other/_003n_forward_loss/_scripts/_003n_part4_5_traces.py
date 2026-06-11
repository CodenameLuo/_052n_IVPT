#!/usr/bin/env python
# Part 4(compute_feat) + Part 5(细→粗路由) 的精确中间量
# 场景构造与 _001n/_002n 逐字一致（同 seed）
# 跑法：_004n_IVPT/bin/python _003n_part4_5_traces.py

import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)
torch.manual_seed(0)

B, D, H, W, P1 = 6, 16, 6, 6, 8
NL = 3   # num_landmarks 粗部件数
HIGH, LOW = 1.0, 0.10

def archetype(block):
    v = torch.full((D,), LOW)
    if block == 'head': v[0:4]   = HIGH
    if block == 'wing': v[4:8]   = HIGH
    if block == 'body': v[8:12]  = HIGH
    if block == 'bg':   v[12:16] = 0.5; v[0:12] = 0.08
    return v
A = {k: archetype(k) for k in ['head','wing','body','bg']}
hd, wg, bd, bg = 'head','wing','body','bg'
LAYOUT = [
    [bg, bg, hd, hd, bg, bg],
    [bg, hd, hd, hd, hd, bg],
    [wg, wg, bd, bd, wg, wg],
    [wg, wg, bd, bd, wg, wg],
    [bg, wg, bd, bd, wg, bg],
    [bg, bg, bd, bd, bg, bg],
]
x_ = torch.zeros(B, D, H, W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            x_[b, :, r, c] = A[LAYOUT[r][c]] + 0.04 * torch.randn(D)

proto_specs = [('head','head-L'),('head','head-R'),('wing','wing-L'),('wing','wing-R'),
               ('body','body-U'),('body','body-D'),('body','body-C'),('bg','bg')]
q = torch.stack([A[blk] for blk,_ in proto_specs],0).unsqueeze(0).expand(B,-1,-1).contiguous()
proto_names = [nm for _,nm in proto_specs]
fine_names  = proto_names[:-1]  # 去掉 bg 的 7 个细前景原型

p_bias = torch.zeros(B, P1, H, W)
def set_bias(idx, mask, val=3.0):
    for r in range(H):
        for c in range(W):
            if mask(r,c): p_bias[:, idx, r, c] = val
set_bias(0, lambda r,c: LAYOUT[r][c]=='head' and c<3)
set_bias(1, lambda r,c: LAYOUT[r][c]=='head' and c>=3)
set_bias(2, lambda r,c: LAYOUT[r][c]=='wing' and c<3)
set_bias(3, lambda r,c: LAYOUT[r][c]=='wing' and c>=3)
set_bias(4, lambda r,c: LAYOUT[r][c]=='body' and r<4)
set_bias(5, lambda r,c: LAYOUT[r][c]=='body' and r>=4)
set_bias(6, lambda r,c: LAYOUT[r][c]=='body' and r in (3,4))
set_bias(7, lambda r,c: LAYOUT[r][c]=='bg')

def compute_xq(x, q):
    ab = torch.einsum('bchw,blc->blhw', x, q)
    b_sq = x.pow(2).sum(1, keepdim=True).expand(-1, q.shape[1], -1, -1).contiguous()
    a_sq = q.pow(2).sum(-1, keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(b_sq - 2*ab + a_sq)
maps = torch.nn.functional.softmax(compute_xq(x_, q) + p_bias, dim=1)

# ============================================================
# compute_feat —— 逐字搬自 individual_landmark_vit.py:439-510
def compute_feat(maps, x):
    N = maps.shape[1]
    one_hot_map = F.one_hot(torch.argmax(maps, dim=1), num_classes=N).permute(0,3,1,2) * maps
    all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
    sum_pool = all_features.sum(dim=(3,4)).permute(0,2,1)
    count_map = one_hot_map.sum(dim=(2,3), keepdim=True)
    count_map_ = count_map + (count_map==0).float()
    all_features = sum_pool / count_map_.squeeze(-1)
    return all_features, count_map, one_hot_map  # 多返回 one_hot_map 给展示

q_x, count_map, one_hot_map = compute_feat(maps.detach(), x_)   # q_x [6,8,16]  count_map [6,8,1,1]

# ---- 1) argmax 归属图（每个 patch 归哪个原型）----
argmax_map = torch.argmax(maps, dim=1)   # [6,6,6]
print("################ Part 4 ################")
print("argmax 硬归属图 (b=0)：每格=该 patch 归属的原型索引")
for r in range(H):
    print("   " + " ".join(f"{argmax_map[0,r,c].item():d}:{proto_names[argmax_map[0,r,c].item()]:6s}" for c in range(W)))

# ---- 2) one_hot_map 两个代表通道（body-U 有patch / body-C 空）----
def show_ch(t, ch, name):
    print(f"\n[one_hot_map ch{ch}={name}] (b=0)（=argmax 命中处保留软值，其余清零）")
    for r in range(H):
        print("   " + " ".join(f"{t[0,ch,r,c].item():5.2f}" for c in range(W)))
show_ch(one_hot_map, 4, 'body-U')
show_ch(one_hot_map, 6, 'body-C')

# ---- 3) count_map（每个原型的软质量）----
print("\ncount_map (b=0)：每个原型 argmax 命中处的软值之和（=软质量/有效大小）")
for k in range(P1):
    print(f"  {proto_names[k]:7s}: {count_map[0,k,0,0].item():.3f}")
print("  ↑ 注意 body-C = 0 → 它是空原型（和 body-U/body-D 完全重叠、平局总输给更小索引，一个 patch 没抢到）")

# ---- 4) q_x 部件描述向量 [6,8,16] (b=0) ----
print("\nq_x 部件描述向量 (b=0) 8×16：")
print("  "+ " "*7 + " ".join(f"c{d:02d}" for d in range(D)))
for k in range(P1):
    print(f"  {proto_names[k]:7s}: " + " ".join(f"{q_x[0,k,d].item():4.2f}" for d in range(D)))
print("  ↑ body-C 整行=0（空原型 → sum_pool=0 / 兜底分母1 = 0 向量）")

# ============================================================
# Part 5：细→粗路由
# p_classifier：手设权重「读通道块」—— 头块→粗0, 翅块→粗1, 身块→粗2
print("\n################ Part 5 ################")
p_classifier = torch.nn.Linear(D, NL, bias=True)
with torch.no_grad():
    Wm = torch.zeros(NL, D)
    Wm[0, 0:4]  = 1.0   # 粗0 head：读 ch0-3
    Wm[1, 4:8]  = 1.0   # 粗1 wing：读 ch4-7
    Wm[2, 8:12] = 1.0   # 粗2 body：读 ch8-11
    p_classifier.weight.copy_(Wm)
    p_classifier.bias.zero_()

q_c = p_classifier(q_x[:, :-1])               # [6,7,3] 去 bg，7 个细前景 → 3 粗部件打分
count_mask = (count_map[:, :-1] == 0).squeeze(-1).expand(-1, -1, q_c.shape[-1])  # [6,7,3]
q_maps = torch.nn.functional.softmax(q_c, dim=-1)          # 普通 softmax（非 gumbel，为可复现）
q_maps = q_maps.masked_fill(count_mask, 1.0/q_c.shape[-1]) # 空原型置 1/3

print("q_c 路由打分 (b=0) 7×3  [logit: 粗0头 粗1翅 粗2身]：")
for k in range(P1-1):
    print(f"  {fine_names[k]:7s}: " + " ".join(f"{q_c[0,k,j].item():6.3f}" for j in range(NL)))

print("\ncount_mask (b=0) 7×3（True=空原型该被填 1/3）：")
for k in range(P1-1):
    print(f"  {fine_names[k]:7s}: " + " ".join(f"{bool(count_mask[0,k,j].item())!s:5s}" for j in range(NL)))

print("\nq_maps 细→粗软分配 (b=0) 7×3（每行和=1）：")
for k in range(P1-1):
    s = q_maps[0,k]
    print(f"  {fine_names[k]:7s}: " + " ".join(f"{s[j].item():.3f}" for j in range(NL)) + f"   (和={s.sum().item():.2f})")
print("  ↑ body-C 那行被 masked_fill 成 [0.333,0.333,0.333]（空原型不站队）")
