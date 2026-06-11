#!/usr/bin/env python
# 验证用脚本：先把"大号例子"的场景特征图 x_ 和 layer-0 的软分割图算出来看一眼
# 目的：动笔写 forward_loss.md 之前，确认结构化设计真能产出"干净、可读的每通道数字矩阵"
# 跑法：用 IVPT 专用 env  _004n_IVPT/bin/python 跑

import torch

torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)
torch.manual_seed(0)

# ============================================================
# 例子规模（大号）
B = 6        # 批量 6 张图
D = 16       # 嵌入维
H = W = 6    # 特征图网格 6×6 = 36 个 patch
# 这一层（layer-0，最细）原型数 = 8 = 7 前景细原型 + 1 背景
P1 = 8
# ============================================================

# ---- 通道语义：16 维分成 4 个语义块，每块 4 维 ----
# ch 0-3  : head（头）探测通道
# ch 4-7  : wing（翅）探测通道
# ch 8-11 : body（身）探测通道
# ch 12-15: bg/纹理 通道
HIGH, LOW = 1.0, 0.10

def archetype(block):
    # 造一个 16 维"区域原型向量"：指定语义块=HIGH，其余=LOW
    v = torch.full((D,), LOW)
    if block == 'head': v[0:4]   = HIGH
    if block == 'wing': v[4:8]   = HIGH
    if block == 'body': v[8:12]  = HIGH
    if block == 'bg':   v[12:16] = 0.5; v[0:12] = 0.08   # 背景：自己有个弱纹理签名
    return v

A = {k: archetype(k) for k in ['head', 'wing', 'body', 'bg']}

# ---- 6×6 区域布局（每个 patch 属于哪个区域）----
# 头在顶部中间；翅在左右两翼；身在中下；其余是背景
hd, wg, bd, bg = 'head', 'wing', 'body', 'bg'
LAYOUT = [
    [bg, bg, hd, hd, bg, bg],
    [bg, hd, hd, hd, hd, bg],
    [wg, wg, bd, bd, wg, wg],
    [wg, wg, bd, bd, wg, wg],
    [bg, wg, bd, bd, wg, bg],
    [bg, bg, bd, bd, bg, bg],
]

# ---- 由布局 + 原型 + 噪声，铺出特征图 x_  [B, D, H, W] ----
x_ = torch.zeros(B, D, H, W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            x_[b, :, r, c] = A[LAYOUT[r][c]] + 0.04 * torch.randn(D)

# ============================================================
# layer-0 的 8 个原型（7 前景 + 1 背景）
# 7 个前景 = 头L 头R 翅L 翅R 身上 身下 身中（=2头 2翅 3身）
# 原型向量都≈对应区域 archetype；区域内部的"左右/上下"切分交给 p_bias 空间先验
proto_specs = [
    ('head', 'head-L'), ('head', 'head-R'),
    ('wing', 'wing-L'), ('wing', 'wing-R'),
    ('body', 'body-U'), ('body', 'body-D'), ('body', 'body-C'),
    ('bg',   'bg'),
]
q = torch.stack([A[blk] for blk, _ in proto_specs], dim=0)      # [8, 16]
q = q.unsqueeze(0).expand(B, -1, -1).contiguous()               # [B, 8, 16]

# ---- p_bias：空间先验 [B, 8, 6, 6]，把区域内部按位置切给不同细原型 ----
p_bias = torch.zeros(B, P1, H, W)
def set_bias(idx, mask, val=3.0):
    for r in range(H):
        for c in range(W):
            if mask(r, c): p_bias[:, idx, r, c] = val
# 头L=头区左半(col<3)，头R=头区右半
set_bias(0, lambda r, c: LAYOUT[r][c] == 'head' and c < 3)
set_bias(1, lambda r, c: LAYOUT[r][c] == 'head' and c >= 3)
# 翅L=左翼(col<3)，翅R=右翼
set_bias(2, lambda r, c: LAYOUT[r][c] == 'wing' and c < 3)
set_bias(3, lambda r, c: LAYOUT[r][c] == 'wing' and c >= 3)
# 身上(row<4) 身下(row>=4) 身中(row==3,4 中两格) —— 让三者有点重叠，体现区域内竞争
set_bias(4, lambda r, c: LAYOUT[r][c] == 'body' and r < 4)
set_bias(5, lambda r, c: LAYOUT[r][c] == 'body' and r >= 4)
set_bias(6, lambda r, c: LAYOUT[r][c] == 'body' and r in (3, 4))
# 背景
set_bias(7, lambda r, c: LAYOUT[r][c] == 'bg')

# ============================================================
# compute_xq —— 逐字搬自 individual_landmark_vit.py:407-437
def compute_xq(x, q):
    ab = torch.einsum('bchw,blc->blhw', x, q)
    b_sq = x.pow(2).sum(1, keepdim=True)
    b_sq = b_sq.expand(-1, q.shape[1], -1, -1).contiguous()
    a_sq = q.pow(2).sum(-1, keepdim=True)
    a_sq = a_sq.expand(-1, -1, x.shape[-2] * x.shape[-1])
    a_sq = a_sq.view(x.shape[0], q.shape[1], x.shape[-2], x.shape[-1])
    dist = b_sq - 2 * ab + a_sq
    return -dist

sim = compute_xq(x_, q)                                  # [B, 8, 6, 6] 负欧氏距离相似度
maps = torch.nn.functional.softmax(sim + p_bias, dim=1)  # [B, 8, 6, 6] 软分割图（通道 softmax）

# ============================================================
# 打印工具
def show_grid_labels():
    print("区域布局 (6×6)：")
    short = {'head': 'hd', 'wing': 'wg', 'body': 'bd', 'bg': 'bg'}
    for r in range(H):
        print("   " + " ".join(short[LAYOUT[r][c]] for c in range(W)))

def show_channels(t, names, b=0, title=""):
    # t: [B, C, H, W]；逐通道打印 b 这张图的 H×W 数字矩阵
    print(f"\n===== {title}  shape={list(t.shape)}  (展示 b={b}) =====")
    C = t.shape[1]
    for ch in range(C):
        nm = names[ch] if names else f"ch{ch}"
        print(f"\n[{title} | 通道 {ch} = {nm}]  ({H}×{W})")
        m = t[b, ch]
        for r in range(H):
            print("   " + " ".join(f"{m[r,c].item():5.2f}" for c in range(W)))

print("################ 例子规模 ################")
print(f"B={B}  D={D}  网格={H}×{W}={H*W}  layer-0 原型数 P+1={P1}")
print(f"x_ (特征图) shape = {list(x_.shape)}")
print(f"q  (原型)   shape = {list(q.shape)}")
print(f"p_bias      shape = {list(p_bias.shape)}")
print(f"sim         shape = {list(sim.shape)}")
print(f"maps        shape = {list(maps.shape)}")
print()
show_grid_labels()

feat_names = ([f"head{i}" for i in range(4)] + [f"wing{i}" for i in range(4)] +
              [f"body{i}" for i in range(4)] + [f"bg{i}" for i in range(4)])
proto_names = [nm for _, nm in proto_specs]

# 特征图：16 个通道矩阵（b=0）
show_channels(x_, feat_names, b=0, title="x_ 特征图")

# compute_xq 相似度：8 个原型矩阵（b=0）
show_channels(sim, proto_names, b=0, title="sim 负欧氏距离")

# 软分割图：8 个原型矩阵（b=0）
show_channels(maps, proto_names, b=0, title="maps 软分割图")

# 校验：每个 patch 的 8 通道软分配应和为 1
print(f"\n[校验] maps 沿通道求和（应全=1.00），b=0：")
s = maps[0].sum(0)
for r in range(H):
    print("   " + " ".join(f"{s[r,c].item():5.2f}" for c in range(W)))
