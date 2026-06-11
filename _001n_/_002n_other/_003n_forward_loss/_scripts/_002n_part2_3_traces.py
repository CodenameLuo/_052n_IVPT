#!/usr/bin/env python
# Part 2/3 需要的精确中间量：原型 q、单 patch 三项展开(compute_xq)、p_bias、softmax 逐项
# 场景构造与 _001n 逐字一致（同 seed 同顺序 → x_ 完全一致），只是多打印一些 trace
# 跑法：_004n_IVPT/bin/python _002n_part2_3_traces.py

import torch
torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)
torch.manual_seed(0)

B, D, H, W, P1 = 6, 16, 6, 6, 8
HIGH, LOW = 1.0, 0.10

def archetype(block):
    v = torch.full((D,), LOW)
    if block == 'head': v[0:4]   = HIGH
    if block == 'wing': v[4:8]   = HIGH
    if block == 'body': v[8:12]  = HIGH
    if block == 'bg':   v[12:16] = 0.5; v[0:12] = 0.08
    return v

A = {k: archetype(k) for k in ['head', 'wing', 'body', 'bg']}
hd, wg, bd, bg = 'head', 'wing', 'body', 'bg'
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
q = torch.stack([A[blk] for blk, _ in proto_specs], dim=0).unsqueeze(0).expand(B,-1,-1).contiguous()
proto_names = [nm for _, nm in proto_specs]

p_bias = torch.zeros(B, P1, H, W)
def set_bias(idx, mask, val=3.0):
    for r in range(H):
        for c in range(W):
            if mask(r, c): p_bias[:, idx, r, c] = val
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
    a_sq = q.pow(2).sum(-1, keepdim=True).expand(-1, -1, x.shape[-2]*x.shape[-1]).view(x.shape[0], q.shape[1], x.shape[-2], x.shape[-1])
    return -(b_sq - 2*ab + a_sq)

sim  = compute_xq(x_, q)
maps = torch.nn.functional.softmax(sim + p_bias, dim=1)

def show_channels(t, names, b=0, title=""):
    print(f"\n===== {title}  shape={list(t.shape)}  (b={b}) =====")
    for ch in range(t.shape[1]):
        print(f"\n[{title} | ch{ch} = {names[ch]}]")
        for r in range(H):
            print("   " + " ".join(f"{t[b,ch,r,c].item():6.2f}" for c in range(W)))

# ---- 原型 q：8×16 ----
print("################ Part 2：原型 q [b=0] 8×16 ################")
print("（每行一个原型 16 维；注意 head-L/head-R 同向、wing-L/R 同向、body-U/D/C 同向）")
for k in range(P1):
    print(f"  {proto_names[k]:7s}: " + " ".join(f"{q[0,k,d].item():4.2f}" for d in range(D)))

# ---- 单 patch 三项展开 ----
def trace_patch(r, c):
    print(f"\n---- patch (r={r}, c={c}) 区域={LAYOUT[r][c]} ----")
    xv = x_[0, :, r, c]
    b_sq = xv.pow(2).sum().item()
    print(f"  patch 特征 x[0,:,{r},{c}] = " + " ".join(f"{xv[d].item():4.2f}" for d in range(D)))
    print(f"  ‖patch‖² (b_sq) = {b_sq:.3f}")
    print(f"  {'proto':8s} {'a·b(ab)':>9s} {'‖q‖²(a_sq)':>11s} {'dist':>8s} {'sim=-dist':>10s} {'+p_bias':>9s} {'sim+bias':>9s} {'softmax':>8s}")
    logits = []
    rows = []
    for k in range(P1):
        qk = q[0, k]
        ab = (xv * qk).sum().item()
        a_sq = qk.pow(2).sum().item()
        dist = b_sq - 2*ab + a_sq
        s = -dist
        pb = p_bias[0, k, r, c].item()
        logits.append(s + pb)
        rows.append((proto_names[k], ab, a_sq, dist, s, pb, s+pb))
    logits_t = torch.tensor(logits)
    sm = torch.softmax(logits_t, dim=0)
    for k,(nm,ab,a_sq,dist,s,pb,sb) in enumerate(rows):
        print(f"  {nm:8s} {ab:9.3f} {a_sq:11.3f} {dist:8.3f} {s:10.3f} {pb:9.1f} {sb:9.3f} {sm[k].item():8.3f}")

print("\n################ Part 2/3：单 patch 三项展开 + softmax ################")
trace_patch(0, 2)   # 左头 patch：看 head-L vs head-R 怎么被 p_bias 切开
trace_patch(3, 2)   # 身体 patch：看 body-U / body-D / body-C 三方软竞争

# ---- p_bias 8 通道矩阵 ----
show_channels(p_bias, proto_names, b=0, title="Part 3: p_bias 空间先验")

# ---- 软分割图 maps 8 通道矩阵 ----
show_channels(maps, proto_names, b=0, title="Part 3: maps 软分割图")
print("\n[校验] maps 通道和 (b=0)：")
s = maps[0].sum(0)
for r in range(H):
    print("   " + " ".join(f"{s[r,c].item():.2f}" for c in range(W)))
