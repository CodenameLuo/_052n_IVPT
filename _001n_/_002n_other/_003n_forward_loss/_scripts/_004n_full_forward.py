#!/usr/bin/env python
# 完整前向：4 个层级(注入 L0/L1/L2 + 收尾读出) + 分类，忠实复刻 forward()，只抽象 block
# 抽象点：每层的特征图 x_level[i] = 场景 + 小扰动(代表 block 变换)，结构不变好让矩阵可读
# 产出全部 buffer / q_x / f_maps / scores，供 Part 6/7/8/9 取数
# 跑法：_004n_IVPT/bin/python _004n_full_forward.py

import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)
torch.manual_seed(0)

B, D, H, W = 6, 16, 6, 6
NL = 3   # num_landmarks 粗部件
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
# 场景特征图（必须最先抽，保证 L0 与 _001n~_003n 完全一致）
scene = torch.zeros(B, D, H, W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            scene[b, :, r, c] = A[LAYOUT[r][c]] + 0.04 * torch.randn(D)

# 每层的细前景原型（语义逐层变粗：7 -> 5 -> 4 -> 3）
LEVEL_FG = {
    0: ['head','head','wing','wing','body','body','body'],   # 7 fg：头L 头R 翅L 翅R 身上 身下 身中
    1: ['head','wing','wing','body','body'],                 # 5 fg：头  翅L 翅R 身上 身下
    2: ['head','wing','body','body'],                        # 4 fg：头  翅  身上 身下
    3: ['head','wing','body'],                               # 3 fg：头  翅  身（收尾=粗部件本身）
}
LEVEL_NAMES = {
    0: ['head-L','head-R','wing-L','wing-R','body-U','body-D','body-C'],
    1: ['head','wing-L','wing-R','body-U','body-D'],
    2: ['head','wing','body-U','body-D'],
    3: ['head','wing','body'],
}
# 每层每个 fg 原型的空间先验区域 (region, split)；split=None 表示占满整个 region
def split_lt(col):  return lambda r,c: c < col
def split_ge(col):  return lambda r,c: c >= col
def split_rlt(row): return lambda r,c: r < row
def split_rge(row): return lambda r,c: r >= row
def split_rin(rows):return lambda r,c: r in rows
LEVEL_BIAS = {
    0: [('head',split_lt(3)),('head',split_ge(3)),('wing',split_lt(3)),('wing',split_ge(3)),
        ('body',split_rlt(4)),('body',split_rge(4)),('body',split_rin((3,4)))],
    1: [('head',None),('wing',split_lt(3)),('wing',split_ge(3)),('body',split_rlt(4)),('body',split_rge(4))],
    2: [('head',None),('wing',None),('body',split_rlt(4)),('body',split_rge(4))],
    3: [('head',None),('wing',None),('body',None)],
}

def build_q(level):
    fg = LEVEL_FG[level]
    q = torch.stack([A[b] for b in fg] + [A['bg']], 0)        # [n_pro, 16]
    return q.unsqueeze(0).expand(B, -1, -1).contiguous()

def build_pbias(level):
    specs = LEVEL_BIAS[level]
    n = len(specs) + 1                                        # +1 bg
    pb = torch.zeros(B, n, H, W)
    for idx, (reg, split) in enumerate(specs):
        for r in range(H):
            for c in range(W):
                if LAYOUT[r][c] == reg and (split is None or split(r, c)):
                    pb[:, idx, r, c] = 3.0
    for r in range(H):                                        # bg 通道
        for c in range(W):
            if LAYOUT[r][c] == 'bg':
                pb[:, -1, r, c] = 3.0
    return pb

# 每层特征图 = 场景 + 小扰动(L0 无扰动，保证和前几个 Part 完全一致)
x_level = {0: scene.clone()}
for l in (1, 2, 3):
    x_level[l] = scene + 0.03 * torch.randn(B, D, H, W)

def compute_xq(x, q):
    ab = torch.einsum('bchw,blc->blhw', x, q)
    b_sq = x.pow(2).sum(1, keepdim=True).expand(-1, q.shape[1], -1, -1).contiguous()
    a_sq = q.pow(2).sum(-1, keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(b_sq - 2*ab + a_sq)

def compute_feat(maps, x):
    N = maps.shape[1]
    one_hot_map = F.one_hot(torch.argmax(maps, 1), N).permute(0,3,1,2) * maps
    all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
    sum_pool = all_features.sum(dim=(3,4)).permute(0,2,1)
    count_map = one_hot_map.sum(dim=(2,3), keepdim=True)
    count_map_ = count_map + (count_map==0).float()
    return sum_pool / count_map_.squeeze(-1), count_map

# 三个注入层各一套 p_classifier(手设读块) / p_norm / p_linear
p_classifier, p_norm, p_linear = [], [], []
for l in range(3):
    pc = torch.nn.Linear(D, NL, bias=True)
    with torch.no_grad():
        Wm = torch.zeros(NL, D); Wm[0,0:4]=1.0; Wm[1,4:8]=1.0; Wm[2,8:12]=1.0
        pc.weight.copy_(Wm); pc.bias.zero_()
    p_classifier.append(pc)
    p_norm.append(torch.nn.LayerNorm([NL, D]))
    p_linear.append(torch.nn.Linear(D, D, bias=True))

# ============================================================
# 注入层 L0/L1/L2
m_buffer, qm_buffer = [], []
coarse_feat_store, fmaps_store, qmaps_store, maps_store = {}, {}, {}, {}
for l in range(3):
    q = build_q(l); pb = build_pbias(l)
    maps = F.softmax(compute_xq(x_level[l], q) + pb, dim=1)         # [B, n_pro, 6,6]
    maps_store[l] = maps
    q_x, count_map = compute_feat(maps.detach(), x_level[l])        # [B,n_pro,16]
    q_c = p_classifier[l](q_x[:, :-1])                             # [B,P,3]
    count_mask = (count_map[:, :-1]==0).squeeze(-1).expand(-1,-1,q_c.shape[-1])
    q_maps = F.softmax(q_c, dim=-1).masked_fill(count_mask, 1.0/NL) # [B,P,3]
    qm_buffer.append(q_maps); qmaps_store[l] = q_maps

    # === 两路聚合（detach 不对称）===
    q_maps_expanded = q_maps.unsqueeze(-1)                          # [B,P,3,1]
    f_maps = maps.flatten(2)                                        # [B,n_pro,36]
    q_maps_detach = q_maps_expanded.detach()
    q_x_w = q_x[:, :-1].unsqueeze(2) * q_maps_detach                # [B,P,3,16]
    q_x_coarse = q_x_w.sum(dim=1)                                   # [B,3,16]
    q_maps_sum = q_maps_detach.sum(1).squeeze(1)                    # [B,3,1]
    q_x_coarse = q_x_coarse / q_maps_sum                            # [B,3,16] 加权平均
    f_bg = f_maps[:, -1:]                                           # [B,1,36]
    f_maps_w = f_maps[:, :-1].unsqueeze(2) * q_maps_expanded        # [B,P,3,36]
    f_maps_c = f_maps_w.sum(dim=1)                                  # [B,3,36] 加权和
    f_maps_c = torch.cat([f_maps_c, f_bg], dim=1)                   # [B,4,36]
    m_buffer.append(f_maps_c); fmaps_store[l] = f_maps_c
    coarse_feat_store[l] = q_x_coarse

    prompt = p_linear[l](p_norm[l](q_x_coarse))                     # [B,3,16] 归一化+投影=prompt
    if l == 0: prompt_store = prompt

# ============================================================
# 收尾读出：maps_list 干净重算（含 +1e-6），最终图 → 分类
maps_list = []
for l in range(4):
    q = build_q(l); pb = build_pbias(l)
    maps = F.softmax(compute_xq(x_level[l], q) + pb, dim=1)
    maps = maps + 1e-6
    maps = maps / maps.sum(dim=1, keepdim=True)
    maps_list.append(maps)

maps_final = maps_list[-1]                                          # [B,4,6,6]
m_buffer.append(maps_final.flatten(2))                             # m_buffer 第 4 条

modulation = torch.nn.LayerNorm([D, NL+1])
fc = torch.nn.Linear(D, 10, bias=False)
all_features = compute_feat(maps_final, x_level[3])[0].permute(0,2,1)   # [B,16,4]
all_features_mod = modulation(all_features)                            # [B,16,4]
scores = fc(all_features_mod[..., :-1].permute(0,2,1)).permute(0,2,1)  # [B,10,3]
outputs = scores.mean(-1)                                              # [B,10]

# ============================================================
def show_grid(t, b=0, title=""):
    print(f"\n[{title}]")
    for r in range(H):
        print("   " + " ".join(f"{t[b,r,c].item():5.2f}" for c in range(W)))

print("############### Part 6：L0 两路聚合 ###############")
print(f"特征路 q_x_coarse [B,3,16]={list(coarse_feat_store[0].shape)} (b=0)：3 个粗部件特征")
for k, nm in enumerate(['粗0头','粗1翅','粗2身']):
    print(f"  {nm}: " + " ".join(f"{coarse_feat_store[0][0,k,d].item():4.2f}" for d in range(D)))
print(f"\n空间图路 f_maps [B,4,36]={list(fmaps_store[0].shape)} → reshape[B,4,6,6] (b=0)：3 粗+bg 空间图")
fm = fmaps_store[0].view(B, 4, H, W)
for k, nm in enumerate(['粗0头','粗1翅','粗2身','bg']):
    show_grid(fm[:, k], 0, f"f_maps {nm}")
print(f"\nprompt = p_linear(p_norm(q_x_coarse))  [B,3,16]={list(prompt_store.shape)} (b=0，p_linear 随机权重)：")
for k, nm in enumerate(['粗0头','粗1翅','粗2身']):
    print(f"  {nm}: " + " ".join(f"{prompt_store[0,k,d].item():5.2f}" for d in range(D)))
print(f"\n拼回序列：x[B,41,16] + prompt[B,3,16] = [B,44,16] → block → [B,44,16]（形状）")

print("\n############### Part 7：多层 buffer 累积 ###############")
print("各层原型数 / 软分割图通道 / 路由 q_maps 形状 / f_maps 形状：")
for l in range(3):
    print(f"  L{l}: n_pro={maps_store[l].shape[1]}  maps={list(maps_store[l].shape)}  "
          f"q_maps={list(qmaps_store[l].shape)}  f_maps(粗)={list(fmaps_store[l].shape)}")
print(f"  收尾: n_pro={maps_final.shape[1]}  maps_final={list(maps_final.shape)}")
print(f"\nm_buffer 共 {len(m_buffer)} 条（前 3 条=注入层聚合粗图，第 4 条=收尾最终图），每条形状：")
for i, fmb in enumerate(m_buffer):
    print(f"  m_buffer[{i}] = {list(fmb.shape)}")
print(f"qm_buffer 共 {len(qm_buffer)} 条，形状：" + ", ".join(str(list(t.shape)) for t in qm_buffer))

# L1/L2 的粗 f_maps（头通道）看一眼，确认跨层相似
print("\nL1 / L2 的 f_maps「粗0头」通道 (b=0)，应和 L0 头通道相似（跨层一致 = consistency 目标）：")
for l in (1, 2):
    show_grid(fmaps_store[l].view(B,4,H,W)[:, 0], 0, f"L{l} f_maps 粗0头")

# 存盘给后续 Part 8/9
torch.save({'m_buffer':[t for t in m_buffer], 'qm_buffer':[t for t in qm_buffer],
            'maps_list':[t for t in maps_list], 'all_features_mod':all_features_mod,
            'scores':scores, 'outputs':outputs, 'x_level3':x_level[3]},
           'forward_dump.pt')
print("\n[已存 forward_dump.pt 供 Part 8/9]")
print(f"scores [B,10,3]={list(scores.shape)}  outputs [B,10]={list(outputs.shape)}")
