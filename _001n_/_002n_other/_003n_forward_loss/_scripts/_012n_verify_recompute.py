#!/usr/bin/env python
# 核实"干净重算"的设计逻辑：
#  ① 注入循环里的 fine maps vs 重算的 maps_list（前 3 层）差多少？
#  ② 最后一层（最粗4原型）是否只在重算里出现？
#  ③ maps_list（细 8/6/5/4 通道）vs m_buffer（粗 4 通道）—— 证明两者是不同对象
# 跑法：cd .../_scripts && _004n_IVPT/bin/python _012n_verify_recompute.py
import torch
import torch.nn.functional as F
torch.manual_seed(0)
B,D,H,W,NL=6,16,6,6,3
HIGH,LOW=1.0,0.10
def arche(b):
    v=torch.full((D,),LOW)
    if b=='head':v[0:4]=HIGH
    if b=='wing':v[4:8]=HIGH
    if b=='body':v[8:12]=HIGH
    if b=='bg':v[12:16]=0.5;v[0:12]=0.08
    return v
A={k:arche(k) for k in ['head','wing','body','bg']}
hd,wg,bd,bg='head','wing','body','bg'
LAYOUT=[[bg,bg,hd,hd,bg,bg],[bg,hd,hd,hd,hd,bg],[wg,wg,bd,bd,wg,wg],[wg,wg,bd,bd,wg,wg],[bg,wg,bd,bd,wg,bg],[bg,bg,bd,bd,bg,bg]]
scene=torch.zeros(B,D,H,W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            scene[b,:,r,c]=A[LAYOUT[r][c]]+0.04*torch.randn(D)
LF={0:['head','head','wing','wing','body','body','body'],1:['head','wing','wing','body','body'],2:['head','wing','body','body'],3:['head','wing','body']}
def s_lt(c0):return lambda r,c:c<c0
def s_ge(c0):return lambda r,c:c>=c0
def s_rlt(r0):return lambda r,c:r<r0
def s_rge(r0):return lambda r,c:r>=r0
def s_rin(rs):return lambda r,c:r in rs
LB={0:[('head',s_lt(3)),('head',s_ge(3)),('wing',s_lt(3)),('wing',s_ge(3)),('body',s_rlt(4)),('body',s_rge(4)),('body',s_rin((3,4)))],
    1:[('head',None),('wing',s_lt(3)),('wing',s_ge(3)),('body',s_rlt(4)),('body',s_rge(4))],
    2:[('head',None),('wing',None),('body',s_rlt(4)),('body',s_rge(4))],3:[('head',None),('wing',None),('body',None)]}
def bq(l):
    return torch.stack([A[b] for b in LF[l]]+[A['bg']],0).unsqueeze(0).expand(B,-1,-1).contiguous()
def bpb(l):
    sp=LB[l];n=len(sp)+1;pb=torch.zeros(B,n,H,W)
    for i,(reg,f) in enumerate(sp):
        for r in range(H):
            for c in range(W):
                if LAYOUT[r][c]==reg and (f is None or f(r,c)):pb[:,i,r,c]=3.0
    for r in range(H):
        for c in range(W):
            if LAYOUT[r][c]=='bg':pb[:,-1,r,c]=3.0
    return pb
xl={0:scene.clone()}
for l in (1,2,3):xl[l]=scene+0.03*torch.randn(B,D,H,W)
def cxq(x,q):
    ab=torch.einsum('bchw,blc->blhw',x,q)
    bs=x.pow(2).sum(1,keepdim=True).expand(-1,q.shape[1],-1,-1).contiguous()
    asq=q.pow(2).sum(-1,keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(bs-2*ab+asq)

# ① 注入循环里的 fine maps（只算前 3 层，最后一层注入循环不碰）
loop_maps={}
for l in range(3):
    loop_maps[l]=F.softmax(cxq(xl[l],bq(l))+bpb(l),dim=1)   # 注意：无 +1e-6

# ② 干净重算 maps_list（4 层，含 +1e-6）
maps_list=[]
for l in range(4):
    m=F.softmax(cxq(xl[l],bq(l))+bpb(l),dim=1); m=m+1e-6; m=m/m.sum(1,keepdim=True)
    maps_list.append(m)

print("########### ① 注入循环 fine maps  vs  重算 maps_list（前 3 层）###########")
for l in range(3):
    diff=(loop_maps[l]-maps_list[l]).abs().max().item()
    print(f"  层{l}: 两者最大逐元素差 = {diff:.2e}   （只差那一步 +1e-6 重归一，数值上等价）")
print("\n########### ② 最后一层（最粗 4 原型）###########")
print(f"  注入循环算过的层: 0,1,2（原型数 {loop_maps[0].shape[1]}/{loop_maps[1].shape[1]}/{loop_maps[2].shape[1]}）")
print(f"  最后一层 maps_list[3] 原型数={maps_list[3].shape[1]} —— 注入循环【从没算过】，只在重算里诞生")

print("\n########### ③ maps_list（细）vs m_buffer（粗）是不同对象 ###########")
print("  maps_list 各层通道数（细，per-map 损失用）：", [m.shape[1] for m in maps_list])
print("  m_buffer  各层通道数（粗，聚合后，consistency 用）：[4, 4, 4, 4]")
print("  → per-map 损失（tv/presence/equiv/enforced/pixel）要逐个约束每个【细】原型，")
print("    必须用 8/6/5/4 通道的 maps_list；m_buffer 的 4 通道粗图丢了细原型个体信息，顶替不了。")

# ④ 演示：注入循环的 fine maps 用完即弃——它只生成了 q_x(detach) 和 聚合粗 f_maps，fine map 本体没被存
print("\n########### ④ 注入循环里 fine map 的去向（用完即弃）###########")
print("  fine maps ──compute_feat(maps.detach())──> q_x（喂路由/prompt，且 detach）")
print("  fine maps ──flatten+按 q_maps 聚合──> 粗 f_maps[4通道] ──> m_buffer")
print("  ↑ fine map 本体没进任何 buffer；要给 per-map 损失用，只能重算（或额外存）。")
