#!/usr/bin/env python
# loss_detail.md 第 3 批：正交 orth + 等变 equiv 的逐元素中间量
# orth 从 forward_dump.pt 取 all_features_mod；equiv 重建变换前向 + 真·rigid_transform
# 跑法（须从 IVPT 根目录跑以 import）：cd .../IVPT && _004n_IVPT/bin/python .../_scripts/_009n_lossdetail_5_6.py
import sys
IVPT_ROOT="/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, IVPT_ROOT)
DUMP="/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/_002n_other/_003n_forward_loss/_scripts/forward_dump.pt"
import torch
import torch.nn.functional as F
from utils.data_utils.reversible_affine_transform import rigid_transform
from engine.losses.equivarance_loss import equivariance_loss
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)

def grid(t, fmt="{:6.3f}"):
    for r in range(t.shape[0]):
        print("    " + " ".join(fmt.format(t[r,c].item()) for c in range(t.shape[1])))

# ============================================================
# 损失5：正交 orth
print("################ 损失5 正交 orth ################")
d = torch.load(DUMP)
afm = d['all_features_mod']     # [6,16,4]
print("计算对象：all_features_mod [6,16,4]（调制后部件特征，含 bg）")
normed = F.normalize(afm, dim=1)               # 按 16 维归一化成单位向量
G = torch.matmul(normed.permute(0,2,1), normed)   # [6,4,4] 余弦矩阵
print("\n① 余弦矩阵 G = normedᵀ·normed (b=0) 4×4（行列=头/翅/身/bg）：")
grid(G[0])
I = torch.eye(4)
print("\n② G − I（对角线减 1，自己对自己不罚）：")
grid((G[0]-I))
print("\n③ (G − I)²（逐元素平方）：")
grid((G[0]-I)**2)
print(f"\n④ mean((G−I)²)：b=0 的 16 格均值 = {((G[0]-I)**2).mean().item():.4f}")
print(f"   全 6 batch 的均值 = loss_orth = {((G-I)**2).mean().item():.4f}")

# ============================================================
# 损失6：等变 equiv —— 重建变换前向
print("\n\n################ 损失6 等变 equiv ################")
B,Dd,H,W,NL=6,16,6,6,3
HIGH,LOW=1.0,0.10
torch.manual_seed(0)
def arche(blk):
    v=torch.full((Dd,),LOW)
    if blk=='head':v[0:4]=HIGH
    if blk=='wing':v[4:8]=HIGH
    if blk=='body':v[8:12]=HIGH
    if blk=='bg':v[12:16]=0.5;v[0:12]=0.08
    return v
A={k:arche(k) for k in ['head','wing','body','bg']}
hd,wg,bd,bg='head','wing','body','bg'
LAYOUT=[[bg,bg,hd,hd,bg,bg],[bg,hd,hd,hd,hd,bg],[wg,wg,bd,bd,wg,wg],[wg,wg,bd,bd,wg,wg],[bg,wg,bd,bd,wg,bg],[bg,bg,bd,bd,bg,bg]]
scene=torch.zeros(B,Dd,H,W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            scene[b,:,r,c]=A[LAYOUT[r][c]]+0.04*torch.randn(Dd)
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
    q=torch.stack([A[b] for b in LF[l]]+[A['bg']],0); return q.unsqueeze(0).expand(B,-1,-1).contiguous()
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
for l in (1,2,3):xl[l]=scene+0.03*torch.randn(B,Dd,H,W)
def cxq(x,q):
    ab=torch.einsum('bchw,blc->blhw',x,q)
    bs=x.pow(2).sum(1,keepdim=True).expand(-1,q.shape[1],-1,-1).contiguous()
    asq=q.pow(2).sum(-1,keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(bs-2*ab+asq)
def clean(xlv):
    out=[]
    for l in range(4):
        m=F.softmax(cxq(xlv[l],bq(l))+bpb(l),dim=1);m=m+1e-6;m=m/m.sum(1,keepdim=True);out.append(m)
    return out
maps_list=clean(xl)
angle,translate,scale,shear=15.0,[1.0,0.5],1.0,0.0
xlt={l:rigid_transform(img=xl[l],angle=angle,translate=translate,scale=scale,shear=shear,invert=False) for l in range(4)}
equiv_maps=clean(xlt)
source=torch.zeros(B,3,H,W)

print("计算对象：maps_list（原图部件图）+ equiv_maps（变换图的部件图，第2次前向）")
print(f"仿射参数：angle={angle}°, translate={translate}, scale={scale}（6×6 网格上）")
print("\n以 maps_list[-1] 的【头】通道 (b=0) 三态对比：")
print("\n  (a) 原图部件图 maps_list[-1][头]：")
grid(maps_list[-1][0,0],"{:5.2f}")
print("\n  (b) 变换图部件图 equiv_maps[-1][头]（头 blob 被仿射移走了）：")
grid(equiv_maps[-1][0,0],"{:5.2f}")
rot_back=rigid_transform(img=equiv_maps[-1],angle=angle,translate=translate,scale=scale,shear=shear,invert=True)
print("\n  (c) 反变换回来 rot_back[头]（理想应=(a)，但边界丢信息+p_bias不等变→对不齐）：")
grid(rot_back[0,0],"{:5.2f}")

# 头通道 orig vs rot_back 的余弦
o=maps_list[-1][0,0].flatten(); rb=rot_back[0,0].flatten()
cos_head=F.cosine_similarity(o,rb,dim=0)
print(f"\n  头通道 cos(原图, 反变换) = {cos_head.item():.4f}")

print("\n每张图各自 1−cos（对 3 个前景部件平均；真·仓库 equivariance_loss）：")
eqs=[equivariance_loss(maps_list[i],equiv_maps[i],source,NL,translate,angle,scale,shear=0.0) for i in range(4)]
for i,e in enumerate(eqs):print(f"  maps_list[{i}] : 1−cos = {e.item():.4f}")
print(f"  loss_equiv = 平均 = {sum(eqs).item()/4:.4f}")
