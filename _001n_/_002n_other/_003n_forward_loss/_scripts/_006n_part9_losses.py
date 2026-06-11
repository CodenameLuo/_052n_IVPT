#!/usr/bin/env python
# Part 9：8 个损失全精确手算。前向构造与 _004n 逐字一致（同 seed → 同 maps_list/m_buffer/outputs）
# 8 个 loss 函数逐字搬自 engine/losses/*；equiv 用仓库真实 rigid_transform 真算
# 跑法（必须从 IVPT 根目录跑，好 import rigid_transform）：
#   cd .../IVPT && _004n_IVPT/bin/python .../_scripts/_006n_part9_losses.py
import sys, os
IVPT_ROOT = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, IVPT_ROOT)
import torch
import torch.nn.functional as F
from utils.data_utils.reversible_affine_transform import rigid_transform
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)
torch.manual_seed(0)

# ===== 前向构造（与 _004n 逐字一致）=====
B, D, H, W, NL = 6, 16, 6, 6, 3
HIGH, LOW = 1.0, 0.10
def archetype(blk):
    v = torch.full((D,), LOW)
    if blk=='head': v[0:4]=HIGH
    if blk=='wing': v[4:8]=HIGH
    if blk=='body': v[8:12]=HIGH
    if blk=='bg':   v[12:16]=0.5; v[0:12]=0.08
    return v
A = {k:archetype(k) for k in ['head','wing','body','bg']}
hd,wg,bd,bg='head','wing','body','bg'
LAYOUT=[[bg,bg,hd,hd,bg,bg],[bg,hd,hd,hd,hd,bg],[wg,wg,bd,bd,wg,wg],
        [wg,wg,bd,bd,wg,wg],[bg,wg,bd,bd,wg,bg],[bg,bg,bd,bd,bg,bg]]
scene=torch.zeros(B,D,H,W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            scene[b,:,r,c]=A[LAYOUT[r][c]]+0.04*torch.randn(D)
LEVEL_FG={0:['head','head','wing','wing','body','body','body'],1:['head','wing','wing','body','body'],
          2:['head','wing','body','body'],3:['head','wing','body']}
def s_lt(c0):return lambda r,c:c<c0
def s_ge(c0):return lambda r,c:c>=c0
def s_rlt(r0):return lambda r,c:r<r0
def s_rge(r0):return lambda r,c:r>=r0
def s_rin(rs):return lambda r,c:r in rs
LEVEL_BIAS={0:[('head',s_lt(3)),('head',s_ge(3)),('wing',s_lt(3)),('wing',s_ge(3)),('body',s_rlt(4)),('body',s_rge(4)),('body',s_rin((3,4)))],
            1:[('head',None),('wing',s_lt(3)),('wing',s_ge(3)),('body',s_rlt(4)),('body',s_rge(4))],
            2:[('head',None),('wing',None),('body',s_rlt(4)),('body',s_rge(4))],
            3:[('head',None),('wing',None),('body',None)]}
def build_q(l):
    q=torch.stack([A[b] for b in LEVEL_FG[l]]+[A['bg']],0)
    return q.unsqueeze(0).expand(B,-1,-1).contiguous()
def build_pb(l):
    specs=LEVEL_BIAS[l]; n=len(specs)+1; pb=torch.zeros(B,n,H,W)
    for idx,(reg,sp) in enumerate(specs):
        for r in range(H):
            for c in range(W):
                if LAYOUT[r][c]==reg and (sp is None or sp(r,c)): pb[:,idx,r,c]=3.0
    for r in range(H):
        for c in range(W):
            if LAYOUT[r][c]=='bg': pb[:,-1,r,c]=3.0
    return pb
x_level={0:scene.clone()}
for l in (1,2,3): x_level[l]=scene+0.03*torch.randn(B,D,H,W)
def compute_xq(x,q):
    ab=torch.einsum('bchw,blc->blhw',x,q)
    b_sq=x.pow(2).sum(1,keepdim=True).expand(-1,q.shape[1],-1,-1).contiguous()
    a_sq=q.pow(2).sum(-1,keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(b_sq-2*ab+a_sq)
def compute_feat(maps,x):
    N=maps.shape[1]
    ohm=F.one_hot(torch.argmax(maps,1),N).permute(0,3,1,2)*maps
    af=(ohm.unsqueeze(1)*x.unsqueeze(2)).contiguous()
    sp=af.sum(dim=(3,4)).permute(0,2,1)
    cm=ohm.sum(dim=(2,3),keepdim=True)
    return sp/(cm+(cm==0).float()).squeeze(-1), cm
p_classifier,p_norm,p_linear=[],[],[]
for l in range(3):
    pc=torch.nn.Linear(D,NL,bias=True)
    with torch.no_grad():
        Wm=torch.zeros(NL,D);Wm[0,0:4]=1;Wm[1,4:8]=1;Wm[2,8:12]=1;pc.weight.copy_(Wm);pc.bias.zero_()
    p_classifier.append(pc);p_norm.append(torch.nn.LayerNorm([NL,D]));p_linear.append(torch.nn.Linear(D,D,bias=True))
m_buffer=[]
for l in range(3):
    q=build_q(l);pb=build_pb(l)
    maps=F.softmax(compute_xq(x_level[l],q)+pb,dim=1)
    q_x,cm=compute_feat(maps.detach(),x_level[l])
    q_c=p_classifier[l](q_x[:,:-1])
    cmask=(cm[:,:-1]==0).squeeze(-1).expand(-1,-1,q_c.shape[-1])
    q_maps=F.softmax(q_c,dim=-1).masked_fill(cmask,1.0/NL)
    qe=q_maps.unsqueeze(-1);fm=maps.flatten(2);qd=qe.detach()
    f_bg=fm[:,-1:];fmw=fm[:,:-1].unsqueeze(2)*qe
    fmc=torch.cat([fmw.sum(1),f_bg],dim=1)
    m_buffer.append(fmc)
def clean_maps(xl):
    out=[]
    for l in range(4):
        q=build_q(l);pb=build_pb(l)
        m=F.softmax(compute_xq(xl[l],q)+pb,dim=1); m=m+1e-6; m=m/m.sum(1,keepdim=True)
        out.append(m)
    return out
maps_list=clean_maps(x_level)
maps_final=maps_list[-1]
m_buffer.append(maps_final.flatten(2))
modulation=torch.nn.LayerNorm([D,NL+1]); fc=torch.nn.Linear(D,10,bias=False)
all_features=compute_feat(maps_final,x_level[3])[0].permute(0,2,1)
all_features_mod=modulation(all_features)
scores=fc(all_features_mod[...,:-1].permute(0,2,1)).permute(0,2,1)
outputs=scores.mean(-1)

# ===== 8 个 loss 函数（逐字搬自 engine/losses/*）=====
def consistency_loss(pred,target,eps=1e-10):
    pred=pred+eps;target=target+eps
    kl=target*torch.log(target/pred); kl=kl.sum(dim=1); kl=kl.mean(dim=1)
    return kl.mean()
def orthogonality_loss(all_features):
    nf=F.normalize(all_features,dim=1); tl=all_features.shape[-1]
    sim=torch.matmul(nf.permute(0,2,1).contiguous(),nf)
    sim=torch.sub(sim,torch.eye(tl))
    return torch.mean(torch.square(sim))
def pixel_wise_entropy_loss(maps):
    maps=maps.float().clamp(min=1e-6,max=1.0); maps=maps/maps.sum(dim=1,keepdim=True)
    ent=torch.distributions.categorical.Categorical(probs=maps.permute(0,2,3,1).contiguous()).entropy()
    return ent.mean()
def presence_original(maps):
    return 1-F.adaptive_max_pool2d(F.avg_pool2d(maps,3,stride=1),1).flatten(start_dim=1).max(dim=0)[0].mean()
def tv_mean(img):
    d1=img[...,1:,:]-img[...,:-1,:]; d2=img[...,:,1:]-img[...,:,:-1]
    score=d1.abs().sum([1,2,3])+d2.abs().sum([1,2,3])
    return score.sum()/(img.shape[0]*img.shape[2]*img.shape[3])
_epl_cache={}
def enforced_presence(maps):
    ap=F.avg_pool2d(maps,3,stride=1)
    key=ap.shape[2]
    if key not in _epl_cache:
        gx,gy=torch.meshgrid(torch.arange(ap.shape[2]),torch.arange(ap.shape[3]),indexing='ij')
        gx=gx.unsqueeze(0).unsqueeze(0).float();gy=gy.unsqueeze(0).unsqueeze(0).float()
        gx=(gx/gx.max())*2-1;gy=(gy/gy.max())*2-1
        mask=gx**2+gy**2;mask=mask/mask.max();_epl_cache[key]=mask
    mask=_epl_cache[key]
    mbg=(ap*mask)[:,-1,:,:]
    mp=F.adaptive_max_pool2d(mbg,1).flatten(start_dim=0)
    return F.binary_cross_entropy(mp,torch.ones_like(mp))
from engine.losses.equivarance_loss import equivariance_loss   # 真·仓库实现

# ===== equiv 第二趟前向：变换特征网格 → equiv_maps =====
angle,translate,scale,shear=15.0,[1.0,0.5],1.0,0.0
x_level_t={l:rigid_transform(img=x_level[l],angle=angle,translate=translate,scale=scale,shear=shear,invert=False) for l in range(4)}
equiv_maps=clean_maps(x_level_t)
source_dummy=torch.zeros(B,3,H,W)   # 只用到 shape[-1]=6 → translate 缩放=恒等

# ===== 逐个算 8 个 loss（权重=仓库默认）=====
targets=torch.tensor([2,2,2,2,2,2])
loss_fn=torch.nn.CrossEntropyLoss()
L_CLASS=L_PRES=L_EQUIV=L_ORTH=L_TV=L_PIX=1.0; L_ENF=2.0

print("############ Part 9：8 个损失 ############")
# 1 分类
loss_classification=loss_fn(outputs,targets)*L_CLASS
print("\n[1] classification（CE, target=cls2）")
print("  outputs(b=0)=",[round(v,3) for v in outputs[0].tolist()])
print("  softmax(b=0)[cls2]=",round(F.softmax(outputs[0],0)[2].item(),4),"  -ln=",round(-torch.log(F.softmax(outputs[0],0)[2]).item(),4))
print("  loss_classification =",round(loss_classification.item(),4))

# 2 consistency
target_c=m_buffer[-1].detach()
kls=[consistency_loss(m,target_c) for m in m_buffer[:-1]]
loss_consistency=sum(kls)/len(m_buffer[:-1])
print("\n[2] consistency（KL，前 3 层粗图 vs 收尾图）")
for i,k in enumerate(kls): print(f"  m_buffer[{i}] vs m_buffer[-1] : KL = {k.item():.5f}")
print("  loss_consistency(平均) =",round(loss_consistency.item(),5))

# 3 tv
tvs=[tv_mean(m) for m in maps_list]
loss_tv=sum([t*L_TV for t in tvs])/len(maps_list)
print("\n[3] total variation（每张图相邻像素差）")
for i,t in enumerate(tvs): print(f"  maps_list[{i}] (ch={maps_list[i].shape[1]}) : TV = {t.item():.4f}")
print("  loss_tv(平均) =",round(loss_tv.item(),4))

# 4 presence
prs=[presence_original(m[:,:-1]) for m in maps_list]
loss_presence=sum([p*L_PRES for p in prs])/len(maps_list)
print("\n[4] presence（1 - 各前景部件空间最大池化的均值）")
for i,p in enumerate(prs): print(f"  maps_list[{i}] : presence = {p.item():.4f}")
print("  loss_presence(平均) =",round(loss_presence.item(),4))

# 5 orth
loss_orth=orthogonality_loss(all_features_mod)*L_ORTH
nf=F.normalize(all_features_mod,dim=1)
G=torch.matmul(nf.permute(0,2,1),nf)[0]
print("\n[5] orthogonality（部件特征两两余弦，减单位阵，平方均值）")
print("  余弦矩阵 G (b=0) 4×4：")
for r in range(4): print("   "+" ".join(f"{G[r,c].item():6.3f}" for c in range(4)))
print("  loss_orth =",round(loss_orth.item(),4))

# 6 equiv
eqs=[equivariance_loss(maps_list[i],equiv_maps[i],source_dummy,NL,translate,angle,scale,shear=0.0) for i in range(4)]
loss_equiv=sum([e*L_EQUIV for e in eqs])/len(maps_list)
print("\n[6] equivariance（变换图找部件→反变换→比余弦, 1-cos）真算")
for i,e in enumerate(eqs): print(f"  maps_list[{i}] : 1-cos = {e.item():.4f}")
print("  loss_equiv(平均) =",round(loss_equiv.item(),4))

# 7 enforced_presence
enfs=[enforced_presence(m) for m in maps_list]
loss_enforced=sum([e*L_ENF for e in enfs])/len(maps_list)
print("\n[7] enforced_presence（径向 mask×背景通道→空间max→BCE→1）权重2.0")
for i,e in enumerate(enfs): print(f"  maps_list[{i}] : BCE = {e.item():.4f}")
print("  loss_enforced_presence(平均×2) =",round(loss_enforced.item(),4))

# 8 pixel_wise_entropy
pes=[pixel_wise_entropy_loss(m) for m in maps_list]
loss_pix=sum([p*L_PIX for p in pes])/len(maps_list)
print("\n[8] pixel_wise_entropy（每 patch 在各部件上的分布熵, 全图平均）")
for i,p in enumerate(pes): print(f"  maps_list[{i}] : entropy = {p.item():.4f}")
print("  loss_pixel_wise_entropy(平均) =",round(loss_pix.item(),4))

total=loss_consistency+loss_presence+loss_classification+loss_orth+loss_equiv+loss_tv+loss_enforced+loss_pix
print("\n############ 8 项求和 ############")
print(f"  classification {loss_classification.item():.4f} | consistency {loss_consistency.item():.5f} | tv {loss_tv.item():.4f} | presence {loss_presence.item():.4f}")
print(f"  orth {loss_orth.item():.4f} | equiv {loss_equiv.item():.4f} | enforced {loss_enforced.item():.4f} | pixel_entropy {loss_pix.item():.4f}")
print(f"  >>> total loss = {total.item():.4f}")
