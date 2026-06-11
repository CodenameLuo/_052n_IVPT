#!/usr/bin/env python
# Part 6 精修：两路聚合的逐项算术 + detach 梯度逻辑的 autograd 实验
# 重建 L0（与 _003n/_004n 一致），拿到 fine q_x[6,8,16] / q_maps[6,7,3] / 软分割图 maps[6,8,6,6]
# 跑法：cd .../_scripts && _004n_IVPT/bin/python _011n_part6_detail.py
import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=4, linewidth=200)
torch.manual_seed(0)

B, D, H, W, NL, P1 = 6, 16, 6, 6, 3, 8
HIGH, LOW = 1.0, 0.10
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
x_=torch.zeros(B,D,H,W)
for b in range(B):
    for r in range(H):
        for c in range(W):
            x_[b,:,r,c]=A[LAYOUT[r][c]]+0.04*torch.randn(D)
specs=[('head','head-L'),('head','head-R'),('wing','wing-L'),('wing','wing-R'),('body','body-U'),('body','body-D'),('body','body-C'),('bg','bg')]
q=torch.stack([A[b] for b,_ in specs],0).unsqueeze(0).expand(B,-1,-1).contiguous()
names=[n for _,n in specs]; fine_names=names[:-1]
pb=torch.zeros(B,P1,H,W)
def sb(i,m,v=3.0):
    for r in range(H):
        for c in range(W):
            if m(r,c):pb[:,i,r,c]=v
sb(0,lambda r,c:LAYOUT[r][c]=='head' and c<3); sb(1,lambda r,c:LAYOUT[r][c]=='head' and c>=3)
sb(2,lambda r,c:LAYOUT[r][c]=='wing' and c<3); sb(3,lambda r,c:LAYOUT[r][c]=='wing' and c>=3)
sb(4,lambda r,c:LAYOUT[r][c]=='body' and r<4); sb(5,lambda r,c:LAYOUT[r][c]=='body' and r>=4)
sb(6,lambda r,c:LAYOUT[r][c]=='body' and r in (3,4)); sb(7,lambda r,c:LAYOUT[r][c]=='bg')
def cxq(x,q):
    ab=torch.einsum('bchw,blc->blhw',x,q)
    bs=x.pow(2).sum(1,keepdim=True).expand(-1,q.shape[1],-1,-1).contiguous()
    asq=q.pow(2).sum(-1,keepdim=True).expand(-1,-1,x.shape[-2]*x.shape[-1]).view(x.shape[0],q.shape[1],x.shape[-2],x.shape[-1])
    return -(bs-2*ab+asq)
def cfeat(maps,x):
    N=maps.shape[1];ohm=F.one_hot(torch.argmax(maps,1),N).permute(0,3,1,2)*maps
    af=(ohm.unsqueeze(1)*x.unsqueeze(2)).contiguous();sp=af.sum(dim=(3,4)).permute(0,2,1)
    cm=ohm.sum(dim=(2,3),keepdim=True);return sp/(cm+(cm==0).float()).squeeze(-1),cm
maps=F.softmax(cxq(x_,q)+pb,dim=1)               # [6,8,6,6]
q_x,cm=cfeat(maps.detach(),x_)                   # [6,8,16]
pc=torch.nn.Linear(D,NL,bias=True)
with torch.no_grad():
    Wm=torch.zeros(NL,D);Wm[0,0:4]=1;Wm[1,4:8]=1;Wm[2,8:12]=1;pc.weight.copy_(Wm);pc.bias.zero_()
q_c=pc(q_x[:,:-1]);cmask=(cm[:,:-1]==0).squeeze(-1).expand(-1,-1,q_c.shape[-1])
q_maps=F.softmax(q_c,dim=-1).masked_fill(cmask,1.0/NL)   # [6,7,3]

# ============================================================
print("########### 6.3 特征路逐项：q_x_coarse[粗0头] 怎么由 7 个细原型加权平均出来 ###########")
w = q_maps[0,:,0]                                  # 7 个细原型 → 粗0头 的路由权重
print("路由权重 q_maps[:,头] (7):", " ".join(f"{name}={w[k].item():.3f}" for k,name in enumerate(fine_names)))
denom = w.sum()
print(f"分母 Σ权重 = {denom.item():.3f}   （注意含 body-C 的 0.333）")
print("\n以特征通道 ch0 为例，逐项 (细原型 ch0 值 × 路由权重)：")
num0 = 0.0
for k,name in enumerate(fine_names):
    val=q_x[0,k,0].item(); ww=w[k].item(); num0+=val*ww
    print(f"  {name:7s}: {val:5.2f} × {ww:.3f} = {val*ww:.4f}")
print(f"  分子 Σ = {num0:.4f}")
print(f"  q_x_coarse[头,ch0] = 分子/分母 = {num0:.4f}/{denom.item():.3f} = {num0/denom.item():.4f}")
print(f"  ★若没有空原型 body-C：分母={denom.item()-w[6].item():.3f}，结果={num0/(denom.item()-w[6].item()):.4f}（≈0.94，body-C 把它稀释到 0.81）")

# ============================================================
print("\n########### 6.4 空间图路逐项：f_maps[粗0头] 在 (3,2) 怎么由 7 个细原型加权求和 ###########")
fmap=maps.flatten(2)                               # [6,8,36]
pos=3*W+2
print(f"位置 (3,2) = 第 {pos} 个 patch。各细原型在此的软分割值 × 路由到头的权重：")
ssum=0.0
for k,name in enumerate(fine_names):
    sv=fmap[0,k,pos].item(); ww=w[k].item(); ssum+=sv*ww
    tag=" ← body-C 泄漏主力" if name=='body-C' else ""
    print(f"  {name:7s}: 软值{sv:.3f} × 权重{ww:.3f} = {sv*ww:.4f}{tag}")
print(f"  f_maps[头,(3,2)] = Σ = {ssum:.4f}   （不除，加权求和）")
print(f"  其中 body-C 单独贡献 {fmap[0,6,pos].item():.3f}×{w[6].item():.3f}={fmap[0,6,pos].item()*w[6].item():.4f}，占了大头 → 这就是 0.18 泄漏的来源")

# ============================================================
print("\n########### 6.5 detach 梯度逻辑：autograd 实验（打印 q_maps.grad）###########")
qx7 = q_x[:, :-1].detach().clone().requires_grad_(True)   # 7 个细特征，当可学习叶子
qm  = q_maps.detach().clone().requires_grad_(True)        # 路由，当可学习叶子
f_fine = fmap[:, :-1].detach().clone()                    # 7 个细软图（常量）

# --- 路A 特征路：detach(qm)，形成 prompt，代理"分类 loss" ---
qm_d = qm.detach()
qxc = (qx7.unsqueeze(2)*qm_d.unsqueeze(-1)).sum(1) / qm_d.unsqueeze(-1).sum(1)   # [6,3,16] 粗特征
lossA = qxc.sum()                                                  # 代理：经 prompt 的下游分类
lossA.backward()
print("【路A：分类 loss 经 prompt 反传】")
print(f"  q_x(细特征).grad 是否非零 = {qx7.grad is not None and qx7.grad.abs().sum().item()>0}"
      f"   例 q_x[head-L,ch0].grad = {qx7.grad[0,0,0].item():.4f}   → 特征在学")
print(f"  q_maps(路由).grad = {qm.grad}   → 路由【没有】梯度（被 detach 切断）")

# --- 路B 空间图路：不 detach(qm)，形成 f_maps，代理"consistency loss" ---
qx7.grad=None; qm.grad=None
fmc = (f_fine.unsqueeze(2)*qm.unsqueeze(-1)).sum(1)               # [6,3,36] 粗软图
lossB = fmc.sum()                                                 # 代理：作用在 f_maps 上的 consistency
lossB.backward()
print("\n【路B：consistency loss 经 f_maps 反传】")
print(f"  q_maps(路由).grad 是否非零 = {qm.grad is not None and qm.grad.abs().sum().item()>0}")
print("  q_maps.grad (b=0, 7×3) —— 每个细原型的路由都收到了梯度：")
for k,name in enumerate(fine_names):
    print(f"    {name:7s}: " + " ".join(f"{qm.grad[0,k,j].item():7.3f}" for j in range(3)))
print("  （每行 3 列相等 = 该细原型软图的总质量 Σ_pos f_fine；不同损失给的方向不同，但都能流回 q_maps）")
