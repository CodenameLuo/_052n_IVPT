# 探针：吃透 ab = torch.einsum('bchw,blc->blhw', x, q) 这一行
# 核实：① 标签→形状映射；② 单个元素=点积(手算)；③ 等于五重 for 循环；
#       ④ 等于 batched matmul(bmm)；⑤ 字母只是名字(换名同结果)；⑥ 输出字母顺序决定布局；⑦ 算量。
import sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)
import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

torch.manual_seed(0)
base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
m = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5", classifier_type="linear")
m.eval()
def sh(t): return tuple(t.shape)

# === 取真实的 x_(特征图) 和 q(原型)，同 compute_xq 入口 ===
B = 2
img = torch.randn(B, 3, 518, 518)
with torch.no_grad():
    x = m.patch_embed(img); x = m._pos_embed(x); x_len = x.shape[1]
    for i in range(8):
        x = m.blocks[i](x)
    x = x[:, :x_len]
    q = m.p_token[0].expand(x.shape[0], -1, -1)
    x_ = m.norm(x.detach())[:, m.num_prefix_tokens:, :]
    x_ = m.unflatten(x_).permute(0, 3, 1, 2).contiguous()

print("① 标签 → 形状映射")
print(f"   x ('bchw') = {sh(x_)}   → b={x_.shape[0]}, c={x_.shape[1]}, h={x_.shape[2]}, w={x_.shape[3]}")
print(f"   q ('blc')  = {sh(q)}     → b={q.shape[0]}, l={q.shape[1]}, c={q.shape[2]}")
print("   规则: 两边都出现但输出没有的字母 c → 求和(收缩掉); 输出里的 b,l,h,w → 保留为输出轴")

with torch.no_grad():
    ab = torch.einsum('bchw,blc->blhw', x_, q)
print(f"   ab ('blhw') = {sh(ab)}   (c=768 被求和没了; l 来自 q, h/w 来自 x, b 是公共 batch)")

print("\n② 单个元素就是一次点积 (手算核对)")
with torch.no_grad():
    one = ab[0, 0, 0, 0].item()
    byhand = (x_[0, :, 0, 0] * q[0, 0, :]).sum().item()   # Σ_c x[0,c,0,0]*q[0,0,c]
print(f"   ab[0,0,0,0] = {one:.4f}")
print(f"   Σ_c x[0,c,0,0]*q[0,0,c] = {byhand:.4f}   (原型0 与 patch(0,0) 的 768 维内积)  一致✓")

print("\n③ 等价于五重 for 循环 (小张量上逐元素核对)")
xb = torch.randn(2, 4, 3, 3)      # b,c,h,w
qb = torch.randn(2, 5, 4)         # b,l,c
ein = torch.einsum('bchw,blc->blhw', xb, qb)
loop = torch.zeros(2, 5, 3, 3)
for b in range(2):
    for l in range(5):
        for h in range(3):
            for w in range(3):
                s = 0.0
                for c in range(4):
                    s += xb[b, c, h, w] * qb[b, l, c]
                loop[b, l, h, w] = s
print(f"   einsum vs 五重循环  最大绝对差 = {(ein-loop).abs().max().item():.3e}  → 就是这个嵌套求和")

print("\n④ 等价于 batched 矩阵乘 bmm")
with torch.no_grad():
    bmm = torch.bmm(q, x_.flatten(2)).view_as(ab)   # q[B,L,C] @ x[B,C,HW] -> [B,L,HW] -> [B,L,H,W]
print(f"   bmm(q, x.flatten(2)).view -> {sh(bmm)}   与 einsum 差 = {(ab-bmm).abs().max().item():.3e}")

print("\n⑤ 字母只是名字, 换一套字母结果不变")
with torch.no_grad():
    ab2 = torch.einsum('ncij,nkc->nkij', x_, q)   # b→n, l→k, h→i, w→j, c 不变
print(f"   einsum('ncij,nkc->nkij') 与原写法 差 = {(ab-ab2).abs().max().item():.3e}  (关键是'哪些共享/哪个被丢',不是字母本身)")

print("\n⑥ 输出字母的顺序 = 输出轴的顺序 (布局由你定)")
with torch.no_grad():
    ab_hwl = torch.einsum('bchw,blc->bhwl', x_, q)   # 输出写成 bhwl
print(f"   '->bhwl' 形状 = {sh(ab_hwl)}   它 == ab.permute(0,2,3,1)? {torch.allclose(ab_hwl, ab.permute(0,2,3,1))}  (数一样,只是轴顺序换了)")

print("\n⑦ 算量")
Bn, L, C, H, W = q.shape[0], q.shape[1], x_.shape[1], x_.shape[2], x_.shape[3]
print(f"   输出元素数 B*L*H*W = {Bn*L*H*W:,}; 每个是 C={C} 维点积 → 乘加 ≈ {Bn*L*H*W*C:,}")

print("\n=== 小算例 (B=1,C=2,L=3,H=W=1, 全手算) ===")
xt = torch.tensor([[[[2.]], [[3.]]]])               # x=[1,2,1,1], patch 特征=(2,3)
qt = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]]) # q=[1,3,2], 3 个原型=(1,0)(0,1)(1,1)
with torch.no_grad():
    abt = torch.einsum('bchw,blc->blhw', xt, qt)
print("   patch=(2,3)  原型=(1,0),(0,1),(1,1)")
print(f"   einsum -> {abt.flatten().tolist()}   形状 {sh(abt)}")
print("   手算: (2,3)·(1,0)=2 ; (2,3)·(0,1)=3 ; (2,3)·(1,1)=5  → [2,3,5] ✓")
