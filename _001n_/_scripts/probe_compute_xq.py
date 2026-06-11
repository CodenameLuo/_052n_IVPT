# 探针：compute_xq —— 原型↔patch 的"负欧氏距离图"到底怎么算的
# 核实：① 每个中间量形状；② 用 ‖a-b‖²=‖a‖²-2a·b+‖b‖² 展开是否真等于暴力 -‖q-x‖²；
#       ③ 真实数值量级；④ 关键洞见：b_sq(patch范数)在下游 softmax(dim=1) 里会抵消；⑤ 省了多少显存。
import sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)
import torch
import torch.nn.functional as F
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

torch.manual_seed(0)
base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
m = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5", classifier_type="linear",
                          gumbel_softmax=True, part_dropout=0.3, modulation_type="layer_norm")
m.eval()
def sh(t): return tuple(t.shape)

# === 取第 1 个注入层(blk8)真实的 x_(特征图) 和 q(原型) ===
B = 2
img = torch.randn(B, 3, 518, 518)
with torch.no_grad():
    x = m.patch_embed(img); x = m._pos_embed(x); x_len = x.shape[1]
    for i in range(8):                       # 过前 8 个普通块
        x = m.blocks[i](x)
    x = x[:, :x_len]
    q = m.p_token[0].expand(x.shape[0], -1, -1)          # [2,17,768]
    x_ = m.norm(x.detach())[:, m.num_prefix_tokens:, :]
    x_ = m.unflatten(x_).permute(0, 3, 1, 2).contiguous() # [2,768,37,37]

print("输入 compute_xq 的两个张量：")
print("  x_(特征图 [B,C,H,W]) :", sh(x_))
print("  q (原型  [B,L,C])    :", sh(q), "  L=17 个原型, C=768")

# === 逐行复刻 compute_xq 并打印形状 ===
with torch.no_grad():
    ab = torch.einsum('bchw,blc->blhw', x_, q)            # a·b = Σ_c q[l,c]*x[c,h,w]
    print("\nab = einsum('bchw,blc->blhw') :", sh(ab), "  每个(原型l,位置hw)的点积; c 被求和掉")

    b_sq = x_.pow(2).sum(1, keepdim=True)                 # ‖patch‖²
    print("b_sq = Σ_c x² (keepdim)        :", sh(b_sq), "  [B,1,H,W] 只随 patch 位置")
    b_sq = b_sq.expand(-1, q.shape[1], -1, -1).contiguous()
    print("b_sq.expand->contiguous        :", sh(b_sq), "  广播到每个原型(同位置同值)")

    a_sq = q.pow(2).sum(-1, keepdim=True)                 # ‖原型‖²
    print("a_sq = Σ_c q² (keepdim)        :", sh(a_sq), "  [B,L,1] 只随原型")
    a_sq = a_sq.expand(-1, -1, x_.shape[-2] * x_.shape[-1])
    a_sq = a_sq.view(x_.shape[0], q.shape[1], x_.shape[-2], x_.shape[-1])
    print("a_sq.expand->view              :", sh(a_sq), "  铺到所有 37×37 位置(同原型同值)")

    dist = b_sq - 2 * ab + a_sq                           # ‖q-x‖²
    maps = -dist
    print("dist = b_sq - 2ab + a_sq       :", sh(dist))
    print("maps = -dist                   :", sh(maps), "  负距离=相似度(越大越像)")

# === ② 数值等价：展开式 vs 暴力 -‖q-x‖² ===
with torch.no_grad():
    # 暴力：把 q 摆成 [B,L,1,1,C], x 摆成 [B,1,C,H,W]->[B,1,H,W,C], 直接平方差求和
    x_hwC = x_.permute(0, 2, 3, 1)                        # [B,H,W,C]
    brute = -((q[:, :, None, None, :] - x_hwC[:, None]) ** 2).sum(-1)   # [B,L,H,W]
    diff = (maps - brute).abs().max().item()
    # 顺带验证 einsum 点积 == 手写 batched matmul
    ab_mm = torch.bmm(q, x_.flatten(2)).view_as(ab)       # q[B,L,C] @ x[B,C,HW]
    ab_diff = (ab - ab_mm).abs().max().item()
print(f"\n② 展开式 maps vs 暴力 -‖q-x‖²  最大绝对差 = {diff:.3e}  (≈ 浮点误差 → 完全等价)")
print(f"   einsum 点积 vs bmm 手写       最大绝对差 = {ab_diff:.3e}")

# === ③ 真实量级 ===
with torch.no_grad():
    print("\n③ 量级(真实随机权重):")
    print(f"   ‖patch‖² b_sq: 均值 {b_sq.mean():.1f}  (≈768, 因 x_ 过了 LayerNorm)")
    print(f"   ‖原型‖²  a_sq: 均值 {a_sq.mean():.3f}  (≈768*0.05²≈1.9, p_token 初始 std=0.05)")
    print(f"   maps        : min {maps.min():.1f}  max {maps.max():.1f}  (全负且量级大→softmax 看相对差)")

# === ④ 洞见：b_sq 在 softmax(dim=1) 里抵消 ===
with torch.no_grad():
    sm_full = F.softmax(maps, dim=1)                      # 真正用的
    sm_noB  = F.softmax(2 * ab - a_sq, dim=1)             # 故意丢掉 -b_sq 那一项
    d_soft = (sm_full - sm_noB).abs().max().item()
print(f"\n④ softmax(maps,1) vs softmax(2ab-a_sq,1) 最大绝对差 = {d_soft:.3e}")
print("   → b_sq 只随 patch、对每个 patch 是跨原型的常数, 在 softmax(over 原型) 里完全抵消")
print("   → 软分配实质由 2·q·x − ‖q‖² 决定(点积相似度 − 原型自范数偏置)")

# === ⑤ 显存：展开式省了多少 ===
Bn, L, C, H, W = B, q.shape[1], x_.shape[1], x_.shape[2], x_.shape[3]
naive = Bn * L * C * H * W
trick = Bn * L * H * W
print(f"\n⑤ 峰值元素数: 暴力中间量 [B,L,C,H,W]={naive:,}  vs  展开式 [B,L,H,W]={trick:,}")
print(f"   比值 = {naive//trick} = C(=768) → 展开式把距离计算的峰值显存压了 C 倍")

# === 小算例：手可验证 ===
print("\n=== 小算例(B=1,L=2,C=3,H=W=1, 数都设好可手算) ===")
xt = torch.tensor([[[[1.]], [[0.]], [[0.]]]])            # x_=[1,3,1,1], patch 特征=(1,0,0)
qt = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])        # q=[1,2,3], 原型0=(1,0,0) 原型1=(0,1,0)
with torch.no_grad():
    mt = m.compute_xq(xt, qt)
print("  patch=(1,0,0)  原型0=(1,0,0) 原型1=(0,1,0)")
print("  compute_xq ->", mt.flatten().tolist(), " 形状", sh(mt))
print("  手算: -‖(1,0,0)-(1,0,0)‖²=0(完全重合) ; -‖(1,0,0)-(0,1,0)‖²=-(1+1)= -2  ✓")
