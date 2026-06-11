# 探针：打印 p_norm 这个 ModuleList(里面是 LayerNorm) 的真实形状 + 初始化 + "联合归一化"行为
# 例：LayerNorm([num_landmarks, feature_dim]) = LayerNorm([4, 768]) → weight/bias 都是 (4,768)
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
model = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5")

print("== 形状参数 ==")
print("num_landmarks :", model.num_landmarks)            # 4
print("feature_dim D :", model.feature_dim)              # 768
print("layer_n       :", model.layer_n)                  # 4
print("len(p_norm)   :", len(model.p_norm), "(= layer_n，和 p_linear 一样，不是 layer_n+1)")

print("\n== p_norm 逐个真实形状（每个是一整个 LayerNorm，含 weight(γ)+bias(β)）==")
for i, ln in enumerate(model.p_norm):
    w, b = ln.weight, ln.bias
    print(f"p_norm[{i}]: {ln}")
    print(f"          normalized_shape={tuple(ln.normalized_shape)}  weight.shape={str(tuple(w.shape)):<10} "
          f"bias.shape={str(tuple(b.shape)):<10} params={w.numel()+b.numel()}")
per = model.p_norm[0].weight.numel() + model.p_norm[0].bias.numel()
print(f"\n单个 p_norm 参数: 4*768(γ) + 4*768(β) = {per}")
print(f"p_norm 总参数量 : {per} * {len(model.p_norm)} = {per*len(model.p_norm)}")

print("\n== 初始化（LayerNorm 默认：weight=1, bias=0；本模型未额外改）==")
w0, b0 = model.p_norm[0].weight, model.p_norm[0].bias
print(f"weight: all_one={bool((w0==1).all())}  bias: all_zero={bool((b0==0).all())}")

print("\n== 关键：normalized_shape=[4,768] → 沿'最后两维'联合归一化（不是逐 landmark）==")
B, N, D = 2, model.num_landmarks, model.feature_dim
x = torch.randn(B, N, D) * 5 + 3                  # 例：故意造非零均值/非单位方差的输入 (B,4,768)
out = model.p_norm[0](x)                          # 真实跑一遍
print(f"输入 {tuple(x.shape)}  --p_norm[0]-->  输出 {tuple(out.shape)}")
for b in range(B):
    print(f"  样本{b}: 整块[4x768=3072] mean={out[b].mean().item():+.4f} std={out[b].std().item():.4f}  "
          f"(→ 0/1，说明 4 个 landmark 被一起归一化)")
    print(f"         仅 landmark0 的 768 维 mean={out[b,0].mean().item():+.4f} std={out[b,0].std().item():.4f}  "
          f"(→ 不保证 0/1，证明不是逐 landmark 各归各的)")
