# 探针：实例化真实 IVPT 模型，打印 p_bias 这个 ParameterList 的真实形状 + 初始化情况
# 例：n_pro=17,14,11,8,5 + DINOv2 ViT-B/14 + img=518 → 看每个 p_bias[i] 的 4 维到底多大、初值是不是 0
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
print("== 骨架 / 网格信息（决定 p_bias 后两维）==")
print("img_size          :", base.patch_embed.img_size)
print("patch_size        :", base.patch_embed.patch_size)
print("grid_size (H,W)    :", base.patch_embed.grid_size)     # = (37,37)
print("num_patches       :", base.patch_embed.num_patches)    # = 37*37 = 1369
print("num_prefix_tokens :", base.num_prefix_tokens, "(1 cls + 4 reg，不进 p_bias 的空间网格)")

model = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5")
print("\n== 推出来的形状参数 ==")
print("n_pro          :", model.n_pro)
print("h_fmap, w_fmap :", model.h_fmap, model.w_fmap, "(= img//patch = 518//14 = 37)")
print("len(p_bias)    :", len(model.p_bias), "(= layer_n+1)")

print("\n== p_bias 逐个真实形状 ==")
for i, p in enumerate(model.p_bias):
    print(f"p_bias[{i}]: shape={str(tuple(p.shape)):<18} numel={p.numel():>6}  requires_grad={p.requires_grad}")
total = sum(p.numel() for p in model.p_bias)
print(f"\np_bias 总参数量: {total}  (= sum(n_pro)*h*w = {sum(model.n_pro)}*{model.h_fmap}*{model.w_fmap})")

# 关键：p_bias 有没有像 p_token 那样被 normal_ 重新初始化？
print("\n== 初始化对比（p_bias 全 0 vs p_token 被 normal 覆盖）==")
pb0 = model.p_bias[0]
print(f"p_bias[0] : mean={pb0.mean().item():+.5f} std={pb0.std().item():.5f} all_zero={bool((pb0==0).all())}")
pt0 = model.p_token[0]
print(f"p_token[0]: mean={pt0.mean().item():+.5f} std={pt0.std().item():.5f} all_zero={bool((pt0==0).all())}")

# 例：forward 里怎么用 —— expand 到 batch，再加到相似度图上(softmax 前)
B = 4
expanded = model.p_bias[0].expand(B, -1, -1, -1)
print(f"\nforward 中 p_bias[0].expand(B={B},-1,-1,-1) -> {tuple(expanded.shape)}")
print("用法: maps = softmax(compute_xq(x_,q) + p_bias, dim=1)  # 加到 [B, n_pro[i], 37, 37] 的相似度图上")
