# 探针：打印 p_linear 这个 ModuleList(里面是 nn.Linear) 的真实形状 + 初始化
# 例：n_pro=17,14,11,8,5 + DINOv2 ViT-B/14 → 每个 p_linear[i] = Linear(768,768)，看 weight/bias 形状
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
model = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5")

print("== 形状参数 ==")
print("D (in=out)   :", base.cls_token.shape[-1])      # 768
print("layer_n      :", model.layer_n)                  # 4
print("len(p_linear):", len(model.p_linear), "(= layer_n，注意不是 layer_n+1)")
print("num_landmarks:", model.num_landmarks, "(= 进 p_linear 的 q_x 的 token 数)")

print("\n== p_linear 逐个真实形状（每个是一整个 Linear 模块，含 weight+bias 两组参数）==")
for i, lin in enumerate(model.p_linear):
    w, b = lin.weight, lin.bias
    print(f"p_linear[{i}]: {lin}")
    print(f"            weight.shape={str(tuple(w.shape)):<12}(out,in)  bias.shape={str(tuple(b.shape)):<8}  "
          f"params={w.numel()+b.numel()}")
per = model.p_linear[0].weight.numel() + model.p_linear[0].bias.numel()
print(f"\n单个 p_linear 参数: 768*768 + 768 = {per}")
print(f"p_linear 总参数量 : {per} * {len(model.p_linear)} = {per*len(model.p_linear)}")

print("\n== 初始化（_init_weights_linear: weight=trunc_normal(0.02), bias=0）==")
w0, b0 = model.p_linear[0].weight, model.p_linear[0].bias
print(f"weight[0]: mean={w0.mean().item():+.5f} std={w0.std().item():.5f}")
print(f"bias[0]  : all_zero={bool((b0==0).all())}")

print("\n== forward 用法：对 q_x 的最后一维 D 做 D->D 投影（前面的 batch/landmark 维不变）==")
B, N, D = 4, model.num_landmarks, base.cls_token.shape[-1]
q_x = torch.randn(B, N, D)                 # 例：聚合后的 prompt 特征 (B, num_landmarks, D)
out = model.p_linear[0](q_x)               # 真实跑一遍
print(f"输入 q_x: {tuple(q_x.shape)}  --p_linear[0]-->  输出: {tuple(out.shape)}")
print("（forward 实际是 p_linear[q_index](p_norm[q_index](q_x))，再 cat 进 transformer 序列当 prompt）")
