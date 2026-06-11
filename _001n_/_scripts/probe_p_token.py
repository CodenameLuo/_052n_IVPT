# 探针：实例化真实 IVPT 模型，打印 p_token 这个 ParameterList 的真实形状
# 例：n_pro=17,14,11,8,5 + DINOv2 ViT-B/14 骨架 → 看每个 p_token[i] 到底多大
import os, sys
REPO = os.path.join(os.path.dirname(__file__), "IVPT")
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

# 造骨架(纯 timm ViT，pretrained=False 不下权重)
base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
print("== 骨架信息 ==")
print("cls_token.shape      :", tuple(base.cls_token.shape))   # 取 [-1] 当 p_token 的最后一维
print("embed_dim (D)        :", base.embed_dim)
print("num_prefix_tokens    :", base.num_prefix_tokens)        # 1 cls + 4 reg
print("len(blocks)          :", len(base.blocks))

# 套上部件机制
model = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5")
print("\n== 推出来的形状参数 ==")
print("n_pro        :", model.n_pro)
print("layer_n      :", model.layer_n, "(= len(n_pro)-1, 注入层数)")
print("num_landmarks:", model.num_landmarks, "(= n_pro[-1]-1)")
print("len(p_token) :", len(model.p_token), "(= layer_n+1)")

print("\n== p_token 逐个真实形状 ==")
for i, p in enumerate(model.p_token):
    print(f"p_token[{i}]: shape={str(tuple(p.shape)):<14} numel={p.numel():>6}  requires_grad={p.requires_grad}")
total = sum(p.numel() for p in model.p_token)
print(f"\np_token 总参数量: {total} (= sum(n_pro)*D = {sum(model.n_pro)}*{base.embed_dim})")

# 例：取一个看初始化是不是"zeros"(其实被 normal_(std=0.05) 覆盖了)
p0 = model.p_token[0]
print(f"\np_token[0] 统计: mean={p0.mean().item():+.4f} std={p0.std().item():.4f} (zeros 只是分配、随即被 normal std=0.05 填)")

# 例：forward 里怎么用 —— expand 到 batch
B = 4
expanded = model.p_token[0].expand(B, -1, -1)
print(f"forward 中 p_token[0].expand(B={B},-1,-1) -> {tuple(expanded.shape)} (第0维 1->B 广播，参数不复制)")
