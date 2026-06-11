# 探针：打印 IVPT 从 timm 骨架"偷"过来的那批属性 + 几个配置标志的真实值/形状/类型
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
m = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5",
                          gumbel_softmax=True, noise_variance=0.0)

def shp(x):  # Parameter/Tensor → 形状
    return tuple(x.shape) if hasattr(x, "shape") else x

print("== 标量 / 布尔配置 ==")
print("num_classes        :", m.num_classes)
print("noise_variance     :", m.noise_variance)
print("num_prefix_tokens  :", m.num_prefix_tokens)
print("num_reg_tokens     :", m.num_reg_tokens)
print("has_class_token    :", m.has_class_token)
print("no_embed_class     :", m.no_embed_class)
print("gumbel_softmax     :", m.gumbel_softmax)
print("feature_dim (D)    :", m.feature_dim)
print("return_transformer_qkv:", m.return_transformer_qkv)
print("h_fmap, w_fmap     :", m.h_fmap, m.w_fmap)

print("\n== 可学习参数(Parameter) → 形状 ==")
print("cls_token .shape   :", shp(m.cls_token))
print("reg_token .shape   :", shp(m.reg_token))
print("pos_embed .shape   :", shp(m.pos_embed))

print("\n== 子模块(Module) → 类型/关键信息 ==")
print("patch_embed        :", type(m.patch_embed).__name__,
      "| img_size", m.patch_embed.img_size, "patch", m.patch_embed.patch_size,
      "grid", m.patch_embed.grid_size, "num_patches", m.patch_embed.num_patches)
print("patch_embed.proj   :", m.patch_embed.proj)
print("pos_drop           :", m.pos_drop)
print("norm_pre           :", type(m.norm_pre).__name__, "(Identity 表示骨架没用 pre-norm)")
print("blocks             :", type(m.blocks).__name__, "len =", len(m.blocks),
      "| 单块类型 =", type(m.blocks[0]).__name__, "(已被 convert_blocks_and_attention 改类)")
print("norm (最终)        :", m.norm)

print("\n== pos_embed 是否覆盖 prefix token？==")
np_ = m.patch_embed.num_patches
print(f"num_patches={np_}, num_prefix_tokens={m.num_prefix_tokens}, pos_embed 第二维={m.pos_embed.shape[1]}")
print("→ pos_embed 第二维 == num_patches 说明位置编码只给 patch、不含 cls/reg（即 no_embed_class 路线）"
      if m.pos_embed.shape[1] == np_ else
      "→ pos_embed 第二维 == num_patches+prefix 说明含 cls/reg 位置")
