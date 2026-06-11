# 探针：__init__ 尾段（unflatten / gumbel / modulation / dropout / 分类头 / 末尾两调用）真实形状
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)
import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
m = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5", classifier_type="linear",
                          gumbel_softmax=True, part_dropout=0.3, modulation_type="layer_norm")

print("== unflatten：把扁平 patch 序列摊回 2D 网格 ==")
print("unflatten :", m.unflatten)
x = torch.randn(2, m.h_fmap*m.w_fmap, m.feature_dim)        # [2,1369,768]
print("  demo:", tuple(x.shape), "-> unflatten ->", tuple(m.unflatten(x).shape))

print("\n== gumbel / modulation 配置 ==")
print("gumbel_softmax_temperature:", m.gumbel_softmax_temperature)
print("gumbel_softmax_hard       :", m.gumbel_softmax_hard)
print("modulation_type           :", repr(m.modulation_type), "(★存了但 forward 不按它分支，见 Q64)")

print("\n== modulation：LayerNorm([D, num_landmarks+1]) ==")
mod = m.modulation
print("modulation :", mod)
print("  normalized_shape:", tuple(mod.normalized_shape),
      "weight", tuple(mod.weight.shape), "bias", tuple(mod.bias.shape),
      "params", mod.weight.numel()+mod.bias.numel())
af = torch.randn(2, m.feature_dim, m.num_landmarks+1)       # [2,768,5]
print("  demo:", tuple(af.shape), "-> modulation ->", tuple(mod(af).shape))

print("\n== dropout_full_landmarks ==")
print("dropout_full_landmarks:", m.dropout_full_landmarks, "(★定义了但 forward 没用，见 Q64)")

print("\n== 分类头 classifier_type='linear' ==")
print("classifier_type   :", m.classifier_type)
print("fc_class_landmarks:", m.fc_class_landmarks)
print("  weight", tuple(m.fc_class_landmarks.weight.shape), " bias =", m.fc_class_landmarks.bias,
      " params", m.fc_class_landmarks.weight.numel())
feat = torch.randn(2, m.num_landmarks, m.feature_dim)       # [2,4,768] 去背景后的部件特征
print("  demo:", tuple(feat.shape), "-> fc ->", tuple(m.fc_class_landmarks(feat).shape))

print("\n== 分类头 classifier_type='independent_mlp'（每部件一条独立 Linear）==")
m2 = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5", classifier_type="independent_mlp")
print("fc_class_landmarks:", m2.fc_class_landmarks)
print("  子层数:", len(m2.fc_class_landmarks.feature_layers), "(= num_landmarks=4)")
print("  单条:", m2.fc_class_landmarks.feature_layers[0])
tot = sum(p.numel() for p in m2.fc_class_landmarks.parameters())
print("  总参数:", tot, "(= 4 * 768*200 ，无 bias)")
print("  demo:", (2,4,768), "-> fc ->", tuple(m2.fc_class_landmarks(torch.randn(2,4,768)).shape))

print("\n== 末尾两调用的效果 ==")
print("convert_blocks_and_attention -> blocks[0] 类型 =", type(m.blocks[0]).__name__)
print("_init_weights -> fc weight std =", round(m.fc_class_landmarks.weight.std().item(),5),
      "(trunc_normal 0.02)；p_linear[0].bias all_zero =", bool((m.p_linear[0].bias==0).all()))
