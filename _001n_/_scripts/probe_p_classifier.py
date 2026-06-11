# 探针：打印 p_classifier 这个 ModuleList(里面是 nn.Linear) 的真实形状 + 初始化 + 每层前向
# 例：Linear(768, num_landmarks=4) → 把"细原型特征"映射到"4 个粗部件"的打分（fine→coarse 对齐头）
import os, sys
REPO = "/root/autodl-tmp/_000n_00000000/_000n_uni/_001n_my_codes/_001n_projects/_005n_claude_code_projs/_001n_cc_translation/_010n_codes/_019n_IVPT/IVPT"
sys.path.insert(0, REPO)

import torch
from timm.models import create_model
from models.individual_landmark_vit import IndividualLandmarkViT

base = create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False, img_size=518)
model = IndividualLandmarkViT(base, num_classes=200, n_pro="17,14,11,8,5")

print("== 形状参数 ==")
print("D (in)        :", base.cls_token.shape[-1])       # 768
print("num_landmarks :", model.num_landmarks, "(= out_features，固定 4)")
print("n_pro         :", model.n_pro)
print("layer_n       :", model.layer_n)
print("len(p_classifier):", len(model.p_classifier), "(= layer_n)")

print("\n== p_classifier 逐个真实形状（Linear 768 -> 4，含 weight+bias）==")
for i, lin in enumerate(model.p_classifier):
    w, b = lin.weight, lin.bias
    print(f"p_classifier[{i}]: {lin}")
    print(f"               weight.shape={str(tuple(w.shape)):<10}(out,in)  bias.shape={str(tuple(b.shape)):<6}  "
          f"params={w.numel()+b.numel()}")
per = model.p_classifier[0].weight.numel() + model.p_classifier[0].bias.numel()
print(f"\n单个 p_classifier 参数: 4*768 + 4 = {per}")
print(f"p_classifier 总参数量 : {per} * {len(model.p_classifier)} = {per*len(model.p_classifier)}")

print("\n== 初始化（★没被 _init_weights 重置 → 保持 nn.Linear 默认 kaiming_uniform）==")
w0, b0 = model.p_classifier[0].weight, model.p_classifier[0].bias
print(f"weight[0]: mean={w0.mean().item():+.5f} std={w0.std().item():.5f}  (默认 bound=1/sqrt(768)≈0.0361)")
print(f"bias[0]  : all_zero={bool((b0==0).all())}  (对比 p_linear 的 bias=0：这里默认初始化 → 非0)")

print("\n== forward 用法：输入'细原型特征' [B,P,768] -> 输出 [B,P,4]（P 每层不同，768->4 不变）==")
B = 4
for qi in range(model.layer_n):
    P = model.n_pro[qi] - 1                     # 去掉背景后的前景原型数：16/13/10/7
    q_x_fg = torch.randn(B, P, base.cls_token.shape[-1])
    q_c = model.p_classifier[qi](q_x_fg)        # 真实跑一遍
    print(f"p_classifier[{qi}]: 输入 {tuple(q_x_fg.shape)} (该层 {P} 个前景原型) -> 输出 {tuple(q_c.shape)}")
print("→ 中间那维 P(原型数) 每层不同(16/13/10/7)，但 Linear 只动末维 768->4；")
print("  输出 [B,P,4] 再 softmax/gumbel(dim=-1) 得'每个细原型该归到 4 个粗部件中谁'的分配权重")
