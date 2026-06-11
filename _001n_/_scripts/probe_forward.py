# 探针：forward 前向推理全过程的真实 tensor 形状 / 数据流
# 做法：用模型自己的子模块(patch_embed/_pos_embed/blocks/compute_xq/compute_feat/p_*)
#       原样复刻 forward 的每一步并打印形状；最后再跑一次真实 m(img) 交叉验证输出形状一致。
import os, sys
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

B = 2
img = torch.randn(B, 3, 518, 518)
print("n_pro =", m.n_pro, " layer_n =", m.layer_n, " num_landmarks(N) =", m.num_landmarks,
      " num_prefix_tokens =", m.num_prefix_tokens)
print("输入图像 img :", sh(img))

with torch.no_grad():
    # === 0. patch + pos ===
    x = m.patch_embed(img)
    print("patch_embed  :", sh(x), " (37*37=1369 个 patch token)")
    x = m._pos_embed(x)
    print("_pos_embed   :", sh(x), " (cls1 + reg4 + patch1369 = 1374)")
    x_len = x.shape[1]
    l = len(m.blocks)
    print(f"x_len = {x_len}   总块数 l = {l}   普通块 i<{l-m.layer_n}   注入块 i>={l-m.layer_n}")

    x_buffer, q_buffer, m_buffer, qm_buffer = [], [], [], []
    for i, block in enumerate(m.blocks):
        if i < l - m.layer_n:
            x = block(x)
            print(f"[blk {i:2d} 普通] -> {sh(x)}")
        else:
            x = x[:, :x_len]
            q_index = i - l + m.layer_n
            q = m.p_token[q_index].expand(x.shape[0], -1, -1)
            print(f"\n[blk {i:2d} 注入 q_index={q_index}] 该层原型数 P+1 = n_pro[{q_index}] = {m.n_pro[q_index]}  (P={m.n_pro[q_index]-1} 前景+1 背景)")
            print(f"  strip prompt 后 x={sh(x)}   q(该层原型)={sh(q)}")
            x_buffer.append(x); q_buffer.append(q)
            x_ = m.norm(x.detach())
            x_ = x_[:, m.num_prefix_tokens:, :]
            x_ = m.unflatten(x_)
            x_ = x_.permute(0, 3, 1, 2).contiguous()
            print(f"  norm+去前缀+unflatten+permute -> x_ = {sh(x_)}  [B,D,H,W] 特征图")
            maps = m.compute_xq(x_, q)
            print(f"  compute_xq            -> maps = {sh(maps)}  [B,P+1,H,W] (原型↔patch 负欧氏距离)")
            p_bias = m.p_bias[q_index].expand(x.shape[0], -1, -1, -1)
            maps = F.softmax(maps + p_bias, dim=1)
            print(f"  +p_bias, softmax(dim=1) -> maps = {sh(maps)}  每 patch 在 P+1 原型上软分配  Σ_ch≈{maps.sum(1).mean().item():.3f}")
            q_x, count_map = m.compute_feat(maps.detach(), x_)
            print(f"  compute_feat          -> q_x={sh(q_x)} [B,P+1,D] 各原型区域均值特征   count_map={sh(count_map)}  Σ={count_map.sum().item():.0f} (=B*1369)")
            q_c = m.p_classifier[q_index](q_x[:, :-1])
            print(f"  p_classifier(去背景)   -> q_c={sh(q_c)} [B,P,N] 细原型→{m.num_landmarks} 粗部件打分")
            count_mask = (count_map[:, :-1] == 0).squeeze(-1).expand(-1, -1, q_c.shape[-1])
            if m.gumbel_softmax:
                q_maps = F.gumbel_softmax(q_c, dim=-1, tau=m.gumbel_softmax_temperature, hard=m.gumbel_softmax_hard)
            else:
                q_maps = F.softmax(q_c, dim=-1)
            q_maps = q_maps.masked_fill(count_mask, 1 / q_c.shape[-1])
            print(f"  gumbel/softmax(dim=-1) -> q_maps={sh(q_maps)} [B,P,N] 细→粗 软对齐(每细原型一行和≈1)")
            qm_buffer.append(q_maps)
            q_maps_expanded = q_maps.unsqueeze(-1)
            f_maps = maps.flatten(2)
            print(f"  maps.flatten(2)        -> f_maps={sh(f_maps)} [B,P+1,L]  L=H*W=1369")
            q_maps_detach = q_maps_expanded.detach()
            q_x_weighted = q_x[:, :-1].unsqueeze(2) * q_maps_detach
            q_x = q_x_weighted.sum(dim=1)
            q_maps_sum = q_maps_detach.sum(1).squeeze(1)
            q_x = q_x / q_maps_sum
            print(f"  按 q_maps 聚 P 细原型→N 粗部件 -> q_x={sh(q_x)} [B,N,D]")
            f_bg = f_maps[:, -1:]
            f_maps_weighted = f_maps[:, :-1].unsqueeze(2) * q_maps_expanded
            f_maps = f_maps_weighted.sum(dim=1)
            f_maps = torch.cat([f_maps, f_bg], dim=1)
            print(f"  分配图同样聚成粗部件+接回背景 -> f_maps={sh(f_maps)} [B,N+1,L]")
            m_buffer.append(f_maps)
            q_x = m.p_linear[q_index](m.p_norm[q_index](q_x))
            print(f"  p_norm+p_linear        -> prompt q_x={sh(q_x)} [B,N,D]")
            x = torch.cat([x, q_x], dim=1)
            print(f"  cat 进序列             -> x={sh(x)} [B, x_len+N, D]")
            x = block(x)
            print(f"  block(x)               -> {sh(x)}")

    # === 末读出层 ===
    q = m.p_token[-1].expand(x.shape[0], -1, -1)
    x = x[:, :x_len]
    x_buffer.append(x); q_buffer.append(q)
    print(f"\n[末读出层] q=p_token[-1]={sh(q)} (n_pro[-1]=5)   x(strip)={sh(x)}")
    print(f"x_buffer len={len(x_buffer)}   q_buffer len={len(q_buffer)}   (=layer_n+1=5)")

    print("\n--- 读出循环：每个缓存层重算一遍 part-assignment map ---")
    maps_list = []
    for i, (xb, qb) in enumerate(zip(x_buffer, q_buffer)):
        x_ = m.norm(xb.detach())
        x_ = x_[:, m.num_prefix_tokens:, :]
        x_ = m.unflatten(x_).permute(0, 3, 1, 2).contiguous()
        maps = m.compute_xq(x_, qb)
        p_bias = m.p_bias[i].expand(xb.shape[0], -1, -1, -1)
        maps = F.softmax(maps + p_bias, dim=1)
        maps = maps + 1e-6
        maps = maps / maps.sum(dim=1, keepdim=True)
        maps_list.append(maps)
        print(f"  maps_list[{i}] = {sh(maps)}  (n_pro[{i}]={m.n_pro[i]} 通道)")

    maps = maps_list[-1]
    f_maps = maps.flatten(2)
    m_buffer.append(f_maps)
    print(f"m_buffer len={len(m_buffer)} (4 注入 + 1 末)   qm_buffer len={len(qm_buffer)} (4 注入)")

    x = m.unflatten(m.norm(x_buffer[-1])[:, m.num_prefix_tokens:, :]).permute(0, 3, 1, 2).contiguous()
    all_features = m.compute_feat(maps, x)[0]
    print(f"\n末层 compute_feat -> all_features={sh(all_features)} [B,N+1,D]")
    all_features = all_features.permute(0, 2, 1)
    all_features_mod = m.modulation(all_features)
    print(f"permute+modulation -> all_features_mod={sh(all_features_mod)} [B,D,N+1]")
    scores = m.fc_class_landmarks(all_features_mod[..., :-1].permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
    print(f"fc_class_landmarks(去背景) -> scores={sh(scores)} [B,num_classes,N]")
    print(f"返回的 maps_list[-3] = {sh(maps_list[-3])}  (索引2, n_pro[2]=11 通道)")

# === 交叉验证：真实 forward 的 5 个返回 ===
print("\n=== 交叉验证：真实 m(img) 的 5 个返回 ===")
with torch.no_grad():
    out = m(img)
print("ret[0] all_features_mod :", sh(out[0]))
print("ret[1] maps_list[-3]    :", sh(out[1]))
print("ret[2] scores           :", sh(out[2]))
print("ret[3] map_feat (list)  : len", len(out[3]), " 各:", [sh(t) for t in out[3]])
print("ret[4][0] m_buffer      : len", len(out[4][0]), " 各:", [sh(t) for t in out[4][0]])
print("ret[4][1] qm_buffer     : len", len(out[4][1]), " 各:", [sh(t) for t in out[4][1]])
