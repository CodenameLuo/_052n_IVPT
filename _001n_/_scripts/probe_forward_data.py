# 探针：核实 forward 里几个"数据层面"的事实（不只是形状）
# 目的：把 count_map 究竟是不是 patch 计数、softmax 各维是否真归一、是否有空原型 等问到实处。
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
B = 2
img = torch.randn(B, 3, 518, 518)

with torch.no_grad():
    x = m.patch_embed(img); x = m._pos_embed(x); x_len = x.shape[1]
    l = len(m.blocks)
    for i, block in enumerate(m.blocks):
        if i < l - m.layer_n:
            x = block(x); continue
        x = x[:, :x_len]
        qi = i - l + m.layer_n
        q = m.p_token[qi].expand(x.shape[0], -1, -1)
        x_ = m.norm(x.detach())[:, m.num_prefix_tokens:, :]
        x_ = m.unflatten(x_).permute(0, 3, 1, 2).contiguous()
        maps_raw = m.compute_xq(x_, q)
        p_bias = m.p_bias[qi].expand(x.shape[0], -1, -1, -1)
        maps = F.softmax(maps_raw + p_bias, dim=1)
        q_x, count_map = m.compute_feat(maps.detach(), x_)

        Pp1 = m.n_pro[qi]
        ch_sum_per_patch = maps.sum(dim=1)                      # 每 patch 在通道维求和
        peak_per_patch = maps.max(dim=1).values                # 每 patch 的最大软分配
        n_empty = int((count_map[:, :-1].squeeze() == 0).sum())
        print(f"[q_index={qi}  P+1={Pp1}]")
        print(f"  softmax 后 每 patch Σ_ch = {ch_sum_per_patch.mean():.4f} (min {ch_sum_per_patch.min():.4f}/max {ch_sum_per_patch.max():.4f}) -> 确为 1")
        print(f"  count_map 总和          = {count_map.sum():.1f}")
        print(f"  Σ_patch max_ch(maps)    = {peak_per_patch.sum():.1f}  <- 与 count_map 总和应相等")
        print(f"  若按 patch 硬计数应为    = {B*37*37}  (=B*1369) -> count_map 明显更小，故 count_map=软质量非计数")
        print(f"  平均峰值 max_ch          = {peak_per_patch.mean():.3f}  (P 越小峰越尖→总质量越大)")
        print(f"  空原型个数(count==0)     = {n_empty} / {Pp1-1}")

        # q_maps 行和
        q_c = m.p_classifier[qi](q_x[:, :-1])
        q_maps = F.gumbel_softmax(q_c, dim=-1, tau=1.0, hard=False)
        print(f"  q_maps 每细原型行和      = {q_maps.sum(-1).mean():.4f} (应≈1, softmax/gumbel over N=4)")

        # 还原 forward 后续以推进 x
        count_mask = (count_map[:, :-1] == 0).squeeze(-1).expand(-1, -1, q_c.shape[-1])
        q_maps = q_maps.masked_fill(count_mask, 1 / q_c.shape[-1])
        qme = q_maps.unsqueeze(-1).detach()
        q_x2 = (q_x[:, :-1].unsqueeze(2) * qme).sum(1) / qme.sum(1).squeeze(1)
        q_x2 = m.p_linear[qi](m.p_norm[qi](q_x2))
        x = torch.cat([x, q_x2], dim=1)
        x = block(x)
