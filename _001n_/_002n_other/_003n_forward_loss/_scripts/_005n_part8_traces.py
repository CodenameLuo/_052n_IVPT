#!/usr/bin/env python
# Part 8（收尾读出 + 分类）：从 forward_dump.pt 取数，补算 modulation 前的 all_features
# 跑法：_004n_IVPT/bin/python _005n_part8_traces.py

import torch
import torch.nn.functional as F
torch.set_printoptions(sci_mode=False, precision=3, linewidth=200)

d = torch.load('forward_dump.pt')
maps_list = d['maps_list']          # 4 条：通道 8/6/5/4
all_features_mod = d['all_features_mod']   # [6,16,4]
scores = d['scores']                # [6,10,3]
outputs = d['outputs']              # [6,10]
x_level3 = d['x_level3']            # [6,16,6,6]
B, D, H, W = x_level3.shape

def compute_feat(maps, x):
    N = maps.shape[1]
    one_hot_map = F.one_hot(torch.argmax(maps, 1), N).permute(0,3,1,2) * maps
    af = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
    sum_pool = af.sum(dim=(3,4)).permute(0,2,1)
    count_map = one_hot_map.sum(dim=(2,3), keepdim=True)
    count_map_ = count_map + (count_map==0).float()
    return sum_pool / count_map_.squeeze(-1), count_map

all_features = compute_feat(maps_list[-1], x_level3)[0].permute(0,2,1)   # [6,16,4] 调制前

print("############### Part 8 ###############")
print("maps_list 4 条（干净重算的软分割图 = maps_loss / map_feat），通道随层：")
for i, m in enumerate(maps_list):
    print(f"  maps_list[{i}] = {list(m.shape)}")

print("\n最终最粗图 maps_list[-1] [6,4,6,6] (b=0)：4 通道 = 头/翅/身/bg")
part_nm = ['头','翅','身','bg']
for k in range(4):
    print(f"\n[maps_final ch{k}={part_nm[k]}]")
    for r in range(H):
        print("   " + " ".join(f"{maps_list[-1][0,k,r,c].item():5.2f}" for c in range(W)))

print("\nall_features（调制前）[6,16,4] (b=0)：4 个部件各一个 16 维特征（行=部件）")
print("        " + " ".join(f"c{j:02d}" for j in range(D)))
for k in range(4):
    print(f"  {part_nm[k]:3s}: " + " ".join(f"{all_features[0,j,k].item():5.2f}" for j in range(D)))

print("\nall_features_mod（调制后 LayerNorm[16,4]）[6,16,4] (b=0)：")
print("        " + " ".join(f"c{j:02d}" for j in range(D)))
for k in range(4):
    print(f"  {part_nm[k]:3s}: " + " ".join(f"{all_features_mod[0,j,k].item():5.2f}" for j in range(D)))

print("\nscores [6,10,3] (b=0)：去 bg、3 个部件各自给 10 类打分（行=部件，列=类别）")
print("        " + " ".join(f"cls{j}" for j in range(10)))
for k in range(3):
    print(f"  {part_nm[k]:3s}: " + " ".join(f"{scores[0,j,k].item():5.2f}" for j in range(10)))

print("\noutputs = scores.mean(dim=-1) [6,10]：3 部件投票平均 = 最终类别分")
print("        " + " ".join(f"cls{j}" for j in range(10)))
for b in range(B):
    pred = outputs[b].argmax().item()
    print(f"  图{b}: " + " ".join(f"{outputs[b,j].item():5.2f}" for j in range(10)) + f"   → 预测 cls{pred}")
