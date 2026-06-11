# 探针：讲清 compute_feat —— 把「per-patch 特征图 x + 软分配图 maps」压成「每个部件一个特征向量」
# 三步骤：argmax 硬分区 → 软置信加权求和 → 除以质量(=加权平均)；外加 空部件防 NaN、与"纯软池化"的对比
import torch
import torch.nn.functional as F
torch.manual_seed(0)
def sh(t): return tuple(t.shape)

def compute_feat(maps, x):
    # 与源码逐行一致，只多吐一个 one_hot_map 便于观察
    N = maps.shape[1]
    one_hot_map = F.one_hot(torch.argmax(maps, dim=1), num_classes=N).permute(0, 3, 1, 2) * maps
    all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
    sum_pool = all_features.sum(dim=(3, 4)).permute(0, 2, 1)
    count_map = one_hot_map.sum(dim=(2, 3), keepdim=True)
    count_map_ = count_map + (count_map == 0).float()
    feat = sum_pool / count_map_.squeeze(-1)
    return feat, count_map, one_hot_map

print("="*74); print("1) 小算例：B=1, N=2 部件, C=2 通道, 2 个 patch（每步可手算）"); print("="*74)
# maps[b,p,h,w]：每个 patch 在 2 个部件上的软分数
maps = torch.tensor([[[[0.8, 0.3]], [[0.2, 0.7]]]])   # [1,2,1,2]
# x[b,c,h,w]：每个 patch 的 2 维特征   patch0=[10,20]  patch1=[30,40]
x = torch.tensor([[[[10., 30.]], [[20., 40.]]]])      # [1,2,1,2]
print("maps:", sh(maps), " x:", sh(x))
print("argmax(maps,dim=1):", torch.argmax(maps, dim=1).flatten().tolist(), " => patch0→部件0, patch1→部件1")
feat, count, ohm = compute_feat(maps, x)
print("one_hot_map (硬分区 × 软值)：")
print("   部件0:", ohm[0,0].flatten().tolist(), "  部件1:", ohm[0,1].flatten().tolist(), " (各 patch 只在它 argmax 的部件留软值)")
print("count_map (每部件软质量):", count.flatten().tolist(), "  手算 部件0=0.8, 部件1=0.7")
print("feat (每部件一个特征向量)：")
print("   部件0:", feat[0,0].tolist(), "  手算 (0.8*[10,20])/0.8 = [10,20] = patch0 特征")
print("   部件1:", feat[0,1].tolist(), "  手算 (0.7*[30,40])/0.7 = [30,40] = patch1 特征")
print("   注：部件只占 1 个 patch 时，软置信在 分子/分母 抵消 → 特征=该 patch 原特征，不被置信缩放")

print(); print("="*74); print("2) 一个部件占 2 个 patch：得到置信加权平均（高置信 patch 拉得更多）"); print("="*74)
maps2 = torch.tensor([[[[0.8, 0.6, 0.1]], [[0.2, 0.4, 0.9]]]])   # [1,2,1,3] patch0,1→部件0; patch2→部件1
x2 = torch.tensor([[[[10., 100., 0.]], [[20., 100., 0.]]]])      # patch0=[10,20] patch1=[100,100] patch2=[0,0]
feat2, _, _ = compute_feat(maps2, x2)
num = 0.8*torch.tensor([10.,20.]) + 0.6*torch.tensor([100.,100.])
print("部件0 占 patch0(conf0.8,[10,20]) + patch1(conf0.6,[100,100])")
print("   手算 (0.8*[10,20]+0.6*[100,100])/(0.8+0.6) =", [round(v,4) for v in (num/1.4).tolist()])
print("   实测 部件0 =", [round(v,4) for v in feat2[0,0].tolist()])

print(); print("="*74); print("3) 空部件（无 patch 选它）：count=0 → 特征=0 向量，不 NaN"); print("="*74)
maps3 = torch.tensor([[[[0.8, 0.7]], [[0.2, 0.3]], [[0.0, 0.0]]]])  # [1,3,1,2] 部件2 永不胜出
x3 = torch.tensor([[[[10., 30.]], [[20., 40.]]]])
feat3, count3, _ = compute_feat(maps3, x3)
print("count_map:", count3.flatten().tolist(), " (部件2=0)")
print("feat 部件2:", feat3[0,2].tolist(), "  整体有 NaN 吗:", bool(torch.isnan(feat3).any()), " (count_map_ 的 +1 兜底)")

print(); print("="*74); print("4) 真实形状 maps[2,17,37,37] + x[2,768,37,37] 的结构性质"); print("="*74)
B,N,H,W,C = 2,17,37,37,768
maps_r = F.softmax(torch.randn(B,N,H,W), dim=1)   # 模拟 softmax(maps+p_bias) 后的软分配图
x_r = torch.randn(B,C,H,W)
feat_r, count_r, ohm_r = compute_feat(maps_r, x_r)
print("输入 maps:", sh(maps_r), " x:", sh(x_r), " => 输出 feat:", sh(feat_r), " count_map:", sh(count_r))
big = ohm_r.unsqueeze(1) * x_r.unsqueeze(2)
print("中间 5D all_features:", sh(big), " 元素数=", big.numel(), "(≈%.0f MB f32；compute_feat 是物化这张大张量的，和 compute_xq 省显存相反)"%(big.numel()*4/1e6))
nz = (ohm_r != 0).sum(dim=1)                       # 每个 (b,h,w) 跨 17 部件的非零个数
print("硬分区检验: 每 patch 跨 17 部件非零数  min=%d max=%d (恒=1 → patch 被硬划给唯一部件)"%(nz.min().item(), nz.max().item()))
lhs = count_r.sum(dim=1).flatten()                 # 每图 Σ_部件 count
rhs = maps_r.max(dim=1).values.sum(dim=(1,2))      # 每图 Σ_patch (该 patch argmax 处的软值)
print("质量守恒: Σ_部件 count =", [round(v,3) for v in lhs.tolist()], " == Σ_patch max_p maps =", [round(v,3) for v in rhs.tolist()])
print("总 patch 数 =", H*W, " => 平均每 patch 的 argmax 软值 ≈ %.3f (<1，所以 count 不是整数计数而是软质量)"%(lhs[0].item()/(H*W)))

print(); print("="*74); print("5) 关键设计：argmax 硬分区  vs  纯软池化（每 patch 摊给所有部件）"); print("="*74)
def pure_soft(maps, x):
    num = (maps.unsqueeze(1)*x.unsqueeze(2)).sum(dim=(3,4)).permute(0,2,1)   # [B,N,C]
    den = maps.sum(dim=(2,3)).unsqueeze(-1)                                  # [B,N,1]
    return num/den
hard,_,_ = compute_feat(maps, x)
soft = pure_soft(maps, x)
print("部件0  硬分区:", [round(v,4) for v in hard[0,0].tolist()], "   纯软池化:", [round(v,4) for v in soft[0,0].tolist()])
print(" => 硬分区只用 patch0；纯软还混进 patch1(它其实更属于部件1) => argmax 硬分区让部件之间更不串味")
