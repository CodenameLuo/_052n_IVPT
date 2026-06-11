# 探针：讲清 forward 里  maps = softmax(maps + p_bias, dim=1)  里的 "maps + p_bias" 到底按维度怎么加
# 核心三点：广播(broadcast 从右往左对齐) + 逐元素加 + .expand 与裸广播等价
# 小例肉眼可手算；大例核真实形状 [2,17,37,37] + [1,17,37,37]
import torch
import torch.nn.functional as F

torch.manual_seed(0)
def sh(t): return tuple(t.shape)

print("="*72)
print("1) 小算例：[2,2,1,1] + [1,2,1,1]   (缩小版，结果能手算核对)")
print("="*72)
# maps: B=2 张图、P=2 个部件、H=W=1
maps_s = torch.tensor([
    [[[10.]], [[20.]]],   # 图0: 部件0=10, 部件1=20
    [[[30.]], [[40.]]],   # 图1: 部件0=30, 部件1=40
])
# p_bias: batch=1（只存一份）、P=2 个部件各一个先验标量
pbias_s = torch.tensor([
    [[[0.5]], [[-0.5]]],  # 部件0 先验 +0.5, 部件1 先验 -0.5
])
print("maps_s  :", sh(maps_s), "  p_bias_s:", sh(pbias_s))
out_s = maps_s + pbias_s
print("maps_s + p_bias_s ->", sh(out_s))
print("图0:", out_s[0].flatten().tolist(), " 手算 10+0.5=10.5, 20-0.5=19.5")
print("图1:", out_s[1].flatten().tolist(), " 手算 30+0.5=30.5, 40-0.5=39.5")
print("=> 关键：图0、图1 加的是【同一组】先验 [+0.5,-0.5]；batch 维的 1 被复用到 2")

print()
print("="*72)
print("2) 真实形状：maps[2,17,37,37] + p_bias[1,17,37,37]")
print("="*72)
B, P1, H, W = 2, 17, 37, 37
maps = torch.randn(B, P1, H, W)                       # 模拟 compute_xq 出来的相似度图
pbias_param = torch.randn(1, P1, H, W) * 0.1          # 复刻真实 p_bias[0] 的形状(真实初值是全0)
print("maps          :", sh(maps))
print("p_bias(stored):", sh(pbias_param), " <- batch 维只有 1")

# (a) 源码写法：先 .expand 到 batch，再加
pbias_exp = pbias_param.expand(B, -1, -1, -1)
print("p_bias.expand(B,-1,-1,-1):", sh(pbias_exp), " (视图，不复制底层数据)")
out_expand = maps + pbias_exp
# (b) 裸广播：直接把 (1,...) 加上去，torch 自动广播
out_bcast = maps + pbias_param

print("maps + p_bias.expand ->", sh(out_expand))
print("maps + p_bias(裸广播) ->", sh(out_bcast))
print("两者逐元素完全相等?", torch.equal(out_expand, out_bcast), " => .expand 只是把广播显式化，可省略")
print("逐元素加的总次数 = 2*17*37*37 =", B*P1*H*W)

# 共享性验证：两张图加上去的"增量"是不是同一份？
inc0 = out_expand[0] - maps[0]    # = p_bias[0]
inc1 = out_expand[1] - maps[1]    # = p_bias[0]
print("(图0增量 - 图1增量) 最大绝对差 =", (inc0 - inc1).abs().max().item(),
      " (=0 => 两张图共享同一份 p_bias)")

print()
print("="*72)
print("3) 加完再 softmax(dim=1=部件维)：每个 (b,h,w) 位置跨 17 个部件 Σ=1")
print("="*72)
probs = F.softmax(out_expand, dim=1)
s = probs.sum(dim=1)              # 沿部件维求和 -> [2,37,37]，理应全 1
print("softmax 后形状:", sh(probs), " 沿 dim=1 求和: min=%.6f max=%.6f (应≈1)"
      % (s.min().item(), s.max().item()))

print()
print("="*72)
print("4) 维度右对齐规则（broadcast：从最右一维往左逐维比）")
print("="*72)
print(" maps   : [ 2, 17, 37, 37]")
print(" p_bias : [ 1, 17, 37, 37]")
print("          37==37 | 37==37 | 17==17 | 2 vs 1 -> 1 撑成 2    => 结果 [2,17,37,37]")
