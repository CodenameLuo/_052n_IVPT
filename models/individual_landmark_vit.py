"""IndividualLandmarkViT – Vision Transformer extended with part prototypes.

Adds learnable part tokens, per-layer prototype biases, and soft-assignment
heads on top of a ``timm`` ViT backbone.  This is the core IVPT model.

Reference:
    timm VisionTransformer – https://github.com/huggingface/pytorch-image-models
"""

# ======================================
#
# 这是 IVPT 的核心模型：在一个(冻结的)DINOv2 ViT 主干外面包一层，让它在最后几个 block 里
# “长出部件(landmark/part)”，从而既能分类、又能解释“模型看了哪些部位”。
#
# 直觉版工作机制(细节在 forward 里，留到 Part 7)：
#   1) 图片切成 patch 过 ViT；本次 518×518、patch14 → 得到 37×37 的 patch 特征网格。
#   2) 在最后 layer_n=4 个 block，每层注入一组可学习的“原型(prototype)”token：
#      原型与每个 patch 特征算距离 → softmax → 得到“每个 patch 属于哪个原型”的注意力图；
#      再把这些原型按一个分类头软分配/归并到 num_landmarks 个“部件”上。
#   3) 把每个部件聚合出的特征拼回 token 序列继续过 block，最后对各部件特征做分类。
#   n_pro="17,14,11,8,5"：最后 5 个读出位点的原型数(逐层变少，17→14→11→8→5)，
#   最末位 5 = 4 个前景部件 + 1 个背景，所以 num_landmarks = 5-1 = 4。
#
# 本部分(P3)注释的是“静态结构”：__init__(各组件是什么)、compute_xq(距离图)、compute_feat(按图聚特征)、
# _pos_embed、convert_blocks_and_attention。真正把它们串起来跑的 forward 是训练的心脏，留到 Part 7。
#
# ======================================

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Union, Sequence

from timm.models import create_model
from timm.models.vision_transformer import Block, Attention

# compute_attention：从 qkv 复原注意力权重(仅分析方法用)
from utils.misc_utils import compute_attention
# 带 qkv 返回的 Block/Attention(会猴补丁替换主干的同名模块)
from .layers.transformer_layers import BlockWQKVReturn, AttentionWQKVReturn
# 每部件独立 MLP 分类头(本次用 linear，不走它)
from .layers.independent_mlp import IndependentMLPs
import torch.nn.functional as F


class IndividualLandmarkViT(torch.nn.Module):
    # init_model：已建好的 timm ViT 主干；其余都是 IVPT 专属开关(由 builder.init_ivpt_model 透传)
    #   n_pro="17,14,11,8,5"、modulation_type="layer_norm"、gumbel_softmax=True、classifier_type="linear"
    def __init__(self, init_model: torch.nn.Module, num_classes: int = 200,
                 part_dropout: float = 0.3, return_transformer_qkv: bool = False,
                 modulation_type: str = "original", gumbel_softmax: bool = False,
                 gumbel_softmax_temperature: float = 1.0, gumbel_softmax_hard: bool = False,
                 classifier_type: str = "linear", noise_variance: float = 0.0, n_pro: str = "") -> None:
        super().__init__()
        # 把 "17,14,11,8,5" 解析成整数列表 [17,14,11,8,5]
        self.n_pro = [int(n) for n in n_pro.split(',')]
        # 前景部件数 = 最末位 - 1(本次 5-1=4)；那多出来的 1 个是“背景”槽
        self.num_landmarks = self.n_pro[-1] - 1 # num_landmarks
        self.num_classes = num_classes
        # 给特征加高斯噪声的方差(本次 0，不加)
        self.noise_variance = noise_variance
        # 主干的前缀 token 数(cls + register)：DINOv2 reg4 是 1+4=5，它们不算 patch、聚特征时要先剔掉
        self.num_prefix_tokens = init_model.num_prefix_tokens
        self.num_reg_tokens = init_model.num_reg_tokens
        self.has_class_token = init_model.has_class_token
        self.no_embed_class = init_model.no_embed_class
        # 直接复用主干的 cls_token / reg_token(共享参数)
        self.cls_token = init_model.cls_token
        self.reg_token = init_model.reg_token
        # 注入原型的层数 = 列表长度-1(本次 4)：在最后 4 个 block 注入
        self.layer_n = len(self.n_pro) - 1
        self.gumbel_softmax = gumbel_softmax

        # === 直接“借用”主干的各个零件(patch 嵌入、位置编码、blocks、归一化等)，不另起炉灶 ===
        self.feature_dim = init_model.embed_dim          # 特征维 D(ViT-Base=768)
        self.patch_embed = init_model.patch_embed        # 把图切 patch 并线性嵌入
        self.pos_embed = init_model.pos_embed            # 位置编码
        self.pos_drop = init_model.pos_drop
        self.norm_pre = init_model.norm_pre
        self.blocks = init_model.blocks                  # ViT 的 Transformer block 列表(ViT-Base 共 12 个)
        self.norm = init_model.norm                      # 主干最后的 LayerNorm
        self.return_transformer_qkv = return_transformer_qkv
        # patch 网格的高/宽 = 图边长 // patch 边长(本次 518//14 = 37)，即 37×37 个 patch
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        # === IVPT 新增的可学习组件(只有这些 + 分类头 + 各 norm 会被训练，主干冻结) ===
        # p_token[i]：第 i 个注入位点的原型 token，形状 [1, n_pro[i], D]；共 layer_n+1=5 组
        # (前 4 组在 block 循环里注入，最后第 5 组(index -1)在循环结束后用于最终读出)
        self.p_token = nn.ParameterList([nn.Parameter(torch.zeros(1, self.n_pro[i], init_model.cls_token.shape[-1])) for i in range(self.layer_n+1)])
        # 原型初始化为小高斯(std=0.05)
        for i, p in enumerate(self.p_token):
            nn.init.normal_(p, std=0.05)
        # p_bias[i]：可学习的“空间偏置”，形状 [1, n_pro[i], H, W]；加到距离图上再 softmax，
        #   相当于给每个原型一个“倾向于关注图里哪块位置”的先验
        self.p_bias = nn.ParameterList([nn.Parameter(torch.zeros(1, self.n_pro[i], self.h_fmap, self.w_fmap)) for i in range(self.layer_n+1)])
        # p_linear[i]：D→D 线性层，把聚合出的部件特征投影一下再拼回 token 序列；4 个
        self.p_linear = nn.ModuleList([nn.Linear(in_features=init_model.cls_token.shape[-1], out_features=init_model.cls_token.shape[-1], bias=True) for i in range(self.layer_n)])
        # p_norm[i]：对 [num_landmarks, D] 做 LayerNorm，规整部件特征；4 个
        self.p_norm = nn.ModuleList([torch.nn.LayerNorm([self.num_landmarks, self.feature_dim]) for i in range(self.layer_n)])
        # p_classifier[i]：D→num_landmarks 的“软分配头”——把每个原型软分到 num_landmarks 个部件上；4 个(# TODO 是原作者留的)
        self.p_classifier = nn.ModuleList([nn.Linear(in_features=init_model.cls_token.shape[-1], out_features=self.num_landmarks, bias=True) for i in range(self.layer_n)]) # TODO

        # 把 [B, H*W, D] 的 patch 序列重新摊成 [B, H, W, D]，方便当成 2D 特征图处理
        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        self.gumbel_softmax_hard = gumbel_softmax_hard
        self.modulation_type = modulation_type
        # modulation：对最终的 [D, num_landmarks+1] 部件特征做 LayerNorm，分类前规整一下尺度
        self.modulation = torch.nn.LayerNorm([self.feature_dim, self.num_landmarks + 1])
        # part dropout：训练时按概率整条丢弃某些部件通道(正则，防止过度依赖个别部件)
        self.dropout_full_landmarks = torch.nn.Dropout1d(part_dropout)
        self.classifier_type = classifier_type
        # 分类头：本次 classifier_type='linear'，走下面的普通 Linear(D→类别数，无 bias)
        if classifier_type == "independent_mlp":
            self.fc_class_landmarks = IndependentMLPs(part_dim=self.num_landmarks, latent_dim=self.feature_dim,
                                                      num_lin_layers=1, act_layer=False, out_dim=num_classes,
                                                      bias=False, stack_dim=1)
        elif classifier_type == "linear":
            self.fc_class_landmarks = torch.nn.Linear(in_features=self.feature_dim, out_features=num_classes,
                                                      bias=False)
        else:
            raise ValueError("classifier_type not implemented")
        # 把主干所有 block/attention 的类替换成“带 qkv 返回”的版本(行为不变，见 transformer_layers.py)
        self.convert_blocks_and_attention()
        # 初始化新增层的权重(分类头 + p_linear)
        self._init_weights()

    # 初始化分类头：截断正态(std=0.02)初始化权重、bias 清零(本次是 linear 头)
    def _init_weights_head(self):
        # Initialize weights with a truncated normal distribution
        if self.classifier_type == "independent_mlp":
            self.fc_class_landmarks.reset_weights()
        else:
            torch.nn.init.trunc_normal_(self.fc_class_landmarks.weight, std=0.02)
            if self.fc_class_landmarks.bias is not None:
                torch.nn.init.zeros_(self.fc_class_landmarks.bias)

    # 初始化 p_linear 各层：同样截断正态 + bias 清零
    def _init_weights_linear(self):
        for layer in self.p_linear:
            torch.nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    # 统一初始化新增层(分类头 + p_linear)
    def _init_weights(self):
        self._init_weights_head()
        self._init_weights_linear()

    # 猴补丁：遍历所有子模块，把 timm 的 Block/Attention 实例的“类”就地换成带 qkv 返回的版本
    # 只改 __class__、不动权重，所以行为和原版一致，只是多了返回 qkv 的能力
    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = BlockWQKVReturn
            elif isinstance(module, Attention):
                module.__class__ = AttentionWQKVReturn

    # 加位置编码并拼上 cls/reg 等前缀 token(从 timm ViT 原版照搬，处理两种拼接顺序)
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        # 先把要拼到前面的前缀 token 备好(cls_token、reg_token)，按 batch 展开
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        # 情形一(no_embed_class，如 DeiT-3/big vision)：位置编码只加到 patch 上，再把前缀 token 拼到最前
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        # 情形二(原版 timm/JAX/DeiT)：位置编码本身含前缀位，所以先拼前缀 token 再整体加位置编码
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    # 算“每个原型 vs 每个 patch”的相似度图：用负的平方 L2 距离表示(越大越像)
    # 入参 x：patch 特征图 [B, C, H, W]；q：原型 [B, L, C](L 个原型)。出参 maps：[B, L, H, W]
    # 数学：||x-q||² = ||x||² - 2·(x·q) + ||q||²，再取负
    def compute_xq(self, x, q):
        # x·q：每个原型 l 与每个位置(h,w)的点积 -> [B, L, H, W]
        ab = torch.einsum('bchw,blc->blhw', x, q)
        # ||x||²：每个位置特征的模平方 -> [B,1,H,W]，再广播到 L 个原型
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, q.shape[1], -1, -1).contiguous()
        # ||q||²：每个原型的模平方 -> [B,L,1]，再广播/reshape 到 [B,L,H,W]
        a_sq = q.pow(2).sum(-1, keepdim=True)
        a_sq = a_sq.expand(-1, -1, x.shape[-2] * x.shape[-1])
        a_sq = a_sq.view(x.shape[0], q.shape[1], x.shape[-2], x.shape[-1])
        # 合成平方距离，取负作为相似度图(距离越小 -> 值越大 -> 越“像”这个原型)
        dist = b_sq - 2 * ab + a_sq
        maps = -dist
        return maps

    # 按相似度图把 patch 特征聚合成“每个槽一个特征向量”(硬分配 + 加权平均)
    # 入参 maps：[B, N, H, W](N 个槽的相似度图)；x：patch 特征 [B, D, H, W]
    # 出参 all_features：[B, N, D](每个槽的平均特征)；count_map：[B, N, 1, 1](每个槽分到的总权重)
    def compute_feat(self, maps, x):
        N = maps.shape[1]
        # 每个位置只归给“最像的那个槽”(argmax 硬分配)，再乘回 maps 值作为权重；其余槽在该位置为 0
        # one_hot_map：[B, N, H, W]，非零处=该位置对其所属槽的相似度
        one_hot_map = F.one_hot(torch.argmax(maps, dim=1), num_classes=N).permute(0, 3, 1, 2)*maps
        # 用权重图给特征加权：[B,1,N,H,W] × [B,D,1,H,W] -> [B, D, N, H, W]
        all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()
        # 对空间(H,W)求和，得每个槽的“特征和” -> [B, N, D]
        sum_pool = all_features.sum(dim=(3, 4)).permute(0, 2, 1)
        # 每个槽分到的总权重(也是“占了多少像素的量度”) -> [B, N, 1, 1]
        count_map = one_hot_map.sum(dim=(2, 3), keepdim=True)
        # 防止除以 0：某槽一个位置都没分到时，把分母当 1
        count_map_ = count_map + (count_map == 0).float()
        # 特征和 / 总权重 = 加权平均特征 -> [B, N, D]
        all_features = sum_pool / count_map_.squeeze(-1)
        return all_features, count_map

    # ============================================================================
    # forward 是整个训练的“心脏”：把上面的静态组件(原型/距离图/聚合/分类头)真正串起来跑一遍前向，
    # 包含“最后 4 个 block 注入原型 -> 算注意力图 -> gumbel 软分配 -> 聚合部件特征 -> 拼回序列 ->
    # 最终读出并分类”的完整逻辑。它在 Part 7(训练单 batch 那一步)里结合 _run_batch 一起逐行细注，
    # 这里(P3)先不展开，保持原样。
    # ============================================================================
    def forward(self, x: Tensor) -> tuple[Any, Any, Any, Any, int | Any] | tuple[Any, Any, Any, Any, int | Any]:
        # 缩写约定：B=batch，D=特征维(768)，H=W=37(patch 网格)，L=H*W=1369，N=num_landmarks(4 个前景部件)
        # 前缀 token 数=5(cls1+reg4)；下面把整个前向分成 3 段：①嵌入 ②最后 4 个 block 注入原型 ③最终读出+分类
        # === 第①段：切 patch + 加位置编码 ===
        x = self.patch_embed(x)        # 图 -> patch token [B, L, D]
        x = self._pos_embed(x)         # 加位置编码 + 拼前缀 token -> [B, 5+L, D]
        # x = self.norm_pre(x)
        x_len = x.shape[1]             # 记下“前缀+patch”的 token 总数(后面注入的部件 token 不算在内)
        # 四个缓冲区：x_buffer 存各注入位点的特征、q_buffer 存对应原型、m_buffer 存各层部件空间图、qm_buffer 存软分配
        x_buffer, q_buffer, m_buffer, qm_buffer = [], [], [], []
        l = len(self.blocks)           # block 总数(ViT-Base=12)

        # === 第②段：逐 block；前 l-4 个原样过，最后 4 个注入原型 ===
        for i, block in enumerate(self.blocks):
            # 前 8 个 block：普通 Transformer，不动
            if i < l - self.layer_n:
                x = block(x)
            # 后 4 个 block(i=8,9,10,11)：注入原型、生成部件图
            else:
                x = x[:, :x_len]                       # 把上一轮末尾拼进来的部件 token 切掉，只留前缀+patch
                q_index = i - l + self.layer_n         # 当前注入位点序号 0/1/2/3
                q = self.p_token[q_index].expand(x.shape[0], -1, -1) # 取该位点原型 [B, n_pro[i], D] # + q_pre.detach()
                x_buffer.append(x)                     # 存下当前特征(供第③段统一再算一遍图)
                q_buffer.append(q)
                # q = q.detach() # TODO
                # 取 patch 特征图：detach(不让下面算图回传主干)→ 最终 norm → 去掉前缀 → 摊成 2D
                x_ = self.norm(x.detach())
                x_ = x_[:, self.num_prefix_tokens:, :]  # [B, L, D]  去掉 5 个前缀 token
                x_ = self.unflatten(x_)  # [B, H, W, D]
                x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

                # 原型 vs patch 的相似度图(负 L2)，[B, n_pro[i], H, W]
                maps = self.compute_xq(x_, q) # [B, P+1, H, W]
                p_bias = self.p_bias[q_index].expand(x.shape[0], -1, -1, -1)   # 加上该位点的可学习空间偏置
                # 沿“原型”维 softmax：每个 patch 得到一个“属于各原型”的分布(各列和为 1)
                maps = torch.nn.functional.softmax(maps+p_bias, dim=1) #

                # 按图把 patch 特征聚到每个原型上(maps.detach 不回传)：q_x [B, n_pro[i], D]；count_map 记每原型分到多少
                q_x, count_map = self.compute_feat(maps.detach(), x_) # [B, P+1, D] [B, P+1, 1, 1] #
                # 约定最后一个原型是“背景”；前 n_pro[i]-1 个是前景原型。软分配头把每个前景原型映到 N 个部件上
                q_c = self.p_classifier[q_index](q_x[:, :-1]) # [B, P, N]  P=前景原型数

                # 没分到任何 patch 的原型做个掩码(下面给它们填均匀分配)
                count_mask = (count_map[:, :-1]==0).squeeze(-1).expand(-1, -1, q_c.shape[-1]) # [B, P, N]

                # 关键一步：把“前景原型 -> 部件”的分配做 gumbel-softmax(本次开)，让分配接近 one-hot 又可导
                if self.gumbel_softmax:
                    q_maps = torch.nn.functional.gumbel_softmax(q_c, dim=-1, tau=self.gumbel_softmax_temperature, hard=self.gumbel_softmax_hard)
                else:
                    q_maps = torch.nn.functional.softmax(q_c, dim=-1)
                # 空原型用均匀分配(1/N)兜底
                q_maps = q_maps.masked_fill(count_mask, 1/q_c.shape[-1])
                qm_buffer.append(q_maps)                # 存下软分配(评估可视化要用)

                q_maps_expanded = q_maps.unsqueeze(-1)  # (B, P, N, 1)
                f_maps = maps.flatten(2) # (B, P+1, L)  把空间图摊平成向量

                # —— 把“前景原型”按分配归并成 N 个部件：先特征、再空间图 ——
                q_maps_detach = q_maps_expanded.detach()
                # 每个前景原型的特征按分配权重铺到 N 个部件上：(B,P,1,D)×(B,P,N,1)=(B,P,N,D)
                q_x_weighted = q_x[:, :-1].unsqueeze(2) * q_maps_detach  # (B, P, 1, D) × (B, P, N, 1) = (B, P, N, D)
                q_x = q_x_weighted.sum(dim=1)  # (B, N, D)  对原型求和 -> 每个部件一个特征
                q_maps_sum = q_maps_detach.sum(1).squeeze(1)   # 每个部件收到的总权重 (B, N, 1)
                q_x = q_x / q_maps_sum                         # 归一化 -> 部件的加权平均特征

                f_bg = f_maps[:, -1:] # (B, 1, L)  背景原型的空间图单独留出
                # 前景原型的空间图也按同样分配归并到 N 个部件：(B,P,1,L)×(B,P,N,1)=(B,P,N,L)
                f_maps_weighted = f_maps[:, :-1].unsqueeze(2) * q_maps_expanded  # (B, P, 1, L) × (B, P, N, 1) = (B, P, N, L)
                f_maps = f_maps_weighted.sum(dim=1)  # (B, N, L)  得到 N 个部件的空间图

                f_maps = torch.cat([f_maps, f_bg], dim=1) # (B, N+1, L)  拼回背景 -> N+1 个槽

                m_buffer.append(f_maps)                 # 存该层的部件空间图(consistency 损失要跨层比对)
                # 部件特征过 LayerNorm + Linear，作为“部件 token”拼回序列，让它们参与下一层注意力
                q_x = self.p_linear[q_index](self.p_norm[q_index](q_x))
                x = torch.cat([x, q_x], dim=1)          # [B, x_len+N, D]
                x = block(x)                            # 带着部件 token 一起过这个 block

        # === 第③段：最终读出 —— 用第 5 组原型 + 各位点缓存，统一再算一遍部件图，并分类 ===
        q = self.p_token[-1].expand(x.shape[0], -1, -1)   # 最后一组原型 [B, 5, D](5=N+1)
        x = x[:, :x_len]                                  # 切掉末尾部件 token
        x_buffer.append(x)                                # 此时 x_buffer/q_buffer 各 5 项(4 注入位点 + 这个最终位点)
        q_buffer.append(q)

        # 对 5 个缓存位点各算一张“部件图”：流程同上(算距离图 -> +偏置 -> softmax -> 加 eps 重归一化)
        maps_list = []
        for i, (x, q) in enumerate(zip(x_buffer, q_buffer)):
            x_ = self.norm(x.detach())
            x_ = x_[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
            x_ = self.unflatten(x_)  # [B, H, W, embed_dim]
            x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]

            maps = self.compute_xq(x_, q)

            p_bias = self.p_bias[i].expand(x.shape[0], -1, -1, -1)
            maps = torch.nn.functional.softmax(maps+p_bias, dim=1)  # [B, num_landmarks + 1, H, W] #
            maps = maps + 1e-6                         # 加微小量再归一化，避免后续 log/除零
            maps = maps / maps.sum(dim=1, keepdim=True)
            maps_list.append(maps)

        maps = maps_list[-1]                            # 取最终位点的部件图 [B, N+1, H, W]
        f_maps = maps.flatten(2)
        m_buffer.append(f_maps)                         # m_buffer 现在 5 项(consistency 用最后一项当 target)

        # 用最终部件图把最终层 patch 特征聚成每个部件一个特征
        x = self.unflatten(self.norm(x_buffer[-1])[:, self.num_prefix_tokens:, :]).permute(0, 3, 1, 2).contiguous()
        all_features = self.compute_feat(maps, x)[0]    # [B, N+1, D]
        all_features = all_features.permute(0, 2, 1)    # [B, D, N+1]

        # Modulate the features
        # 调制(LayerNorm)规整部件特征尺度
        all_features_mod = self.modulation(all_features)  # [B, embed_dim, num_landmarks + 1]

        # Classification based on the landmark features
        # 只取 N 个前景部件特征(去掉背景)送进分类头：每个部件各出一套类别分数 -> [B, num_classes, N]
        scores = self.fc_class_landmarks(
            all_features_mod[..., :-1].permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        map_feat = maps_list

        # 返回：①调制后部件特征 ②maps_list 的倒数第 3 张图 ③各部件类别分数 ④全部部件图列表 ⑤(m_buffer, qm_buffer)
        # _run_batch(P7b)会用：scores.mean 出最终 logits；map_feat 算 TV/presence/等变/像素熵；m_buffer 算 consistency
        return all_features_mod, maps_list[-3], scores, map_feat, (m_buffer, qm_buffer)

    # ============================================================================
    # ↓↓↓ 以下 get_specific_intermediate_layer / _intermediate_layers / get_intermediate_layers
    # 都【不在本次训练路径上】↓↓↓ 它们是“取某些中间层输出/注意力”给分析、可视化、调试用的(仿 DINO 接口)，
    # 只有 return_transformer_qkv=True 时才用得上(默认 False)。训练 forward 完全不调它们，这里保持原样。
    # ============================================================================
    def get_specific_intermediate_layer(
            self,
            x: torch.Tensor,
            n: int = 1,
            return_qkv: bool = False,
            return_att_weights: bool = False,
    ):
        num_blocks = len(self.blocks)
        attn_weights = []
        if n >= num_blocks:
            raise ValueError(f"n must be less than {num_blocks}")

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        if n == -1:
            if return_qkv:
                raise ValueError("take_indice cannot be -1 if return_transformer_qkv is True")
            else:
                return x

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)

                if return_att_weights:
                    attn_weight, _ = compute_attention(qkv)
                    attn_weights.append(attn_weight.detach())
            else:
                x = blk(x)
            if i == n:
                output = x.clone()
                if self.return_transformer_qkv and return_qkv:
                    qkv_output = qkv.clone()
                break
        if self.return_transformer_qkv and return_qkv and return_att_weights:
            return output, qkv_output, attn_weights
        elif self.return_transformer_qkv and return_qkv:
            return output, qkv_output
        elif self.return_transformer_qkv and return_att_weights:
            return output, attn_weights
        else:
            return output

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        if self.return_transformer_qkv:
            qkv_outputs = []
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        for i, blk in enumerate(self.blocks):
            if self.return_transformer_qkv:
                x, qkv = blk(x, return_qkv=True)
            else:
                x = blk(x)
            if i in take_indices:
                outputs.append(x)
                if self.return_transformer_qkv:
                    qkv_outputs.append(qkv)
        if self.return_transformer_qkv:
            return outputs, qkv_outputs
        else:
            return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> tuple[tuple, Any]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        if self.return_transformer_qkv:
            outputs, qkv = self._intermediate_layers(x, n)
        else:
            outputs = self._intermediate_layers(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return_out = tuple(zip(outputs, prefix_tokens))
        else:
            return_out = tuple(outputs)

        if self.return_transformer_qkv:
            return return_out, qkv
        else:
            return return_out


# ============================================================================
# ↓↓↓ 以下 ivpt_vit_bb / ivptnet_vit_bb 是【坏死代码，不在训练路径上】↓↓↓
# 它们给 IndividualLandmarkViT 传了 num_landmarks / modulation_orth 这些它的 __init__ 根本不接收的参数，
# 真调用会 TypeError。只有 builder.py 里那几个同样不在训练路径的 hub 加载器会引用它们。保持原样、不展开。
# ============================================================================
def ivpt_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="layer_norm", gumbel_softmax=True,
                                  modulation_orth=True)
    return model


def ivptnet_vit_bb(backbone, img_size=224, num_cls=200, k=8, **kwargs):
    base_model = create_model(
        backbone,
        pretrained=False,
        img_size=img_size,
    )

    model = IndividualLandmarkViT(base_model, num_landmarks=k, num_classes=num_cls,
                                  modulation_type="original")
    return model
