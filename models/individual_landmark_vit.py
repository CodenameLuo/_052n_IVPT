"""IndividualLandmarkViT – Vision Transformer extended with part prototypes.

Adds learnable part tokens, per-layer prototype biases, and soft-assignment
heads on top of a ``timm`` ViT backbone.  This is the core IVPT model.

Reference:
    timm VisionTransformer – https://github.com/huggingface/pytorch-image-models
"""

# ======================================

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Union, Sequence

from timm.models import create_model
from timm.models.vision_transformer import Block, Attention

# ======================================

from utils.misc_utils import compute_attention
from .layers.transformer_layers import BlockWQKVReturn, AttentionWQKVReturn
from .layers.independent_mlp import IndependentMLPs

# ======================================

class IndividualLandmarkViT(torch.nn.Module):
    def __init__(
        self, 
        init_model: torch.nn.Module, 
        num_classes: int = 200, 
        part_dropout: float = 0.3, 
        return_transformer_qkv: bool = False, 
        modulation_type: str = "original", 
        gumbel_softmax: bool = False, 
        gumbel_softmax_temperature: float = 1.0, 
        gumbel_softmax_hard: bool = False, 
        classifier_type: str = "linear", 
        noise_variance: float = 0.0, 
        n_pro: str = ""
    ) -> None:
        super().__init__()

        # 例： --n_pro 17,14,11,8,5

        # 逗号串解析成 list：5 个层级，每层的原型数
        self.n_pro = [int(n) for n in n_pro.split(',')]

        # 在最后 4 个 transformer 块 插入部件机制
        self.layer_n = len(self.n_pro) - 1

        # 最后一层(粗粒度) 5 个原型 = 4 个前景部件 + 1 个背景
        # 粗部件数 4
        self.num_landmarks = self.n_pro[-1] - 1 # num_landmarks

        # === 参数可分为 4 类 ===start

        # 1. 本模型自己的超参/开关

        # 分类类别数。作用：最终分类头 fc_class_landmarks 的 out_features。例：CUB=200
        self.num_classes = num_classes
        # 惰性参数：存了但当前 forward 没用到（PDiscoFormer 遗留）
        self.noise_variance = noise_variance

        # 2. 骨架读的"配置数字/标志"（int/bool）

        # 序列开头非 patch 的 token 数 = cls + reg。例：1+4=5
        # 作用：forward 里 x_[:, num_prefix_tokens:] 切掉前缀、只留 patch 再摊成 37×37
        self.num_prefix_tokens = init_model.num_prefix_tokens
        # register token 个数（DINOv2 reg4 的"寄存器"）。例：4
        self.num_reg_tokens = init_model.num_reg_tokens
        # 骨架是否带 [CLS]。例：True
        self.has_class_token = init_model.has_class_token
        # True → 位置编码不覆盖 cls/reg 位置。例：True
        # 作用：决定 _pos_embed 走"先给 patch 加 pos、再把 cls/reg 拼到前面"这条分支
        self.no_embed_class = init_model.no_embed_class
        # 可学习 [CLS] token（与骨架同一对象）。例：(1,1,768)
        # 作用：_pos_embed 里 expand 到 batch 后拼进序列头
        self.cls_token = init_model.cls_token
        # 可学习 register tokens（吸全局信息/压低特征图 artifact）。例：(1,4,768)
        self.reg_token = init_model.reg_token
        # 开关：细原型→粗部件的分配 q_maps 用 gumbel-softmax 还是普通 softmax。例：True
        self.gumbel_softmax = gumbel_softmax

        # 3. 从 backbone(init_model) 搬来的"可学习零件"
        # （self.x = init_model.x 是同对象引用＝直接继承预训练权重，并注册成 IVPT 子模块）

        # 嵌入维 D，全模型到处用。例：768
        self.feature_dim = init_model.embed_dim
        # 切 patch+线性投影：Conv2d(3,768,k=14,s=14)
        # 作用：forward 第一步 图→patch token。例：(518,518)→37×37=1369 个 768 维 token
        self.patch_embed = init_model.patch_embed
        # 位置编码参数（仅 patch、不含前缀）。例：(1,1369,768)
        self.pos_embed = init_model.pos_embed
        # 位置编码后的 dropout。例：Dropout(p=0.0)＝本配置等于没有
        self.pos_drop = init_model.pos_drop
        # 进 block 前的预归一化
        # DINOv2 没用 pre-norm → Identity
        # 而且主 forward 里这步还被注释掉了
        self.norm_pre = init_model.norm_pre
        # backbone 的 12 个 transformer block（Sequential）
        # 作用：主干，forward 逐块过
        # __init__ 末尾 convert_blocks_and_attention 把每块类改成 BlockWQKVReturn
        self.blocks = init_model.blocks
        # backbone 最终 LayerNorm(768, eps=1e-6)
        # 作用：取部件特征前的归一化（forward 多处 self.norm(x.detach())）
        self.norm = init_model.norm
        # 开关：中间层取数接口是否吐 qkv（WIP 路径，主 forward 不走）。例：False
        self.return_transformer_qkv = return_transformer_qkv

        # 4. 推出来的派生量

        # 特征图高＝518//14。例：37
        # 特征图宽＝518//14。例：37
        # 作用：unflatten 把 1369 个 patch token 摊回 37×37；也正是 p_bias 的空间两维
        self.h_fmap = int(self.patch_embed.img_size[0] // self.patch_embed.patch_size[0])
        self.w_fmap = int(self.patch_embed.img_size[1] // self.patch_embed.patch_size[1])

        # === 参数可分为 4 类 ===end

        # ======================================
        # 可解释部件 模块参数 系列

        # 各网络层级的可学习的 原型/prompt 查询 token
        # 每层一组，每组 n_pro[i] 个 token，每个 token 的维度和 cls_token 的维度一样
        # 
        # p_token[0]: shape=(1, 17, 768)
        # p_token[1]: shape=(1, 14, 768)
        # p_token[2]: shape=(1, 11, 768)
        # p_token[3]: shape=(1, 8, 768)
        # p_token[4]: shape=(1, 5, 768)
        # 
        # 维度0：batch 占位维度，留个 1 等 forward 广播
        # 维度1：该层级的原型个数，含背景
        # 维度2：嵌入维度 = init_model.cls_token.shape[-1]
        self.p_token = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, self.n_pro[i], init_model.cls_token.shape[-1])
                ) for i in range(self.layer_n + 1)
            ]
        )
        # 把每个 token 重填成均值 0、标准差 0.05 的正态分布
        for i, p in enumerate(self.p_token):
            nn.init.normal_(p, std=0.05)

        # 加到 原型-patch 相似度图 上的可学习空间先验 (H=W=37)
        # 
        # p_bias[0]: shape=(1, 17, 37, 37)
        # p_bias[1]: shape=(1, 14, 37, 37)
        # p_bias[2]: shape=(1, 11, 37, 37)
        # p_bias[3]: shape=(1, 8, 37, 37)
        # p_bias[4]: shape=(1, 5, 37, 37)
        # 
        # 维度0：batch 占位维度，留个 1 等 forward 广播
        # 维度1：该层级的原型个数，含背景
        # 维度2：特征图高
        # 维度3：特征图宽
        self.p_bias = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(1, self.n_pro[i], self.h_fmap, self.w_fmap)
                ) for i in range(self.layer_n + 1)
            ]
        )

        # 把聚合到的 原型特征 投影成 prompt token
        # 
        # p_linear[0/1/2/3]:
        # weight.shape=(768, 768)
        # bias.shape=(768,)
        self.p_linear = nn.ModuleList(
            [
                nn.Linear(
                    in_features=init_model.cls_token.shape[-1], 
                    out_features=init_model.cls_token.shape[-1], 
                    bias=True
                ) for i in range(self.layer_n)
            ]
        )

        # prompt 前的归一化 (num_landmarks=4)
        # 
        # p_norm[0/1/2/3]：每个 LayerNorm，含 weight(γ)+bias(β)
        # weight.shape=(4, 768) —— 缩放
        # bias.shape=(4, 768) —— 平移
        # 
        # LayerNorm：
        # 1. 归一化：
        #    把每个样本的一组激活按「减均值、除标准差」拉成均值0、标准差1
        # 2. 仿射：
        #    再用可学习的 γ(缩放)、β(平移) 调成网络想要的尺度，
        #    它的核心作用是抹掉输入的"绝对大小"、只留"相对花样"，
        #    给每一层一个稳定可预期的输入尺度，
        #    从而让深层网络（尤其 transformer）训练稳、收敛快、不爆炸不消失
        # 
        # 作用1：
        # 把任意尺度的一组数，统一拉到「均值0 / 标准差1」
        # 不管原始数多大，过完 LayerNorm（默认 γ/β）这组数就是 0 均值、1 标准差
        # 等于给所有激活强行统一了量纲
        # 
        # 作用2：
        # 对输入的整体"缩放 + 平移"免疫 —— 只保留相对模式
        # 例：
        # x        = [10,   12,   14,   16  ]  -> LN = [-1.3416, -0.4472, 0.4472, 1.3416]
        # 100*x+50 = [1050, 1250, 1450, 1650]  -> LN = [-1.3416, -0.4472, 0.4472, 1.3416]
        # LayerNorm 能稳住尺度：
        # 无论前面的层吐出多大幅度的激活，到了 LayerNorm 这儿一律被打回同一把尺子
        # 
        # 作用3：
        # γ/β 是"反悔按钮"——尺度能调回来，归一化不损失表达力
        # 万一「抹掉尺度」这件事抹过头了（某些维度的幅度其实有用），
        # 可学习的 γ、β 能把输出的标准差调回 ≈γ、均值调回 ≈β。
        # 即：先无脑归一化到 0/1，再让网络自己学一个想要的尺度。
        # 所以加 LayerNorm 不会因为"强行归一化"而削弱模型能力
        # 最坏情况它能学出 γ=σ、β=μ 把归一化完全还原
        # 
        # 作用4：
        # 跨深度稳住激活幅度 —— 深层网络/transformer 离不开它
        # 模拟堆 8 层「每层把范数 ×~1.5」的线性变换，对比加不加 LayerNorm：
        #  层 |   无LN 激活std |  有LN 激活std
        #   1 |   1.49e+00    |    1.59
        #   2 |   2.42e+00    |    1.63
        #   3 |   3.74e+00    |    1.53
        #   4 |   5.78e+00    |    1.55
        #   6 |   1.34e+01    |    1.50
        #   8 |   2.64e+01    |    1.44
        # → 无 LN：std 随层数指数漂移(8 层就涨到 26，再深会爆)；有 LN：每层先打回 ~1，全程稳在 ~1.5
        # 没有归一化，激活幅度会随深度 指数爆炸或消失，梯度跟着乱掉、没法训深
        # 每层插一个 LayerNorm，等于在每层入口「把尺度清零重置」
        # 于是无论多深，每层看到的输入都在可控范围
        # 这正是 transformer 每个子层都前置 LayerNorm（Pre-LN）的原因
        # 
        # 为什么用 LayerNorm 而不是 BatchNorm
        # LayerNorm 的 μ/σ 是在单个样本内部、对特征维算的
        # 不依赖 batch、batch=1 也能用、训练/推理行为完全一致、对变长序列友好
        # BatchNorm 的统计量是 跨样本（沿 batch 维）算的
        # 依赖 batch 大小、要维护滑动均值、变长序列别扭
        # 所以 transformer 类几乎一律用 LayerNorm
        # 
        # IVPT 的 p_norm = LayerNorm([4, 768]) 作用就是上面这套
        # 用在「聚合出 4 个 prompt 特征 → 先 p_norm 归一化 → 再 p_linear 投影 → cat 进序列」这一步
        # 不管从原型区域池化出来的特征幅度是大是小（不同图、不同层差异可能很大）
        # p_norm 都把这 4 个 prompt 拉到统一尺度再喂给后面的线性层和 transformer，稳住 prompt 的量纲、让微调更稳
        # 它特殊在 normalized_shape 写成两维 [4,768] → 4 个 prompt 是「绑在一起」联合归一化的
        # 
        # 输入 q_x = (B, 4, 768)：对每个样本，把它那 4×768=3072 个数当成一个整体算一个 mean、一个 std 来归一化
        # 这里写成 [4,768]，把 4 个 prompt token 在归一化时绑在一起统计
        # 是有意为之的设计差异（语义上让 4 个部件 prompt 共享同一套均值方差尺度）
        self.p_norm = nn.ModuleList(
            [
                torch.nn.LayerNorm(
                    [self.num_landmarks, self.feature_dim]
                ) for i in range(self.layer_n)
            ]
        )

        # 跨层对齐头：把 细原型 分配到 4 个 粗部件 桶 (fine -> coarse)
        # p_classifier[0/1/2/3]: 
        # weight.shape=(4, 768)
        # bias.shape=(4,)
        # 
        # 和 p_linear 的区别：
        # p_linear 是"投影"，p_classifier 是"分类"
        self.p_classifier = nn.ModuleList(
            [
                nn.Linear(
                    in_features=init_model.cls_token.shape[-1], 
                    out_features=self.num_landmarks, 
                    bias=True
                ) for i in range(self.layer_n)
            ]
        ) # TODO

        # ======================================

        # 把"扁平 patch 序列"摊回 2D 网格：对 dim=1 拆成 (37,37)
        # 作用：forward 里 [B,1369,768] --unflatten--> [B,37,37,768]
        # 再 permute 重新排列成 [B,768,37,37]， 好和原型算空间相似度
        # 例：(2,1369,768) -> (2,37,37,768)
        self.unflatten = nn.Unflatten(1, (self.h_fmap, self.w_fmap))
        # gumbel-softmax 的温度 τ：越小越接近 one-hot(硬)、越大越平滑
        # 作用：forward 里 gumbel_softmax(q_c, tau=…)
        # 例：1.0
        self.gumbel_softmax_temperature = gumbel_softmax_temperature
        # 是否用"直通式硬 one-hot" (前向硬采样、反向仍可导)
        # 例：False
        self.gumbel_softmax_hard = gumbel_softmax_hard
        # 存了但 forward 不按它分支
        # 无论传什么，下面 self.modulation 恒为 LayerNorm
        self.modulation_type = modulation_type

        # LayerNorm([D,5])，对部件特征做调制
        # 
        # "调制"层 = LayerNorm([D, 部件数+1]) = LayerNorm([768, 5])
        # normalized_shape 两维 → 对 768×5 整块联合归一化(含背景那 1 个)
        # 作用：forward 末尾 all_features[B,768,5] --modulation--> [B,768,5]
        #       给每个(特征维,部件)一对可学习 γ/β，分类前先稳尺度/重加权
        # 例：weight/bias 形状 (768,5)，参数 7680；(2,768,5)->(2,768,5)
        self.modulation = torch.nn.LayerNorm(
            [self.feature_dim, self.num_landmarks + 1]
        )

        # Dropout1d：本意是训练时整条"部件通道"随机置零(part-dropout 增鲁棒)
        # 但 forward 没调用它
        # 例：Dropout1d(p=0.3)
        self.dropout_full_landmarks = torch.nn.Dropout1d(part_dropout)

        # 新分类头(classifier_type=independent_mlp 时换成逐部件 MLP)
        # 
        # 选哪种最终分类头：'independent_mlp' 或 'linear'
        self.classifier_type = classifier_type
        if classifier_type == "independent_mlp":
            # 逐部件独立分类头：4 条互不共享的 Linear(768→200, 无bias)
            # 作用：每个部件用自己的权重投到类别分
            # 例：IndependentMLPs(part_dim=4)，总参数 4*768*200 = 614400；(2,4,768)->(2,4,200)
            self.fc_class_landmarks = IndependentMLPs(
                # 4 个部件，各配一条独立 Linear
                part_dim=self.num_landmarks, 
                # 768
                latent_dim=self.feature_dim, 
                # 单层
                num_lin_layers=1, 
                # 无激活层
                act_layer=False, 
                # 200
                out_dim=num_classes, 
                bias=False, 
                stack_dim=1
            )
        elif classifier_type == "linear":
            # 共享分类头：一条 Linear(768→200,无bias)，4 个部件共用同一套权重
            # 例：weight (200,768)、bias=None、参数 153600
            # (2,4,768) -> (2,4,200)
            self.fc_class_landmarks = torch.nn.Linear(
                # 768
                in_features=self.feature_dim, 
                # 200
                out_features=num_classes, 
                bias=False
            )
        else:
            raise ValueError("classifier_type not implemented")

        # 遍历所有子模块
        # 把 timm 原生 Block/Attention 就地改类成 BlockWQKVReturn/AttentionWQKVReturn
        # 只换 __class__、不重建权重
        # 作用：让借来的 12 个 block 能返回 qkv、能接 prompt
        # 例：blocks[0] 类型变 BlockWQKVReturn
        self.convert_blocks_and_attention()
        # 只初始化"新增的两处"
        # fc_class_landmarks(trunc_normal .02 / 无bias) 和 p_linear(trunc_normal .02 / bias=0)
        # 不碰 modulation / p_norm / p_classifier / p_bias / p_token 
        # 它们各自保留默认或前面已设的初值
        # 例：fc weight std=0.01998；p_linear[0].bias 全 0
        self._init_weights()

    def _init_weights_head(self):
        # Initialize weights with a truncated normal distribution
        if self.classifier_type == "independent_mlp":
            self.fc_class_landmarks.reset_weights()
        else:
            torch.nn.init.trunc_normal_(self.fc_class_landmarks.weight, std=0.02)
            if self.fc_class_landmarks.bias is not None:
                torch.nn.init.zeros_(self.fc_class_landmarks.bias)

    def _init_weights_linear(self):
        for layer in self.p_linear:
            torch.nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

    def _init_weights(self):
        self._init_weights_head()
        self._init_weights_linear()

    def convert_blocks_and_attention(self):
        for module in self.modules():
            if isinstance(module, Block):
                module.__class__ = BlockWQKVReturn
            elif isinstance(module, Attention):
                module.__class__ = AttentionWQKVReturn

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def compute_xq(
        self, 
        x, 
        q
    ):
        # x：特征图 [B, C, H, W] = [2, 768, 37, 37]
        # q：原型   [B, L, C]    = [2, 17, 768]
        # 用  ‖a-b‖² = ‖a‖² - 2·a·b + ‖b‖² 拆成三项，避免拼出 q-x 的 [B, L, C, H, W] 大张量（省峰值显存）

        # a·b 点积 Σ_c ( x[c,h,w] * q[l,c] ) -> [2, 17, 37, 37]
        # c (768) 被求和掉，b 留，h w 来自 x，l 来自 q
        ab   = torch.einsum('bchw,blc->blhw', x, q)

        # ‖patch‖² = Σ_c x² -> [2, 1, 37, 37] 只随位置
        b_sq = x.pow(2).sum(1, keepdim=True)
        # 广播到每个原型 -> [2, 17, 37, 37] (同位置跨原型同值)
        b_sq = b_sq.expand(-1, q.shape[1], -1, -1).contiguous()

        # ‖原型‖² = Σ_c q² -> [2, 17, 1] 只随原型
        a_sq = q.pow(2).sum(-1, keepdim=True)
        # 铺到 37*37=1369 个位置 -> [2, 17, 1369]
        a_sq = a_sq.expand(-1, -1, x.shape[-2] * x.shape[-1])
        # 摊回 -> [2, 17, 37, 37]（同原型跨位置同值）
        a_sq = a_sq.view(x.shape[0], q.shape[1], x.shape[-2], x.shape[-1])

        # ‖q-x‖² -> [2,17,37,37]（理论上≥0）
        dist = b_sq - 2 * ab + a_sq
        # 负距离=相似度（越大=越像）-> [2, 17, 37, 37]
        maps = -dist

        return maps
    
    def compute_feat(
        self, 
        maps, 
        x
    ):
        # maps：[B, N, H, W] = [2,  17, 37, 37]，softmax 后的软分配图
        # x：   [B, C, H, W] = [2, 768, 37, 37]，特征图

        # N=17，部件数，含背景
        N = maps.shape[1]

        # argmax(maps, dim=1)   [2, 17, 37, 37] -> [2, 37, 37]
        # 每个 patch 在 17 个部件里挑分最高的 (可以这么来看：[2, 37, 37, 17]) -> 硬分配索引 0...16
        # 
        # one_hot(..., 17)   [2, 37, 37] -> [2, 37, 37, 17]
        # 每 patch 一个长 17 独热向量（命中=1，其余=0）
        # 
        # permute(0,3,1,2)   [2, 37, 37, 17] -> [2, 17, 37, 37]
        # 把部件维挪回 dim1，对齐 maps
        # 
        # * maps             逐元素 -> [2, 17, 37, 37]
        # one_hot_map[b,p,h,w] = maps[b,p,h,w]（若 p 是该 patch argmax，否则 0）
        # 
        # 把 maps「沿部件维只留最大那个、其余清零」，留的那个仍是它的软值
        # 硬分区 + 软置信：
        # 每 patch 只归它最像的唯一部件（17 部件把 1369 个 patch 切成互不重叠区域，像分割掩码）
        # 归属强度用软置信(softmax 概率)，不是 1
        one_hot_map = F.one_hot(torch.argmax(maps, dim=1), num_classes=N).permute(0, 3, 1, 2) * maps

        # one_hot_map.unsqueeze(1)   [2, 17, 37, 37] -> [2, 1 ,17, 37, 37]
        # x.unsqueeze(2)            [2, 768, 37, 37] -> [2, 768, 1, 37, 37]
        # 广播相乘                                   -> [2, 768, 17, 37, 37]
        # all_features[b,c,p,h,w] = one_hot_map[b,p,h,w] · x[b,c,h,w]
        # 把每个 patch 的 768 维特征 按它的部件归属权重缩放，并按部件 p 摊成 17 份（只有归属 p 的位置非零）
        # 这里物化了一张 5D 大张量 = 2·768·17·37·37 = 35,747,328 个数 ≈ 143MB
        # 注意：和 compute_xq 刻意「不物化 [B,L,C,H,W]」相反，compute_feat 是真把它铺出来 —— 峰值显存大头在此(可优化点)
        all_features = (one_hot_map.unsqueeze(1) * x.unsqueeze(2)).contiguous()

        # sum(dim=(3,4))   [2,768,17,37,37] -> [2,768,17]
        # 对每个(部件p,通道c)把区域内所有 patch 加起来(带软权)
        # permute(0,2,1)   [2,768,17]       -> [2,17,768]
        # sum_pool[b,p,:] = Σ_{(h,w)∈部件p区域} (maps[b,p,h,w]·x[b,:,h,w])   每部件的「加权特征和」(还没平均)
        # 把部件 p 区域内的所有 patch 特征叠成一个 768 维"原始堆叠特征"
        sum_pool = all_features.sum(dim=(3, 4)).permute(0, 2, 1)

        # sum(dim=(2,3))   [2,17,37,37] -> [2,17,1,1]
        # count_map[b,p] = Σ_{(h,w)∈部件p区域} (maps[b,p,h,w])   部件 p 的「软质量」= 区域内软置信之和
        # 部件 p 的「有效大小/权重总量」
        # 不是整数 patch 计数(是软值之和)
        # 平均每 patch 贡献≈0.24，17 个部件总和≈329
        # 这是第二个返回值，forward 里用 (count==0) 判“空部件” (count_mask)
        count_map = one_hot_map.sum(dim=(2, 3), keepdim=True)

        # count==0 处 +1，其余不变 -> 把为 0 的分母改成 1
        # 除零兜底：空部件 (没抢到任何 patch) 分母从 0 改为 1，让 0/1=0，而非 0/0=NaN
        count_map_ = count_map + (count_map == 0).float()

        # count_map_.squeeze(-1)   [2,17,1,1] -> [2,17,1]
        # [2,17,768] / [2,17,1] 广播 -> [2,17,768]
        # feat[b,p,:] = sum_pool / count = (Σ maps·x) / (Σ maps)   把「加权和」变「加权平均」
        # 部件 p 的描述向量 = 区域内 patch 特征的「置信加权平均」=「这张图里部件 p 长什么样」
        # 空部件 -> 0 向量
        all_features = sum_pool / count_map_.squeeze(-1)

        #              ( Σ_{(h,w): argmax_{p'} maps[b,p',h,w] = p}  maps[b,p,h,w] · x[b,:,h,w] )
        # feat[b,p] = ───────────────────────────────────────────────────────────────────────────
        #                          ( Σ_{(h,w): argmax = p}  maps[b,p,h,w] )
        # 
        #           = 部件 p「硬占据」的那些 patch 上，特征向量按软置信的加权平均

        # [2,17,768], [2,17,1,1]
        return all_features, count_map

    def forward(
        self, 
        x: Tensor
    ) -> tuple[Any, Any, Any, Any, int | Any] | tuple[Any, Any, Any, Any, int | Any]:
        # 以 B=2 为例

        # === 第 1 步：图变 token ===

        # x (img) = [2, 3, 518, 518] --- patch_embed (Conv2d 14×14,s14) ---> x = [2, 1369, 768]
        # 37 x 37 = 1369 个 patch token
        # 
        # [2, 3, 518, 518] -> [2, 1369, 768]
        # 图被切成 37×37=1369 个 patch，每个 patch 变成 768 维"局部描述"
        x = self.patch_embed(x)

        # x = [2, 1369, 768] --- _pos_embed (拼 cls1+reg4, 加 pos) ---> x = [2, 1374, 768]
        # 5 前缀 token + 1369 个 patch token = 1374 个 token
        # 
        # [2, 1369, 768] -> [2, 1374, 768]
        # 前面拼 5 个记账 token（1 cls + 4 register）、再给每个 token 加位置编码
        # 这 5 个是 DINOv2 的全局/寄存器槽，与"部件"无关，后面池化会被切掉
        x = self._pos_embed(x)

        # x_len = 1374
        # 
        # 记住"还没注入 prompt 时的序列长度"
        # 后面每个注入层靠它 x[:, :x_len] 把上一层的 prompt 槽切回去
        x_len = x.shape[1]

        # x = self.norm_pre(x)

        # === 第 2 步：前 8 层 纯主干，后 4 层 注入部件 ===

        # 前 8 块纯 DINOv2（i < 8）:
        # 先把通用视觉特征做扎实，部件机制完全不介入 —— 得先"看懂图"才谈得上"找部件"
        # 后 4 块（i = 8...11）每块一个循环，以 block 8（q_index=0，17 原型）为例

        x_buffer, q_buffer, m_buffer, qm_buffer = [], [], [], []
        l = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if i < l - self.layer_n:
                # block 0...7 (i < 12-4=8)
                # x = [2, 1374, 768] --- block ---> x = [2, 1374, 768]
                # x 形状全程不变 (纯 DINOv2 主干)
                x = block(x)
            # i>=8
            # q_index=0，该层 P+1 = n_pro[0] = 17 = 16前景 + 1背景
            # 9/10/11 层 同构，仅 P+1 = 14/11/8
            else:
                # 以 q_index=0 为例

                # 砍掉上一轮拼的 prompt，回到 [2, 1374, 768]
                # 
                # 丢掉上一层塞进来的 prompt 槽位
                # 每个注入层都从"干净的 patch token"重新找部件，prompt 只是上一块的一次性草稿纸
                # 草稿纸已经通过上一块 attention 影响过 patch 了，丢的只是槽、不是它的影响
                x = x[:, :x_len]

                q_index = i - l + self.layer_n
                # 该层原型(prompt 查询) [2, 17, 768]
                # 
                # [2, 17, 768]
                # 这一层的"部件模板库" —— 17 个可学习 768 维锚点（16 前景 + 1 背景）
                # 
                # 每个锚 = "某部件长什么样"
                # 注意它是参数、跨图共享，不随单张图变
                q = self.p_token[q_index].expand(x.shape[0], -1, -1) # + q_pre.detach()

                # x_buffer / q_buffer 各 append 一份(留给第 3 步读出)
                # 
                # 把（特征, 模板）存好，留给第 3 步统一读出（buffer 的用途之一）
                x_buffer.append(x)
                q_buffer.append(q)

                # q = q.detach() # TODO

                # x  = [2, 1374, 768]   --- self.norm -------> x_ = [2, 1374, 768]
                # x_ = [2, 1374, 768]   --- 砍掉前缀token ---> x_ = [2, 1369, 768]
                # x_ = [2, 1369, 768]   --- unflatten -------> x_ = [2, 37, 37, 768]
                # x_ = [2, 37, 37, 768] --- permute ---------> x_ = [2, 768, 37, 37]
                # 
                # 把 token 序列摆回“图像网格”，才能问"部件出现在画面哪个位置"
                # detach：找部件这条路不往主干回灌梯度（部件去适配特征，而非反向扰动主干）
                x_ = self.norm(x.detach()) 
                x_ = x_[:, self.num_prefix_tokens:, :]  # [B, L, D]
                x_ = self.unflatten(x_)  # [B, H, W, D]
                x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

                # 原型 <-> patch 的负欧氏距离
                # 得到 原型-patch 相似度图：[2, 17, 37, 37]
                # 
                # [2, 17, 37, 37]
                # 每个模板对 37×37 每个 patch 的相似度热图 = "这块 patch 多像部件 k"
                maps = self.compute_xq(x_, q) # [B, P+1, H, W]

                # 拿到对应的 可学习空间先验：[2, 17, 37, 37]
                p_bias = self.p_bias[q_index].expand(x.shape[0], -1, -1, -1)

                # 通道维归一: 每 patch 软分到 17 档  Σ_ch=1.000
                # 
                # p_bias = 可学习"空间先验"（如"部件 3 常在左上"），先加上，
                # 再沿 17 个通道 softmax -> 每个 patch 被软分给 17 个部件、Σ通道=1
                # ★这张就是"软分割图"，模型最核心的可解释产物
                maps = torch.nn.functional.softmax(maps + p_bias, dim=1) # 

                # 各原型区域均值特征
                # q_x：      [2, 17, 768]
                # count_map：[2, 17, 1, 1]
                # 
                # 掩码平均池化 —— 每个 patch 归到它 argmax 的那个部件，按置信度加权平均，
                # 得到"这张图里部件 k 的描述向量"
                # count_map = 每个部件抢到多少 patch（软质量）
                q_x, count_map = self.compute_feat(maps.detach(), x_) # [B, P+1, D] [B, P+1, 1, 1] # 

                # self.compute_xq() 回答 “部件在哪”
                # self.compute_feat() 回答 “部件长什么样”

                # 去背景，16 细原型 -> 4 粗部件打分
                # q_c：[2, 16, 4]
                # 
                # 细 -> 粗"路由打分"，
                # 模型用的细原型(16) 比最终汇报的部件(4) 多
                # 这个头决定每个细原型该并进哪个粗部件
                q_c = self.p_classifier[q_index](q_x[:, :-1]) # [B, P, N]

                count_mask = (count_map[:, :-1]==0).squeeze(-1).expand(-1, -1, q_c.shape[-1]) # [B, P, N]

                # 细->粗 软对齐 (每行和 1，空原型置1/4)
                # q_maps：[2, 16, 4]
                # 
                # 空原型 masked_fill 成 1/4
                # 把路由打分变成"每个细原型 -> 某个粗部件"的（近）硬分配
                # 没抢到 patch 的空原型给中性 1/4 —— "没出现就别站队"
                # 存 qm_buffer：记录这一层的细→粗归属（给评估/可视化用）
                if self.gumbel_softmax:
                    q_maps = torch.nn.functional.gumbel_softmax(
                        q_c, 
                        dim=-1, 
                        tau=self.gumbel_softmax_temperature, 
                        hard=self.gumbel_softmax_hard
                    )
                else:
                    q_maps = torch.nn.functional.softmax(q_c, dim=-1)
                q_maps = q_maps.masked_fill(count_mask, 1/q_c.shape[-1])
                qm_buffer.append(q_maps)

                # === 细 -> 粗 聚合，分两路、且 detach 不对称（关键设计）=== start

                # q_x：
                # [2,4,768]     特征路：detach -> 路由比例当"固定路由"
                # 
                # f_maps：
                # [2,5,1369]  空间图路：不 detach -> 梯度能塑造路由
                # 
                # m_buffer.append(f_maps)：
                # 4 个粗部件(+背景) 的空间图，留给 consistency loss
                # 
                # 两路 detach 不一样是有意的：
                # prompt 注入要稳(detach)，而 m_buffer 要喂 consistency 去学路由(不 detach)

                q_maps_expanded = q_maps.unsqueeze(-1)  # (B, P, N, 1)

                # f_maps：[2,17,1369]
                f_maps = maps.flatten(2) # (B, P+1, L)

                q_maps_detach = q_maps_expanded.detach()
                q_x_weighted = q_x[:, :-1].unsqueeze(2) * q_maps_detach  # (B, P, 1, D) × (B, P, N, 1) = (B, P, N, D)
                q_x = q_x_weighted.sum(dim=1)  # (B, N, D)
                q_maps_sum = q_maps_detach.sum(1).squeeze(1)

                q_x = q_x / q_maps_sum

                f_bg = f_maps[:, -1:] # (B, 1, L)
                f_maps_weighted = f_maps[:, :-1].unsqueeze(2) * q_maps_expanded  # (B, P, 1, L) × (B, P, N, 1) = (B, P, N, L)

                f_maps = f_maps_weighted.sum(dim=1)  # (B, N, L)
                f_maps = torch.cat([f_maps, f_bg], dim=1) # (B, N+1, L)
                
                m_buffer.append(f_maps)

                # === 细 -> 粗 聚合，分两路、且 detach 不对称（关键设计）=== end

                # 把 4 个粗部件特征 归一化 + 投影，打包成 4 个 prompt token
                q_x = self.p_linear[q_index](self.p_norm[q_index](q_x))

                # 把部件当 prompt 塞回序列再过这一块：
                # patch 与部件互相 attention，主干和部件“共同精炼”（这正是 IVPT 的 visual-prompt 耦合）
                # 下一层再丢、再来一遍，原型数 17->14->11->8 逐层变粗
                x = torch.cat([x, q_x], dim=1)
                x = block(x)

        # 补上第 5 个、最粗(5 个原型)的读出层级，凑齐 5 个(x, q)
        q = self.p_token[-1].expand(x.shape[0], -1, -1)
        x = x[:, :x_len]
        x_buffer.append(x)
        q_buffer.append(q)

        maps_list = []
        # 对每个层级“干净地重算”一遍软分割图（不带这次的 detach 噪声）
        for i, (x, q) in enumerate(zip(x_buffer, q_buffer)):
            x_ = self.norm(x.detach())
            x_ = x_[:, self.num_prefix_tokens:, :]  # [B, num_patch_tokens, embed_dim]
            x_ = self.unflatten(x_)  # [B, H, W, embed_dim]
            x_ = x_.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, H, W]
            
            maps = self.compute_xq(x_, q)
            
            p_bias = self.p_bias[i].expand(x.shape[0], -1, -1, -1)
            maps = torch.nn.functional.softmax(maps+p_bias, dim=1)  # [B, num_landmarks + 1, H, W] # 
            # 不让某通道恰好为 0
            maps = maps + 1e-6
            maps = maps / maps.sum(dim=1, keepdim=True)

            maps_list.append(maps)

        # maps_list：5张图，通道 17/14/11/8/5，即对外的 maps_loss / map_feat
        # ★几乎所有"部件发现"正则都作用在这张 list 上

        # 最终最粗的 5 通道图（4 部件 + 背景）
        maps = maps_list[-1]
        f_maps = maps.flatten(2)
        m_buffer.append(f_maps)

        # x_ = self.norm(x.detach())
        # x_ = x_[:, self.num_prefix_tokens:, :]
        # x_ = self.unflatten(x_)
        # x_ = x_.permute(0, 3, 1, 2).contiguous()
        x = self.unflatten(self.norm(x_buffer[-1])[:, self.num_prefix_tokens:, :]).permute(0, 3, 1, 2).contiguous()

        # [2,768,5]
        # 用最终图把部件特征池化出来
        all_features = self.compute_feat(maps, x)[0]
        all_features = all_features.permute(0, 2, 1)

        # Modulate the features
        # LayerNorm[768,5]：分类前给每个(特征维，部件)重标定尺度
        all_features_mod = self.modulation(all_features)  # [B, embed_dim, num_landmarks + 1]

        # Classification based on the landmark features
        # [2,200,4]
        # 每个部件各自预测 200 个类
        # ★4 个部件各投一票、平均 = 最终类别分
        # 可解释性的落点：能拆开看每个部件分别投了什么
        scores = self.fc_class_landmarks(
            all_features_mod[..., :-1].permute(0, 2, 1).contiguous()
        ).permute(0, 2, 1).contiguous()

        map_feat = maps_list

        return all_features_mod, maps_list[-3], scores, map_feat, (m_buffer, qm_buffer)

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