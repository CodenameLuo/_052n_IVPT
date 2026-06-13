"""Modified ``timm`` Attention / Block that also return QKV tensors.

Used by :class:`~models.individual_landmark_vit.IndividualLandmarkViT` to
extract intermediate attention information for the part-prototype mechanism.
"""

# ======================================
#
# 这里把 timm 原版的 Attention / Block 各派生一个“顺带返回 q,k,v 三件套”的版本。
# IndividualLandmarkViT.__init__ 里的 convert_blocks_and_attention() 会把主干每个 block/attention
# 的类“偷偷换成”这两个版本(直接改 __class__，即猴补丁)。
#
# 关于本次训练实际怎么用它们：
#   - 训练 forward 里调用 block 时只传一个 x(x_k=x_v=None)，所以是普通自注意力；
#   - 而且 return_qkv 默认 False，所以“返回 q,k,v”这个能力在正常训练里其实没被用到——
#     它是给分析/可视化的 get_intermediate_layers 那套方法(return_transformer_qkv=True)准备的。
#   换句话说：本次训练真正图的只是“行为和 timm 原版一致”，qkv 返回功能是顺带保留的。
#
# ======================================

from typing import Tuple

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block


class AttentionWQKVReturn(Attention):
    """
    Modifications:
         - Return the qkv tensors from the attention
    """

    # x_q/x_k/x_v 分开传是为了支持交叉注意力；本次训练 x_k=x_v=None，会回落成自注意力(三者都用 x_q)
    def forward(self, x_q, x_k=None, x_v=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # k、v 没给就用 q(自注意力)
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        B, NQ, C = x_q.shape   # B=batch, NQ=query token 数, C=通道维
        B, NK, C = x_k.shape   # NK=key token 数
        # 对 x_q 做 qkv 投影并拆成 [3, B, head, N, head_dim]，只取出 q(第 0 份)
        qkv = self.qkv(x_q).reshape(B, NQ, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv[0]

        # 对 x_k 同样投影，只取 k(第 1 份)
        qkv = self.qkv(x_k).reshape(B, NK, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = qkv[1]

        # 对 x_v 同样投影，只取 v(第 2 份)
        # 注：自注意力时这等于把同一个 qkv 投影算了 3 遍，略浪费，但换来对交叉注意力的统一支持
        qkv = self.qkv(x_v).reshape(B, NK, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = qkv[2]

        # 对 q、k 做归一化(timm 的 q_norm/k_norm，无则是恒等)
        q, k = self.q_norm(q), self.k_norm(k)

        # 注意力计算：优先用 PyTorch 融合算子(更快更省显存)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        # 否则手算：q·kᵀ 缩放 -> softmax -> 加权求和 v
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        # 把多头拼回 [B, NQ, C]，再过输出投影 proj
        x = x.transpose(1, 2).reshape(B, NQ, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # torch.stack((q, k, v), dim=0)
        # 返回注意力输出 + (q,k,v) 三件套
        return x, (q, k, v)

class BlockWQKVReturn(Block):
    """
    Modifications:
        - Use AttentionWQKVReturn instead of Attention
        - Return the qkv tensors from the attention
    """

    # 一个标准 Transformer block：先注意力子层(带残差)，再 MLP 子层(带残差)；只是 attn 换成了上面带 qkv 返回的版本
    def forward(self, x_q, x_k=None, x_v=None, return_qkv: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        # k、v 没给就用 q(自注意力)
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_q
        # 注意力子层：先 LayerNorm(norm1) 再进注意力，拿到输出和 qkv
        x_attn, qkv = self.attn(self.norm1(x_q), self.norm1(x_k), self.norm1(x_v))
        # 残差 + (可选)LayerScale ls1 + DropPath
        x = x_q + self.drop_path1(self.ls1(x_attn))
        # MLP 子层：LayerNorm(norm2) -> mlp -> LayerScale ls2 -> DropPath，再残差
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        # 默认只返回特征 x；只有显式 return_qkv=True(分析用)时才把 qkv 一起返回
        if return_qkv:
            return x, qkv
        else:
            return x
