
### 一、哪些重点看、哪些可不看

`models/` 全貌（`*` = 重点看，`~` = 死代码/可跳过）：

```
models/
├── builder.py                      *  构建路径（构建接口，决定用什么参数造模型）
├── individual_landmark_vit.py      *  ★模型本体：class IndividualLandmarkViT(nn.Module)
├── individual_landmark_vit_deit.py ~  重复文件，全仓库无 import
├── layers/
│   ├── transformer_layers.py       *  两个改过的 timm 层：Block/Attention WQKV（小，全读）
│   └── independent_mlp.py          *  逐部件 MLP 分类头（仅 classifier_type=independent_mlp 时用）
└── __init__.py                        只是 re-export，扫一眼即可
```

**已 grep 证实是死代码 / 别读**：
- `individual_landmark_vit_deit.py`：除自身外**全仓库无 import**。它和主文件几乎逐字相同，只相差 2 行，是从 PDiscoFormer 抄来的 deit 分支，当前训练/评估路径根本不走。
- `builder.py` 底部的 `ivpt_vit / ivptnet_vit / ivptnet_resnet101` 这几个 torch.hub 入口：**本仓库没有任何地方调它们**，而且它们调的 `ivpt_vit_bb(...)` 会传 `num_landmarks=`、`modulation_orth=` 两个 `__init__` **根本不存在的参数** → 一调就 `TypeError`；`ivptnet_resnet101` 还调了一个**全仓库未定义**的 `ivptnet_resnet_torchvision_bb` → `NameError`。结论：这些是 PDiscoFormer 的遗留 hub 接口，**调不通**。

---

### 二、构建路径（从这里开始看）——`models/builder.py`

造模型链路（train_net.py / eval 脚本都走这条）：

```
train_net.py:  model = load_model_ivpt(args, num_cls)
        └─> builder.py:  load_model_ivpt
                ├─ load_model_arch(args, num_cls)   # ①造 timm 骨架(返回纯 ViT，还没套部件机制)
                └─ init_ivpt_model(base_model, …)    # ②把骨架塞进 IndividualLandmarkViT
                        └─ IndividualLandmarkViT(base_model, …, n_pro=args.n_pro)
```

- `load_model_arch`：按 `args.model_arch` 字符串分流建骨架——含 `"patch"` → 走 ViT 分支 `create_model(...)`（本项目用 `vit_base_patch14_reg4_dinov2.lvd142m`）；另有 resnet/convnext 分支（用不到，可跳）。
- `init_ivpt_model`：**唯一真正 new 出模型的地方**，只有 `'patch'`(ViT) 分支被实现，其余 `raise`。它把 `args` 里这串参数喂给模型：`part_dropout / modulation_type / gumbel_softmax(+temperature/hard) / classifier_type / noise_variance / n_pro`。
- 看 builder  = 搞清「**哪些 args 决定模型形状**」，其中最关键的是 `n_pro`（见下）。底部 hub 函数可不看。

### 三、★模型定义——`models/individual_landmark_vit.py`

其中， `__init__` 是「模型定义」的核心，**先读 `__init__`，把有哪些子模块/参数搞清楚**。

先搞清楚一个参数 `n_pro`，整座模型的形状都由它推出来（例：脚本 `--n_pro 17,14,11,8,5`）：

```python

# 逗号串解析成 list：5 个层级、每层的原型数
self.n_pro           = [17,14,11,8,5]
# → 在「最后 4 个 transformer 块」插入部件机制
self.layer_n         = len(n_pro) - 1   = 4
# 最末(最粗)层 5 个原型 = 4 个前景部件 + 1 个背景
self.num_landmarks   = n_pro[-1]  - 1   = 4
# 例：17 = 16 前景原型 + 1 背景
# 这串数字 17→14→11→8→5 就是论文说的「fine → coarse 层级」

```

`__init__` 里的「可解释部件」家族（按层级 index 存成 ParameterList / ModuleList）：

```python

# 学习的「原型/prompt 查询」token，每层一组(17/14/11/8/5 个)
self.p_token        = ParameterList(5 个 [1, n_pro[i], D])
# 加到「原型-patch 相似度图」上的可学习空间先验(H=W=37)
self.p_bias         = ParameterList(5 个 [1, n_pro[i], H, W])
# 把聚合到的 原型特征 投影成 prompt token
self.p_linear       = ModuleList(4 个 Linear(D, D))
# prompt 前的归一化(4=num_landmarks)
self.p_norm         = ModuleList(4 个 LayerNorm([4, D]))
# ★跨层对齐头：把「细原型」分配到 4 个「粗部件」桶 (fine→coarse)
self.p_classifier   = ModuleList(4 个 Linear(D, 4))

```
