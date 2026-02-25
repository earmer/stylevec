The current project aims to create an style (not content) embedding for short-length texts.

## Local Models

### Qwen 3 0.6B
- **Location**: `/base-models/qwen-3-0.6b/`
- **Source**: Hugging Face (Qwen/Qwen3-0.6B)
- **Status**: Downloaded and verified locally
- **Usage**:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/base-models/qwen-3-0.6b"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
```

## 结论：残差风格向量方案不可行

Embedding 模型对风格的清洗非常彻底，不能将其当作黑盒、试图从输出向量中分离出 style 分量。

### 实验方案

对原神角色对话数据（10个说话人，每人100句），用残差法提取风格向量：

`style = embed(原文) - embed(标准改写)` ，可选 PCA 正交投影去除内容泄漏。

### 实验数据

| 运行 | 平均余弦一致性 | 轮廓系数 |
|------|--------------|---------|
| qwen3-embedding:0.6b / 纯残差 | 0.0183 | -0.0085 |
| qwen3-embedding:0.6b / PCA投影 | 0.0043 | -0.0055 |
| embeddinggemma:latest / 纯残差 | 0.0089 | -0.0085 |
| embeddinggemma:latest / PCA投影 | 0.0024 | -0.0067 |

一致性接近 0（随机基线），轮廓系数为负（无可分性）。PCA 投影后指标反而更差，说明残差中本就没有可提取的风格信号。

## Instruction-aware Embedding 实验

### 发现：Ollama/OpenRouter 的 prompt 字段无效

`/api/embed` 的 `prompt` 参数被静默忽略，需要将 instruction 拼接到输入文本中：

```
Instruct: {指令}\n{文本}
```

qwen3-embedding 对 instruction 敏感（拼接后 vs 纯文本余弦 ~0.58-0.63），8b 比 0.6b 偏移更大。

### Prompt 残差实验（0.6b，10人×100句）

| Prompt | Mode | 平均一致性 | 轮廓系数 |
|--------|------|-----------|---------|
| baseline | raw | 0.0171 | -0.0087 |
| style_v2 "分析语言风格和说话习惯" | raw | 0.0341 | -0.0147 |
| style_v4 "判断说话人是谁" | raw | 0.0247 | -0.0090 |
| cluster "按说话风格聚类" | raw | 0.0219 | -0.0105 |

PCA 50% 投影后所有 prompt 指标趋同（一致性 0.008~0.015），prompt 差异被抹平。

### Prompt 残差实验（8b，6人×25句）

| Prompt | Mode | 平均一致性 | 轮廓系数 |
|--------|------|-----------|---------|
| baseline | direct | 0.4506 | -0.0368 |
| style_v2 | direct | 0.7675 | -0.0372 |
| style_v2 | raw_residual | 0.0826 | -0.0253 |
| style_v4 | direct | 0.6946 | -0.0427 |

关键发现：8b + style prompt 下原文 embedding 的同说话人一致性高达 0.77，但轮廓为负——不同说话人方向重叠。残差法反而破坏信号（0.77→0.08）。

## LDA 监督降维实验

### 小规模（6人训练 → 5人泛化，8b style_v2）

去均值+L2归一化无效（轮廓仍为负）。LDA 在训练集上有效（轮廓 0.35），但泛化到未见说话人后崩溃（轮廓 -0.03）。

### 大规模（80人训练 → 20人泛化，8b style_v2，每人25句）

使用 eigen solver + shrinkage，尝试多种维度：

| Dim | Train Sil | Train Con | Test Sil | Test Con |
|-----|-----------|-----------|----------|----------|
| 64 | 0.0190 | 0.1811 | -0.0150 | 0.0955 |
| 32 | 0.0027 | 0.2611 | -0.0324 | 0.1237 |
| 16 | -0.0745 | 0.3205 | -0.0681 | 0.1494 |

训练集自身轮廓就很低，80 类在低维空间过于拥挤。LDA 学到的是特定说话人的判别方向，不是通用风格空间。

## MLP + ArcFace 实验

### 小规模（80人训练 → 20人泛化，每人25句）

Triplet loss 版本因随机采样导致大量 easy triplet，模型几乎未学习，各架构结果趋同。

改用 ArcFace (s=30, m=0.5) 后，宽模型（1h-2k）出现模式坍塌（一致性 0.99+，轮廓 -0.1~-0.2），所有输入映射到相似区域。线性模型相对最优但轮廓仍为负。

### 大规模（141人训练 → 36人泛化，分层采样）

数据：177人（≥200句），分层采样（≥10000句采1000，≥1000句采150，其余采50），共12050条。训练集按 80/20 切分为 train/val。

| Arch | Dim | Tr Sil | Tr Con | Va Sil | Va Con | Te Sil | Te Con |
|------|-----|--------|--------|--------|--------|--------|--------|
| linear | 32 | -0.1983 | 0.9807 | -0.2773 | 0.9807 | -0.1459 | 0.9813 |
| linear | 64 | -0.1219 | 0.9649 | -0.1984 | 0.9647 | -0.0927 | 0.9662 |
| linear | 128 | -0.0750 | 0.9840 | -0.1602 | 0.9837 | -0.0714 | 0.9842 |
| 1h-512 | 128 | -0.1097 | 0.9952 | -0.2457 | 0.9949 | -0.1310 | 0.9949 |
| 2h-1k | 128 | -0.3141 | 0.9760 | -0.5926 | 0.9073 | -0.6854 | 0.8870 |
| 1h-2k | 64 | -0.2246 | 0.9940 | -0.3322 | 0.9939 | -0.1898 | 0.9939 |

全面模式坍塌：一致性 0.98~0.99 横跨 train/val/test，但轮廓全为负。深层模型（2h-1k）还叠加过拟合（val/test 轮廓远差于 train）。ArcFace 的 margin 在当前数据量下过于激进，loss 未收敛（141类理论下限 ~0，实际最低 ~8）。

## 总结

| 方法 | 结果 |
|------|------|
| 残差法 (embed(原文) - embed(改写)) | 无风格信号 |
| Instruction prompt 引导 | 提升同类一致性，但不同说话人方向重叠 |
| 去均值 + L2 归一化 | 无改善 |
| LDA 监督降维 | 训练集可分，不泛化 |
| MLP + Triplet Loss | 随机采样导致训练无效 |
| MLP + ArcFace | 模式坍塌，轮廓全负 |

核心问题：qwen3-embedding-8b 的 4096 维空间中，不同说话人的风格方向高度重叠。无监督方法提取不出信号，监督方法要么过拟合要么坍塌。Embedding 模型在预训练阶段已将风格信息清洗，后处理投影无法恢复。

## Causal LM 隐藏层探针实验

### 假设

Embedding 模型的对比学习训练清洗了风格信息，但 Causal LM（Qwen3-0.6B）的中间隐藏层未经此清洗，可能在某些层保留了可提取的风格信号。

### 实验设置

- 模型：Qwen3-0.6B CausalLM，28 transformer 层，hidden_size=1024，`attn_implementation="eager"`
- 数据：原神对话数据库，top-12 说话人（排除？？？），筛选条件 `LENGTH(origin_text) > 4 AND para_text IS NOT NULL`
  - 训练组 8 人（派蒙、娜维娅、纳西妲、温迪、阿贝多、茜特菈莉、芙宁娜、赛诺）：每人 80 训练 + 20 验证，共 train=640 / val=160
  - 泛化组 4 人（旅行者、八重神子、玛拉妮、玛薇卡）：每人 100 句，共 gen=400
- 提取：29 层（embedding + 28 transformer）× 4 种池化策略
- 探针：LDA（n_components=7）、MLP+ArcFace（linear/1h-256 × d32/d64，300 epochs）
- 指标：Cosine Silhouette Score、Intra-class Consistency

### 核心发现

**1. 假设不成立：所有 val_sil 均为负值**

全部 29 层 × 4 池化 × 5 探针组合（共 580 个结果）的 val_sil 无一突破零线。最佳结果：

| Pool | Layer | Method | val_sil | val_cons | gen_sil |
|------|-------|--------|---------|----------|---------|
| rev_inv | 1 | MLP-linear-d64 | -0.0705 | +0.0604 | -0.0224 |
| rev_inv | 1 | MLP-1h-256-d64 | -0.0706 | +0.0974 | -0.0147 |
| last | 9 | MLP-linear-d64 | -0.0713 | +0.0564 | -0.0217 |
| last | 25 | MLP-1h-256-d64 | -0.0743 | +0.2711 | -0.0271 |

负的 silhouette 意味着同说话人句子之间的距离大于不同说话人之间的距离——不是"信号弱"，而是探针方法无法从静态向量中提取可分的风格表示。

**2. 极端过拟合：train_sil 与 val_sil 的断崖**

MLP+ArcFace 在训练集上 train_sil 普遍达到 +0.7~+0.87，而对应 val_sil 跌至 -0.07~-0.12。模型记住了句子身份，而非说话人风格。代表性数据（last pool, MLP-1h-256-d64）：

| Layer | train_sil | val_sil | gap |
|-------|-----------|---------|-----|
| L3 | +0.8650 | -0.0991 | 0.964 |
| L19 | +0.8270 | -0.0901 | 0.917 |
| L20 | +0.8565 | -0.0913 | 0.948 |

**3. Attention 池化病理性坍塌**

正向 attention 加权池化（attn pool）从 L3 起，MLP 变体的 consistency 全部锁死在 1.0000，说明所有向量被压缩到同一方向。attn pool 在 Top 20 中零席位，是四种策略中唯一完全失败的。

**4. 层间差异是噪声，不是结构**

固定 rev_inv + MLP-1h-256-d64，L1-L28 的 val_sil 分布：

| 区间 | 均值 | 范围 |
|------|------|------|
| L01 | -0.071 | — |
| L02-L10 | -0.086 | -0.079 ~ -0.106 |
| L11-L20 | -0.082 | -0.075 ~ -0.088 |
| L21-L28 | -0.081 | -0.075 ~ -0.087 |

L1 略优，L2-L28 在 0.011 的窄带内随机波动，不存在"浅层好/深层好/中间差"的系统性趋势。

### 技术陷阱记录

- `output_attentions=True` 传入 `from_pretrained()` 会被存入 GenerationConfig 而静默失效，需传入 `model.forward()` 调用
- Qwen3 默认使用 SDPA，`outputs.attentions` 返回空 tuple，必须指定 `attn_implementation="eager"` 才能获得注意力权重

### 理论解释

风格信息确实存在于模型中（模型可完成风格续写和风格转换任务），但其编码形式与静态向量提取存在根本性范式错配：

- **假设 C（主）**：风格编码在生成过程的动力学中——每步 token 生成时，上文风格线索通过 28 层完整计算图动态影响下一 token 的条件分布。风格是一个函数，而非一个点。
- **假设 B（辅）**：风格跨层分布式编码，单层快照只含碎片。
- **假设 A**（C 的退化特例）：风格是非线性可分的静态向量，更深探针可解开——但 train/val gap 表明更深探针只会加剧过拟合。

逐层提取静态向量 + 浅层探针的方法论，在假设 C 成立的前提下天花板极低。

### 后续方向

| 方向 | 思路 |
|------|------|
| 生成式风格度量 | 用模型自身困惑度差异衡量风格距离，绕过向量提取 |
| 对比微调 | LoRA/adapter 强制将风格压入可分空间，而非探测预训练表示 |
| 多层联合表示 | 拼接多层 hidden states，而非逐层独立探测 |
| 更大模型 | 4B/8B 参数量下风格表示是否更集中，待验证 |

