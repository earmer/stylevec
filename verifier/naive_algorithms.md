# Naive 判别算法文档

本文档记录 `naive/` 实验中使用的所有判别与评估算法。

---

## 评估指标

### 1. Silhouette Score (余弦距离)

衡量全局聚类分离度。对每个样本计算：

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

- `a(i)`：样本 i 与同类其他样本的平均余弦距离
- `b(i)`：样本 i 与最近异类簇所有样本的平均余弦距离

取所有样本的均值，范围 [-1, 1]。正值表示类间分离 > 类内距离，负值表示聚类结构差。

**实际调用**：
```python
silhouette_score(vecs, labels, metric='cosine')
```

当样本量过大时（>10000），随机采样后计算。

### 2. Consistency (类内余弦一致性)

衡量同一角色内部向量的聚拢程度。对某个角色的所有向量：

```python
normed = vecs / ||vecs||           # L2 归一化
cos_matrix = normed @ normed.T     # 两两余弦相似度
consistency = mean(上三角元素)       # 排除对角线的自相似
```

范围 [-1, 1]，值越高表示同一角色的向量方向越一致。当样本数 > 100 时随机采样 100 条计算。

最终取所有角色 consistency 的均值（`np.nanmean`），跳过样本数 < 2 的角色。

