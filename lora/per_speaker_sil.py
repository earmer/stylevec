"""每个说话人的 silhouette score（相对于所有其他人）。"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_samples

EMB_CACHE = Path(__file__).resolve().parent / "analysis_embeddings.pkl"

with open(EMB_CACHE, "rb") as f:
    emb = pickle.load(f)

names = list(emb.keys())
vecs_list, labels_list = [], []
for i, name in enumerate(names):
    vecs_list.append(emb[name])
    labels_list.extend([i] * len(emb[name]))

X = np.concatenate(vecs_list)
labels = np.array(labels_list)

print(f"总样本: {X.shape[0]}, 说话人: {len(names)}")
print("计算 per-sample silhouette (cosine)...")

sample_sil = silhouette_samples(X, labels, metric="cosine", n_jobs=-1)

# 按说话人聚合
results = []
for i, name in enumerate(names):
    mask = labels == i
    s = sample_sil[mask]
    results.append((name, len(s), float(s.mean()), float(s.std()), float(s.min()), float(s.max())))

results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'说话人':<12} {'N':>4} {'mean_sil':>10} {'std':>8} {'min':>8} {'max':>8}")
print("-" * 56)
for name, n, mean, std, mn, mx in results:
    print(f"{name:<12} {n:>4} {mean:>10.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")

pos = sum(1 for r in results if r[2] > 0)
print(f"\nsilhouette > 0 的说话人: {pos}/{len(results)}")
print(f"全局 silhouette: {sample_sil.mean():.4f}")
