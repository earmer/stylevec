"""评估指标：Silhouette Score + Consistency。"""

import numpy as np
from sklearn.metrics import silhouette_score


def consistency(vecs, labels, n_classes):
    """计算每类内部余弦一致性的均值。"""
    normed = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-10)
    scores = []
    for i in range(n_classes):
        v = normed[labels == i]
        if len(v) < 2:
            scores.append(float("nan"))
            continue
        cos = v @ v.T
        scores.append(float(cos[np.triu_indices(len(v), k=1)].mean()))
    return float(np.nanmean(scores))


def silhouette(vecs, labels):
    """Cosine silhouette score。"""
    n_unique = len(set(labels.tolist()))
    if n_unique < 2:
        return float("nan")
    return float(silhouette_score(vecs, labels, metric="cosine", n_jobs=-1))


def evaluate_all(vecs_train, labels_train, n_train,
                 vecs_val, labels_val, n_val,
                 vecs_gen, labels_gen, n_gen):
    """对 train/val/gen 三个集合计算指标，返回 dict。"""
    return {
        "train_sil": silhouette(vecs_train, labels_train),
        "train_cons": consistency(vecs_train, labels_train, n_train),
        "val_sil": silhouette(vecs_val, labels_val),
        "val_cons": consistency(vecs_val, labels_val, n_val),
        "gen_sil": silhouette(vecs_gen, labels_gen),
        "gen_cons": consistency(vecs_gen, labels_gen, n_gen),
    }
