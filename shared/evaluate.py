"""评估指标：Silhouette Score + Consistency。"""

import numpy as np
from sklearn.metrics import silhouette_score
from typing import Optional, Union
from .data_types import EvalData, EvalMetrics


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


def evaluate_all(train: Union[EvalData, np.ndarray],
                 val: Union[EvalData, np.ndarray],
                 gen: Optional[Union[EvalData, np.ndarray]] = None,
                 # 向后兼容参数
                 labels_train=None, n_train=None,
                 labels_val=None, n_val=None,
                 labels_gen=None, n_gen=None) -> Union[EvalMetrics, dict]:
    """
    对 train/val/gen 三个集合计算指标。

    支持两种调用方式：
    1. 新方式（推荐）：evaluate_all(train_eval, val_eval, gen_eval)
    2. 旧方式（向后兼容）：evaluate_all(vecs_train, labels_train, n_train, ...)
    """
    # 检测是否使用新方式（EvalData 对象）
    if isinstance(train, EvalData):
        train_sil = silhouette(train.vecs, train.labels)
        train_cons = consistency(train.vecs, train.labels, train.n_classes)

        val_sil = silhouette(val.vecs, val.labels)
        val_cons = consistency(val.vecs, val.labels, val.n_classes)

        if gen is not None:
            gen_sil = silhouette(gen.vecs, gen.labels)
            gen_cons = consistency(gen.vecs, gen.labels, gen.n_classes)
        else:
            gen_sil = float('nan')
            gen_cons = float('nan')

        return EvalMetrics(
            train_sil=train_sil,
            train_cons=train_cons,
            val_sil=val_sil,
            val_cons=val_cons,
            gen_sil=gen_sil,
            gen_cons=gen_cons,
        )

    # 向后兼容：旧方式调用
    else:
        vecs_train = train
        vecs_val = val
        vecs_gen = gen

        return {
            "train_sil": silhouette(vecs_train, labels_train),
            "train_cons": consistency(vecs_train, labels_train, n_train),
            "val_sil": silhouette(vecs_val, labels_val),
            "val_cons": consistency(vecs_val, labels_val, n_val),
            "gen_sil": silhouette(vecs_gen, labels_gen),
            "gen_cons": consistency(vecs_gen, labels_gen, n_gen),
        }

