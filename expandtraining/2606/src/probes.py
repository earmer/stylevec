from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def vector_probe_metrics(embeddings: Any, prefix_dims: Iterable[int] = ()) -> dict[str, float]:
    matrix = _as_matrix(embeddings)
    metrics: dict[str, float] = {}
    metrics.update(_matrix_probe_metrics(matrix, "vector"))
    dim = matrix.shape[1]
    for prefix_dim in sorted({int(value) for value in prefix_dims if int(value) > 0}):
        if prefix_dim >= dim:
            continue
        prefix = _l2_normalize(matrix[:, :prefix_dim])
        metrics.update(_matrix_probe_metrics(prefix, f"prefix_{prefix_dim}"))
    return metrics


def length_leakage_metrics(
    embeddings: Any,
    lengths: Iterable[int | float],
    buckets: Iterable[Any],
    max_pair_items: int = 256,
    groups: Iterable[Any] | None = None,
    speakers: Iterable[Any] | None = None,
) -> dict[str, float]:
    """How much text length the embedding encodes.

    `length_linear_r2` pools every row, so it also counts length differences between domains, which the
    embedding can read off provenance. With `groups` (e.g. rule:role), `length_r2_within_group` is the
    same R2 after removing each group's mean length and mean embedding. With `speakers`,
    `length_r2_within_speaker` does that per speaker, over rows that have one: utterance-level length with
    each speaker's own verbosity (a style trait) taken out.
    """
    matrix = _as_matrix(embeddings)
    length_values = np.asarray(list(lengths), dtype=np.float64)
    if matrix.shape[0] != length_values.shape[0]:
        raise ValueError("embeddings and lengths must have the same row count")
    bucket_values = np.asarray(list(buckets), dtype=object)
    if bucket_values.shape[0] != length_values.shape[0]:
        raise ValueError("lengths and buckets must have the same row count")

    metrics = {
        "length_count": float(length_values.shape[0]),
        "length_std": float(length_values.std()) if length_values.size else 0.0,
        "length_linear_r2": 0.0,
        "length_pc1_abs_corr": 0.0,
        "length_cosine_delta_corr": 0.0,
        "length_same_bucket_cos_mean": 0.0,
        "length_diff_bucket_cos_mean": 0.0,
        "length_same_minus_diff_cos": 0.0,
    }
    if groups is not None:
        metrics["length_r2_within_group"] = 0.0
    if speakers is not None:
        metrics["length_r2_within_speaker"] = 0.0
    if matrix.shape[0] < 4 or metrics["length_std"] <= 0.0:
        return metrics

    log_lengths = np.log1p(length_values)
    metrics["length_linear_r2"] = _ridge_regression_r2(matrix, log_lengths)
    if groups is not None:
        metrics["length_r2_within_group"] = _within_r2(matrix, log_lengths, np.asarray(list(groups), dtype=object))
    if speakers is not None:
        metrics["length_r2_within_speaker"] = _within_r2(matrix, log_lengths, np.asarray(list(speakers), dtype=object))
    metrics["length_pc1_abs_corr"] = abs(_pc1_corr(matrix, log_lengths))

    subset = _even_subset(matrix.shape[0], max_pair_items)
    sub_matrix = _l2_normalize(matrix[subset])
    sub_lengths = log_lengths[subset]
    sub_buckets = bucket_values[subset]
    cosine = sub_matrix @ sub_matrix.T
    length_delta = np.abs(sub_lengths[:, None] - sub_lengths[None, :])
    upper = np.triu(np.ones_like(cosine, dtype=bool), k=1)
    if upper.any():
        metrics["length_cosine_delta_corr"] = _pearson(cosine[upper], length_delta[upper])
        same_bucket = (sub_buckets[:, None] == sub_buckets[None, :]) & upper
        diff_bucket = (sub_buckets[:, None] != sub_buckets[None, :]) & upper
        if same_bucket.any():
            metrics["length_same_bucket_cos_mean"] = float(cosine[same_bucket].mean())
        if diff_bucket.any():
            metrics["length_diff_bucket_cos_mean"] = float(cosine[diff_bucket].mean())
        metrics["length_same_minus_diff_cos"] = (
            metrics["length_same_bucket_cos_mean"] - metrics["length_diff_bucket_cos_mean"]
        )
    return metrics


def _matrix_probe_metrics(matrix: np.ndarray, prefix: str) -> dict[str, float]:
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    dim_std = centered.std(axis=0)
    metrics = {
        f"{prefix}_norm_mean": float(norms.mean()) if norms.size else 0.0,
        f"{prefix}_norm_std": float(norms.std()) if norms.size else 0.0,
        f"{prefix}_dim_std_mean": float(dim_std.mean()) if dim_std.size else 0.0,
        f"{prefix}_top_component_ratio": 0.0,
        f"{prefix}_effective_rank": 0.0,
    }
    if matrix.shape[0] < 3:
        return metrics
    covariance = centered.T @ centered / float(max(1, matrix.shape[0] - 1))
    eigenvalues = np.linalg.eigvalsh(covariance).clip(min=0.0)
    total = float(eigenvalues.sum())
    if total <= 1e-12:
        return metrics
    probabilities = eigenvalues / total
    entropy = -float((probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum())
    metrics[f"{prefix}_top_component_ratio"] = float(eigenvalues.max() / total)
    metrics[f"{prefix}_effective_rank"] = float(np.exp(entropy))
    return metrics


def _within_r2(matrix: np.ndarray, target: np.ndarray, labels: np.ndarray) -> float:
    """R2 of length after removing each label's mean from both length and embedding (fixed effects): one
    pooled fit, so small groups do not overfit, and a label's identity cannot stand in for its typical
    length. Rows with an empty label, or alone in their label, are left out."""
    names = [str(value) if value is not None else "" for value in labels.tolist()]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    mask = np.array([bool(name) and counts[name] >= 2 for name in names])
    if mask.sum() < 8:
        return 0.0
    kept = np.asarray(names, dtype=object)[mask]
    x, y = matrix[mask].astype(np.float64), target[mask].astype(np.float64)
    x_within, y_within = x.copy(), y.copy()
    for name in set(kept.tolist()):
        rows = kept == name
        x_within[rows] -= x[rows].mean(axis=0, keepdims=True)
        y_within[rows] -= y[rows].mean()
    return _ridge_regression_r2(x_within, y_within)


def _ridge_regression_r2(matrix: np.ndarray, target: np.ndarray, ridge: float = 0.1) -> float:
    if matrix.shape[0] < 8 or target.std() <= 0.0:
        return 0.0
    indices = np.arange(matrix.shape[0])
    test_mask = indices % 5 == 0
    train_mask = ~test_mask
    if test_mask.sum() < 2 or train_mask.sum() < 4:
        return 0.0
    x_train = matrix[train_mask]
    x_test = matrix[test_mask]
    y_train = target[train_mask]
    y_test = target[test_mask]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    penalty = np.eye(x_train.shape[1]) * float(ridge)
    penalty[-1, -1] = 0.0
    lhs = x_train.T @ x_train + penalty
    rhs = x_train.T @ y_train
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    prediction = x_test @ weights
    denom = float(((y_test - y_test.mean()) ** 2).sum())
    if denom <= 1e-12:
        return 0.0
    score = 1.0 - float(((y_test - prediction) ** 2).sum()) / denom
    return float(score)


def _pc1_corr(matrix: np.ndarray, target: np.ndarray) -> float:
    if matrix.shape[0] < 3 or target.std() <= 0.0:
        return 0.0
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    if singular_values[0] <= 1e-12:
        return 0.0
    pc1_scores = u[:, 0] * singular_values[0]
    return _pearson(pc1_scores, target)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 0.0
    return float((left @ right) / denom)


def _as_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"embeddings must be a 2D matrix, got {matrix.shape}")
    return matrix


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def _even_subset(count: int, max_items: int) -> np.ndarray:
    if count <= max_items:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, max_items).astype(int))
