from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margins: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            f"embedding shape mismatch: anchor={anchor.shape}, "
            f"positive={positive.shape}, negative={negative.shape}"
        )
    if margins.ndim != 1 or weights.ndim != 1:
        raise ValueError("margins and weights must be 1D tensors")
    if margins.shape[0] != anchor.shape[0] or weights.shape[0] != anchor.shape[0]:
        raise ValueError("margins/weights batch size must match embeddings")

    pos_cos = F.cosine_similarity(anchor, positive, dim=-1)
    neg_cos = F.cosine_similarity(anchor, negative, dim=-1)
    pos_dist = 1.0 - pos_cos
    neg_dist = 1.0 - neg_cos
    raw = F.relu(pos_dist - neg_dist + margins)
    loss = (raw * weights).mean()
    stats = {
        "pos_cos": pos_cos.detach(),
        "neg_cos": neg_cos.detach(),
        "violations": (raw.detach() > 0).float(),
        "raw_loss": raw.detach(),
    }
    return loss, stats


def matryoshka_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margins: torch.Tensor,
    weights: torch.Tensor,
    prefix_dims: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            f"embedding shape mismatch: anchor={anchor.shape}, "
            f"positive={positive.shape}, negative={negative.shape}"
        )
    full_dim = anchor.shape[-1]
    dims = sorted({int(value) for value in prefix_dims if 0 < int(value) < full_dim})
    if not dims:
        zero = anchor.new_tensor(0.0)
        return zero, {"loss": zero.detach()}

    losses: list[torch.Tensor] = []
    stats: dict[str, torch.Tensor] = {}
    for dim in dims:
        prefix_anchor = F.normalize(anchor[..., :dim], p=2, dim=-1, eps=1e-8)
        prefix_positive = F.normalize(positive[..., :dim], p=2, dim=-1, eps=1e-8)
        prefix_negative = F.normalize(negative[..., :dim], p=2, dim=-1, eps=1e-8)
        dim_loss, dim_stats = weighted_triplet_margin_loss(
            prefix_anchor,
            prefix_positive,
            prefix_negative,
            margins,
            weights,
        )
        losses.append(dim_loss)
        stats[f"loss_dim_{dim}"] = dim_loss.detach()
        stats[f"violation_dim_{dim}"] = dim_stats["violations"].mean().detach()
        stats[f"pos_cos_dim_{dim}"] = dim_stats["pos_cos"].mean().detach()
        stats[f"neg_cos_dim_{dim}"] = dim_stats["neg_cos"].mean().detach()
    loss = torch.stack(losses).mean()
    stats["loss"] = loss.detach()
    return loss, stats


def scalar_stats(values: torch.Tensor, prefix: str) -> dict[str, float]:
    if values.numel() == 0:
        raise ValueError(f"cannot summarize empty tensor for {prefix}")
    values = values.detach().float().cpu()
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_median": float(values.median().item()),
        f"{prefix}_p05": float(torch.quantile(values, 0.05).item()),
        f"{prefix}_p95": float(torch.quantile(values, 0.95).item()),
    }


def embedding_spread_loss(
    embeddings: torch.Tensor,
    min_dim_std: float,
    variance_weight: float,
    covariance_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [batch, dim], got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        zero = embeddings.new_tensor(0.0)
        return zero, {
            "dim_std_mean": zero.detach(),
            "variance_loss": zero.detach(),
            "covariance_loss": zero.detach(),
            "top_eigen_ratio": zero.detach(),
            "effective_rank": zero.detach(),
        }

    embeddings = embeddings.float()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    dim_std = centered.std(dim=0, unbiased=False)
    variance_loss = F.relu(float(min_dim_std) - dim_std).mean()

    covariance = centered.T @ centered / float(max(1, embeddings.shape[0] - 1))
    diagonal = torch.diagonal(covariance)
    covariance_loss = (covariance - torch.diag(diagonal)).square().sum() / float(
        embeddings.shape[1] * max(1, embeddings.shape[1] - 1)
    )

    total = variance_loss * float(variance_weight) + covariance_loss * float(covariance_weight)
    if embeddings.shape[0] < 3:
        zero = embeddings.new_tensor(0.0)
        diagnostics = {"top_eigen_ratio": zero.detach(), "effective_rank": zero.detach()}
    else:
        diagnostics = _eigen_diagnostics(covariance.detach())
    diagnostics.update(
        {
            "dim_std_mean": dim_std.detach().mean(),
            "variance_loss": variance_loss.detach(),
            "covariance_loss": covariance_loss.detach(),
        }
    )
    return total, diagnostics


def _eigen_diagnostics(covariance: torch.Tensor) -> dict[str, torch.Tensor]:
    try:
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    except NotImplementedError:
        eigenvalues = torch.linalg.eigvalsh(covariance.cpu()).clamp_min(0.0).to(covariance.device)
    total = eigenvalues.sum().clamp_min(1e-12)
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {
        "top_eigen_ratio": (eigenvalues.max() / total).detach(),
        "effective_rank": entropy.exp().detach(),
    }


def embedding_spread_diagnostics(embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [batch, dim], got {embeddings.shape}")
    if embeddings.shape[0] < 3:
        zero = embeddings.new_tensor(0.0)
        return {"top_eigen_ratio": zero.detach(), "effective_rank": zero.detach()}

    values = embeddings.detach().float()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(max(1, centered.shape[0] - 1))
    return _eigen_diagnostics(covariance)
