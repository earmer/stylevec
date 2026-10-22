from __future__ import annotations

import math
from typing import Callable

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


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margins: torch.Tensor,
    weights: torch.Tensor,
    anchor_groups: torch.Tensor,
    negative_groups: torch.Tensor,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """In-batch InfoNCE on cosine / temperature. Row i's candidates: every positive (its own is the target),
    every other anchor, and every mined negative (its own always kept).

    anchor_groups[j] is the style of anchor j and positive j, negative_groups[j] that of negative j. Another
    row's candidate with anchor i's style is a false negative and is dropped from the denominator, not
    pushed away. Distinct ids that never match (e.g. negative values per row) mean "unknown, keep".
    Margins only feed the hinge-style `violations` diagnostic, so logs stay comparable with the triplet loss.
    """
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError(
            f"embedding shape mismatch: anchor={anchor.shape}, "
            f"positive={positive.shape}, negative={negative.shape}"
        )
    batch = anchor.shape[0]
    for name, values in (("margins", margins), ("weights", weights), ("anchor_groups", anchor_groups),
                         ("negative_groups", negative_groups)):
        if values.ndim != 1 or values.shape[0] != batch:
            raise ValueError(f"{name} must be a 1D tensor of batch size {batch}, got {tuple(values.shape)}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # float32 even under autocast: bf16 logits at temperature 0.05 lose most of the cosine resolution.
    with torch.autocast(device_type=anchor.device.type, enabled=False):
        a = F.normalize(anchor.float(), p=2, dim=-1, eps=1e-8)
        p = F.normalize(positive.float(), p=2, dim=-1, eps=1e-8)
        n = F.normalize(negative.float(), p=2, dim=-1, eps=1e-8)
        sim_p, sim_a, sim_n = a @ p.T, a @ a.T, a @ n.T

    eye = torch.eye(batch, dtype=torch.bool, device=anchor.device)
    same_style = anchor_groups[:, None] == anchor_groups[None, :]
    same_negative = anchor_groups[:, None] == negative_groups[None, :]
    # Positives share their anchor's style, so one mask covers both the positive and the anchor block.
    mask_style = same_style & ~eye
    mask_n = same_negative & ~eye
    blocked = torch.cat([mask_style, mask_style | eye, mask_n], dim=1)
    logits = torch.cat([sim_p, sim_a, sim_n], dim=1) / float(temperature)
    logits = logits.masked_fill(blocked, torch.finfo(logits.dtype).min)
    targets = torch.arange(batch, device=anchor.device)
    raw = F.cross_entropy(logits, targets, reduction="none")
    loss = (raw * weights.float()).mean()

    pos_cos = torch.diagonal(sim_p)
    neg_cos = torch.diagonal(sim_n)
    others = max(1, 3 * (batch - 1))
    masked = (2 * mask_style.sum(dim=1) + mask_n.sum(dim=1)).float() / float(others)
    stats = {
        "pos_cos": pos_cos.detach(),
        "neg_cos": neg_cos.detach(),
        "violations": (((1.0 - pos_cos) - (1.0 - neg_cos) + margins.float()) > 0).float().detach(),
        "raw_loss": raw.detach(),
        "masked_fraction": masked.mean().detach(),
        "accuracy": (logits.argmax(dim=1) == targets).float().mean().detach(),
    }
    return loss, stats


def content_pair_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    is_pair: torch.Tensor,
    duplicate: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """InfoNCE on the content head, for rows whose anchor and negative share content (a line and its
    paraphrase). Row i's target is its own negative; its candidates are every negative, every positive and
    every other anchor. Positives are the same speaker with different content, so the head is pushed to
    ignore speaker. duplicate[i, j] marks candidate text j (in [negatives, positives, anchors] order) as
    identical to row i's anchor or negative text; those are dropped from the denominator, not pushed.
    Rows with is_pair False are not queries but stay candidates. Returns 0 when the batch has no pair.
    """
    batch = anchor.shape[0]
    if is_pair.shape != (batch,) or duplicate.shape != (batch, 3 * batch):
        raise ValueError(f"is_pair must be ({batch},) and duplicate ({batch}, {3 * batch}), "
                         f"got {tuple(is_pair.shape)} and {tuple(duplicate.shape)}")
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    with torch.autocast(device_type=anchor.device.type, enabled=False):
        a, p, n = (F.normalize(x.float(), p=2, dim=-1, eps=1e-8) for x in (anchor, positive, negative))
        sims = torch.cat([a @ n.T, a @ p.T, a @ a.T], dim=1)
    eye = torch.eye(batch, dtype=torch.bool, device=anchor.device)
    no_eye = torch.zeros_like(eye)
    # Keep the target (own negative) and drop the anchor itself.
    blocked = (duplicate & ~torch.cat([eye, no_eye, no_eye], dim=1)) | torch.cat([no_eye, no_eye, eye], dim=1)
    logits = (sims / float(temperature)).masked_fill(blocked, torch.finfo(sims.dtype).min)
    targets = torch.arange(batch, device=anchor.device)
    raw = F.cross_entropy(logits, targets, reduction="none")
    pairs = is_pair.bool()
    count = pairs.sum()
    if int(count) == 0:
        # Still tied to the graph, so every rank sees the head's parameters used.
        return (sims * 0.0).sum(), {"pairs": count.float().detach()}
    loss = raw[pairs].mean()
    stats = {
        "pairs": count.float().detach(),
        "accuracy": (logits.argmax(dim=1) == targets)[pairs].float().mean().detach(),
        "pair_cos": torch.diagonal(sims[:, :batch])[pairs].mean().detach(),
        # Same speaker, different content: should fall well below pair_cos.
        "speaker_cos": torch.diagonal(sims[:, batch:2 * batch])[pairs].mean().detach(),
    }
    return loss, stats


def cross_correlation_penalty(style: torch.Tensor, content: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """How linearly visible each content dimension is in the style embedding: per content dimension, the
    sum over style dimensions of their squared batch correlation, averaged over content dimensions. With
    whitened style dimensions this is the R^2 of a linear readout; independent noise gives about
    style_dim / batch. The caller detaches content so only the style side is pushed away."""
    if style.ndim != 2 or content.ndim != 2 or style.shape[0] != content.shape[0]:
        raise ValueError(f"style and content must be [batch, dim] with one batch, got {style.shape}, {content.shape}")
    with torch.autocast(device_type=style.device.type, enabled=False):
        s = style.float() - style.float().mean(dim=0, keepdim=True)
        c = content.float() - content.float().mean(dim=0, keepdim=True)
        s = s / (s.pow(2).mean(dim=0, keepdim=True).sqrt() + eps)
        c = c / (c.pow(2).mean(dim=0, keepdim=True).sqrt() + eps)
        correlation = s.T @ c / float(style.shape[0])
    return correlation.pow(2).sum(dim=0).mean()


def matryoshka_prefix_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    prefix_dims: list[int] | tuple[int, ...],
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """`loss_fn` on each prefix below the full dimension, every prefix re-normalised first; the mean."""
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
        dim_loss, dim_stats = loss_fn(prefix_anchor, prefix_positive, prefix_negative)
        losses.append(dim_loss)
        stats[f"loss_dim_{dim}"] = dim_loss.detach()
        stats[f"violation_dim_{dim}"] = dim_stats["violations"].mean().detach()
        stats[f"pos_cos_dim_{dim}"] = dim_stats["pos_cos"].mean().detach()
        stats[f"neg_cos_dim_{dim}"] = dim_stats["neg_cos"].mean().detach()
    loss = torch.stack(losses).mean()
    stats["loss"] = loss.detach()
    return loss, stats


def matryoshka_triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margins: torch.Tensor,
    weights: torch.Tensor,
    prefix_dims: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return matryoshka_prefix_loss(
        anchor,
        positive,
        negative,
        prefix_dims,
        lambda a, p, n: weighted_triplet_margin_loss(a, p, n, margins, weights),
    )


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


DECORRELATION_MODES = ("covariance", "correlation")


def embedding_spread_loss(
    embeddings: torch.Tensor,
    min_dim_std: float,
    variance_weight: float,
    covariance_weight: float,
    decorrelation: str = "covariance",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Variance floor plus an off-diagonal penalty, weighted by `covariance_weight`.

    decorrelation="covariance" penalises raw covariances. On unit vectors each dimension's std is ~0.05,
    so they are ~1e-3 and the squared penalty ~1e-6: in effect off. "correlation" standardises each
    dimension first (VICReg/Barlow style), so the penalty is a mean squared correlation in [0, 1] and
    its gradient does not shrink with the dimension scale. Both values are always reported.
    """
    if decorrelation not in DECORRELATION_MODES:
        raise ValueError(f"decorrelation must be one of {DECORRELATION_MODES}, got {decorrelation!r}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [batch, dim], got {embeddings.shape}")
    if embeddings.shape[0] < 2:
        zero = embeddings.new_tensor(0.0)
        return zero, {
            "dim_std_mean": zero.detach(),
            "variance_loss": zero.detach(),
            "covariance_loss": zero.detach(),
            "correlation_loss": zero.detach(),
            "top_eigen_ratio": zero.detach(),
            "effective_rank": zero.detach(),
        }

    embeddings = embeddings.float()
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    dim_std = centered.std(dim=0, unbiased=False)
    variance_loss = F.relu(float(min_dim_std) - dim_std).mean()

    pairs = float(embeddings.shape[1] * max(1, embeddings.shape[1] - 1))
    covariance = centered.T @ centered / float(max(1, embeddings.shape[0] - 1))
    diagonal = torch.diagonal(covariance)
    covariance_loss = (covariance - torch.diag(diagonal)).square().sum() / pairs
    standardized = centered / dim_std.clamp_min(1e-4)
    correlation = standardized.T @ standardized / float(embeddings.shape[0])
    correlation_loss = (correlation - torch.diag(torch.diagonal(correlation))).square().sum() / pairs

    off_diagonal = correlation_loss if decorrelation == "correlation" else covariance_loss
    total = variance_loss * float(variance_weight) + off_diagonal * float(covariance_weight)
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
            "correlation_loss": correlation_loss.detach(),
        }
    )
    return total, diagnostics


def uniformity_loss(groups: list[torch.Tensor], t: float = 2.0) -> torch.Tensor:
    """Wang & Isola (2020): log E[exp(-t ||z_i - z_j||^2)] over pairs within each group, averaged.

    Pass anchors, positives and negatives as separate groups so a positive is never pushed away from its
    own anchor. Expects L2-normalised rows; lower is more uniform on the sphere.
    """
    values = []
    for group in groups:
        if group.ndim != 2:
            raise ValueError(f"embeddings must be [batch, dim], got {group.shape}")
        if group.shape[0] < 2:
            continue
        # Matrix form instead of torch.pdist, which MPS lacks.
        rows = group.float()
        norms = rows.square().sum(dim=-1)
        pairwise = (norms[:, None] + norms[None, :] - 2.0 * rows @ rows.T).clamp_min(0.0)
        upper = torch.triu_indices(rows.shape[0], rows.shape[0], offset=1, device=rows.device)
        squared = pairwise[upper[0], upper[1]]
        values.append(torch.logsumexp(-float(t) * squared, dim=0) - math.log(squared.numel()))
    if not values:
        return groups[0].new_tensor(0.0) if groups else torch.tensor(0.0)
    return torch.stack(values).mean()


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
