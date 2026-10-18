from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from losses import embedding_spread_loss


def test_spread_loss_matches_manual_computation() -> None:
    torch.manual_seed(0)
    embeddings = torch.randn(8, 16)
    total, stats = embedding_spread_loss(embeddings, 0.03, 0.01, 0.005)

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    dim_std = centered.std(dim=0, unbiased=False)
    variance_loss = torch.relu(0.03 - dim_std).mean()
    covariance = centered.T @ centered / 7.0
    diagonal = torch.diagonal(covariance)
    covariance_loss = (covariance - torch.diag(diagonal)).square().sum() / (16 * 15)
    expected = variance_loss * 0.01 + covariance_loss * 0.005

    assert torch.allclose(total, expected, atol=1e-6)
    assert abs(stats["variance_loss"].item() - variance_loss.item()) < 1e-6
    assert abs(stats["covariance_loss"].item() - covariance_loss.item()) < 1e-6
    assert 0.0 <= stats["top_eigen_ratio"].item() <= 1.0


def test_spread_loss_small_batch_returns_zero_diagnostics() -> None:
    embeddings = torch.randn(2, 8)
    _total, stats = embedding_spread_loss(embeddings, 0.03, 0.01, 0.005)
    assert stats["top_eigen_ratio"].item() == 0.0
    assert stats["effective_rank"].item() == 0.0


def test_correlation_mode_is_scale_free() -> None:
    from losses import embedding_spread_loss

    torch.manual_seed(0)
    base = torch.randn(64, 16)
    base[:, 1] = base[:, 0] + 0.1 * torch.randn(64)  # two nearly collinear dimensions
    _, small = embedding_spread_loss(base * 0.01, 0.0, 0.0, 1.0, "correlation")
    total, large = embedding_spread_loss(base, 0.0, 0.0, 1.0, "correlation")
    assert abs(small["correlation_loss"].item() - large["correlation_loss"].item()) < 1e-4
    assert small["covariance_loss"].item() < 1e-6 < large["covariance_loss"].item()
    assert torch.allclose(total, large["correlation_loss"])


def test_correlation_mode_rejects_unknown_mode() -> None:
    import pytest
    from losses import embedding_spread_loss

    with pytest.raises(ValueError):
        embedding_spread_loss(torch.randn(8, 4), 0.03, 0.01, 0.005, "pearson")


def test_uniformity_prefers_spread_points() -> None:
    from losses import uniformity_loss

    torch.manual_seed(0)
    spread = torch.nn.functional.normalize(torch.randn(32, 64), dim=-1)
    clumped = torch.nn.functional.normalize(torch.ones(32, 64) + 0.05 * torch.randn(32, 64), dim=-1)
    assert uniformity_loss([spread]).item() < uniformity_loss([clumped]).item()
    manual = torch.log(torch.exp(-2.0 * torch.pdist(spread).square()).mean())
    assert torch.allclose(uniformity_loss([spread]), manual, atol=1e-5)
