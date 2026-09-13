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
