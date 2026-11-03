import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from named_traits import NamedTraitSampler, named_trait_loss
from traits import TRAITS


def test_named_trait_loss_is_low_when_each_dimension_tracks_its_trait():
    torch.manual_seed(0)
    scores = torch.rand(64, 3) * 4
    aligned = torch.cat([scores * 0.3 + 0.01 * torch.randn(64, 3), torch.randn(64, 13)], dim=1)
    shuffled = torch.cat([scores[torch.randperm(64)] * 0.3, torch.randn(64, 13)], dim=1)
    low, stats = named_trait_loss(aligned, scores)
    high, _ = named_trait_loss(shuffled, scores)
    assert low < 0.05 < high and float(stats["named_trait_corr"]) > 0.95
    embeddings = aligned.clone().requires_grad_(True)
    named_trait_loss(embeddings, scores)[0].backward()
    assert embeddings.grad[:, :3].abs().sum() > 0 and embeddings.grad[:, 3:].abs().sum() == 0


def test_a_trait_without_spread_in_the_batch_is_skipped():
    scores = torch.full((16, 2), 2.0)
    scores[:, 0] = torch.arange(16.0) / 4
    loss, stats = named_trait_loss(torch.cat([scores[:, :1], torch.randn(16, 3)], dim=1), scores)
    assert loss < 1e-4 and torch.isfinite(stats["named_trait_corr"])


def test_sampler_reads_jev_labels_in_the_configured_trait_order(tmp_path):
    names = ["archaic", "formality"]
    path = tmp_path / "labels.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(10):
            handle.write(json.dumps({"text": f"t{i}", "scores": {n: [float(i % 5), 0.9] for n in TRAITS}}) + "\n")
    sampler = NamedTraitSampler({"files": [str(path)], "names": names, "batch": 8, "weight": 1.0})
    texts, scores = sampler.sample()
    assert len(texts) == 8 and scores.shape == (8, 2)
    with pytest.raises(ValueError):
        NamedTraitSampler({"files": [str(path)], "names": ["nope"], "batch": 8, "weight": 1.0})


def test_pair_loss_rewards_differences_that_track_the_score_change():
    from named_traits import named_trait_pair_loss
    torch.manual_seed(0)
    content = torch.randn(32, 1, 4).repeat(1, 2, 1) * 5  # shared within each pair, unrelated to the scores
    scores = torch.rand(32, 2, 2) * 4
    tracks = (content[..., :2] + scores * 0.3).flatten(0, 1)
    ignores = (content[..., :2] + scores[torch.randperm(32)] * 0.3).flatten(0, 1)
    flat = scores.flatten(0, 1)
    _, good = named_trait_pair_loss(torch.cat([tracks, torch.randn(64, 3)], dim=1), flat)
    _, bad = named_trait_pair_loss(torch.cat([ignores, torch.randn(64, 3)], dim=1), flat)
    assert float(good["named_trait_delta_corr"]) > 0.95 > 0.5 > float(bad["named_trait_delta_corr"])


def test_sampler_pair_mode_returns_interleaved_pairs(tmp_path):
    path = tmp_path / "pairs.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for i in range(6):
            handle.write(json.dumps({"a": f"a{i}", "b": f"b{i}", "scores_a": {n: [1.0, 0.9] for n in TRAITS},
                                     "scores_b": {n: [3.0, 0.9] for n in TRAITS}}) + "\n")
    sampler = NamedTraitSampler({"files": [], "pair_files": [str(path)], "names": ["warmth"], "batch": 8,
                                 "weight": 1.0})
    texts, scores = sampler.sample()
    assert len(texts) == 8 and [t[0] for t in texts] == ["a", "b"] * 4
    assert scores[0::2].eq(1.0).all() and scores[1::2].eq(3.0).all()
