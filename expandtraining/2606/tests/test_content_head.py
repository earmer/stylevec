from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from losses import content_pair_loss, cross_correlation_penalty
from model import QwenStyleEncoder, encode_triplet, encode_triplet_outputs, trainable_state_dict
from test_model_shapes import FakeBackbone, settings
from train import CONTENT_PAIR_RULES, HEAD_MODULES, content_duplicates, content_settings_from_config, content_step_loss


def encoder(content_dim: int) -> QwenStyleEncoder:
    torch.manual_seed(0)
    return QwenStyleEncoder(
        FakeBackbone(hidden_size=4), hidden_size=4,
        settings=dataclasses.replace(settings("last", ()), content_dim=content_dim),
    )


def token_batch() -> dict:
    tokens = {"input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]), "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]])}
    other = {"input_ids": torch.tensor([[6, 7], [8, 9]]), "attention_mask": torch.ones(2, 2, dtype=torch.long)}
    return {"anchor": tokens, "positive": other, "negative": tokens}


def test_content_head_is_separate_and_off_by_default() -> None:
    assert "content" not in encoder(0)(**token_batch()["anchor"])
    out = encoder(2)(**token_batch()["anchor"])
    assert out["embeddings"].shape == (2, 3) and out["content"].shape == (2, 2)
    assert torch.allclose(out["content"].norm(dim=-1), torch.ones(2), atol=1e-6)
    assert any(name.startswith("content_head.") for name in trainable_state_dict(encoder(2)))
    assert "content_head." in HEAD_MODULES


def test_encode_triplet_outputs_combined_matches_separate() -> None:
    model = encoder(2)
    separate = encode_triplet_outputs(model, token_batch())
    combined = encode_triplet_outputs(model, token_batch(), combine_forward=True)
    for key in ("embeddings", "content"):
        for left, right in zip(separate[key], combined[key]):
            assert torch.allclose(left, right, atol=1e-5)
    anchor, _positive, _negative, norms = encode_triplet(model, token_batch())
    assert torch.equal(anchor, separate["embeddings"][0]) and norms.shape == (6,)


def unit(*rows) -> torch.Tensor:
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def test_content_pair_loss_matches_manual_cross_entropy_and_masks_duplicates() -> None:
    anchor = unit([1, 0], [0, 1], [1, 1])
    positive = unit([0, 1], [1, 0], [1, -1])
    negative = unit([0.9, 0.1], [0.2, 1], [-1, 1])
    is_pair = torch.tensor([True, True, False])
    duplicate = torch.zeros(3, 9, dtype=torch.bool)
    duplicate[0, 3 + 2] = True  # row 0's anchor text reappears as row 2's positive
    loss, stats = content_pair_loss(anchor, positive, negative, is_pair, duplicate, temperature=0.5)
    sims = torch.cat([anchor @ negative.T, anchor @ positive.T, anchor @ anchor.T], dim=1) / 0.5
    expected = []
    for row in (0, 1):
        keep = torch.ones(9, dtype=torch.bool)
        keep[6 + row] = False
        keep &= ~duplicate[row]
        expected.append(torch.logsumexp(sims[row][keep], 0) - sims[row, row])
    assert torch.allclose(loss, torch.stack(expected).mean(), atol=1e-5)
    assert stats["pairs"] == 2 and 0.0 <= stats["accuracy"] <= 1.0
    assert stats["pair_cos"] > stats["speaker_cos"]


def test_content_pair_loss_without_pairs_is_zero_but_connected() -> None:
    anchor = unit([1, 0], [0, 1]).requires_grad_()
    loss, stats = content_pair_loss(anchor, anchor, anchor, torch.tensor([False, False]), torch.zeros(2, 6, dtype=torch.bool))
    assert loss.item() == 0.0 and stats["pairs"] == 0
    loss.backward()
    assert anchor.grad is not None


def test_cross_correlation_penalty_sees_a_copied_content_direction() -> None:
    torch.manual_seed(0)
    content = torch.randn(256, 2)
    independent = torch.randn(256, 8)
    leaked = torch.cat([independent[:, :6], content], dim=1)
    low = cross_correlation_penalty(independent, content)
    high = cross_correlation_penalty(leaked, content)
    assert low < 0.1 and high > 0.9


def test_content_step_loss_trains_the_head_and_detaches_content_in_the_leak() -> None:
    model = encoder(2)
    batch = {
        **token_batch(),
        "rule": ["origin_vs_para", "same_speaker_real"],
        "anchor_text": ["a", "b"], "positive_text": ["c", "d"], "negative_text": ["a2", "e"],
    }
    config = {"model": {"content_dim": 2}, "regularization": {"content_weight": 1.0, "content_leak_weight": 1.0}}
    settings_ = content_settings_from_config(config)
    loss, stats = content_step_loss(encode_triplet_outputs(model, batch), batch, settings_)
    assert stats["pairs"] == 1 and "leak" in stats
    loss.backward()
    assert model.content_head.weight.grad is not None and model.projection.weight.grad is not None

    model.zero_grad()
    leak_only = content_settings_from_config({**config, "regularization": {"content_weight": 0.0, "content_leak_weight": 1.0}})
    loss, _ = content_step_loss(encode_triplet_outputs(model, batch), batch, leak_only)
    loss.backward()
    assert model.content_head.weight.grad is None or torch.count_nonzero(model.content_head.weight.grad) == 0


def test_content_step_loss_is_off_without_the_head() -> None:
    model = encoder(0)
    batch = {**token_batch(), "rule": ["origin_vs_para"] * 2}
    loss, stats = content_step_loss(encode_triplet_outputs(model, batch), batch, content_settings_from_config({"model": {}}))
    assert loss.item() == 0.0 and stats == {}


def test_content_duplicates_marks_repeated_texts() -> None:
    batch = {"anchor_text": ["x", "y"], "positive_text": ["p", "x"], "negative_text": ["x2", "q"]}
    dup = content_duplicates(batch, torch.device("cpu"))
    # Candidates: negatives [x2, q], positives [p, x], anchors [x, y].
    assert dup[0].tolist() == [True, False, False, True, True, False]
    assert dup[1].tolist() == [False, True, False, False, False, True]


def test_content_pair_rules_are_real_rules() -> None:
    assert {"origin_vs_para", "paratarget_original_push"} <= CONTENT_PAIR_RULES


@pytest.mark.parametrize("key,value", [("content_weight", -1.0), ("content_leak_weight", -0.1), ("content_temperature", 0.0)])
def test_config_refuses_bad_content_settings(key, value) -> None:
    from config import validate_loss_config

    with pytest.raises(ValueError):
        validate_loss_config({"train": {}, "model": {"content_dim": 8}, "regularization": {key: value}})
