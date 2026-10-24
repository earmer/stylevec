from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import twin_columns
from leak import LeakState, PairQueue, leak_settings_from_config, leak_step_loss, twin_pairs, validate_leak_config
from model import encode_triplet_outputs
from records import TripletRecord
from test_content_head import encoder, token_batch
from train import twin_mode


def row(negative: str, twin: str | None) -> TripletRecord:
    return TripletRecord("锚点", "正例", negative, "origin_vs_para", 1.0, 0.1, meta={"twin_text": twin} if twin else {})


def test_twin_columns_mark_twins_that_are_the_negative() -> None:
    out = twin_columns([row("别人", "改写"), row("改写", "改写"), row("别人", None)], lambda texts: texts)
    assert out["has_twin"].tolist() == [True, False, False]
    assert out["twin_is_negative"].tolist() == [False, True, False]


def test_twin_pairs_read_the_negative_when_it_is_the_twin() -> None:
    anchor, positive, negative, twin = (torch.randn(3, 4) for _ in range(4))
    outputs = {"embeddings": (anchor, positive, negative), "twin": twin}
    batch = {"has_twin": torch.tensor([True, False, False]), "twin_is_negative": torch.tensor([False, True, False])}
    style, target = twin_pairs(outputs, batch)
    assert torch.equal(style, anchor[:2])
    assert torch.equal(target, torch.stack([twin[0], negative[1]]))


def test_pair_queue_puts_this_batch_first_and_keeps_the_newest() -> None:
    queue = PairQueue(3)
    first = torch.ones(2, 2, requires_grad=True)
    style, _ = queue.extend(first, torch.zeros(2, 2))
    assert style.shape[0] == 2
    second = torch.full((2, 2), 2.0, requires_grad=True)
    style, twin = queue.extend(second, torch.ones(2, 2))
    assert style.shape[0] == 4 and torch.equal(style[:2], second) and not queue.style.requires_grad
    assert queue.style.shape[0] == 3 and torch.equal(queue.style[:2], second.detach())
    assert torch.equal(PairQueue(0).extend(first, first)[0], first)


def test_detached_twin_matches_the_graded_one_without_gradient() -> None:
    model = encoder(0)
    batch = {**token_batch(), "twin": token_batch()["positive"]}
    graded = encode_triplet_outputs(model, batch)
    detached = encode_triplet_outputs(model, batch, detach_twin=True)
    assert torch.allclose(detached["twin"], graded["twin"], atol=1e-5)
    assert not detached["twin"].requires_grad and graded["twin"].requires_grad
    assert len(detached["embeddings"]) == 3


def config(**regularization) -> dict:
    return {"train": {"loss": "triplet", "max_steps": 3000}, "model": {}, "regularization": regularization}


def test_twin_mode_and_leak_settings_are_checked() -> None:
    assert twin_mode(config()) == "none"
    on = config(leak_method="none", leak_weight=1.0)
    assert twin_mode(on) == "none"
    graded = {**config(), "train": {"loss": "infonce", "max_steps": 3000, "infonce_twin_temperature": 0.03}}
    assert twin_mode(graded) == "grad"
    for bad in ({"leak_method": "typo"}, {"leak_weight": -1.0}, {"leak_queue": -1}, {"leak_until_step": 3001}):
        with pytest.raises(ValueError):
            validate_leak_config(config(**bad))


def test_leak_off_adds_nothing() -> None:
    state = LeakState(leak_settings_from_config(config()))
    outputs = {"embeddings": tuple(torch.randn(2, 4) for _ in range(3)), "twin": torch.randn(2, 4)}
    batch = {"has_twin": torch.tensor([True, True]), "twin_is_negative": torch.tensor([False, False])}
    loss, stats = leak_step_loss(state, outputs, batch, 0)
    assert float(loss) == 0.0 and stats == {}


def test_content_null_learns_the_twin_axis_and_projects_it_out() -> None:
    from model import ContentNull

    torch.manual_seed(0)
    null = ContentNull(dim=8, dims=1, strength=1.0, momentum=0.5, warmup=2)
    axis = torch.nn.functional.one_hot(torch.tensor(3), 8).float()
    for _ in range(3):
        null.update(torch.randn(64, 1) * 5.0 * axis + 0.1 * torch.randn(64, 8))
    assert abs(float(null.basis[0] @ axis)) > 0.99
    x = torch.randn(4, 8)
    assert torch.allclose(null(x)[:, 3], torch.zeros(4), atol=0.05)
    assert float(ContentNull(8, 1, 1.0, 0.5, warmup=5).basis.abs().sum()) == 0.0


def test_subspace_method_feeds_the_null_and_needs_the_model_option() -> None:
    from model import ContentNull

    null = ContentNull(dim=4, dims=1, strength=1.0, momentum=0.5, warmup=1)
    state = LeakState(leak_settings_from_config(config(leak_method="subspace")), null=null)
    emb = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    outputs = {"embeddings": (emb, emb, emb), "unprojected": (emb, emb, emb), "twin": emb, "twin_unprojected": emb}
    batch = {"has_twin": torch.tensor([True, True, False]), "twin_is_negative": torch.tensor([False, False, True])}
    loss, stats = leak_step_loss(state, outputs, batch, 0)
    assert float(loss) == 0.0 and int(null.updates) == 1 and "null_share_anchor" in stats
    with pytest.raises(ValueError):
        validate_leak_config(config(leak_method="subspace"))
    with pytest.raises(ValueError):
        LeakState(leak_settings_from_config(config(leak_method="subspace")))
