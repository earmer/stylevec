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


def test_kernel_alignment_is_high_for_copies_and_low_for_independent_rows() -> None:
    from leak import kernel_alignment

    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1)
    noise = torch.nn.functional.normalize(torch.randn(64, 16), dim=-1)
    for kernel in ("rbf", "linear"):
        assert kernel_alignment(x, x, kernel) > 0.99
        assert abs(float(kernel_alignment(x, noise, kernel))) < 0.1
    style = x.clone().requires_grad_(True)
    kernel_alignment(style, x).backward()
    assert style.grad is not None and style.grad.abs().sum() > 0


def test_hsic_method_returns_weighted_alignment_and_queues_pairs() -> None:
    state = LeakState(leak_settings_from_config(config(leak_method="hsic", leak_weight=2.0, leak_queue=8)))
    anchor = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    outputs = {"embeddings": (anchor, anchor, anchor), "twin": anchor.clone()}
    batch = {"has_twin": torch.ones(4, dtype=torch.bool), "twin_is_negative": torch.zeros(4, dtype=torch.bool)}
    loss, stats = leak_step_loss(state, outputs, batch, 0)
    assert torch.isclose(loss, 2.0 * stats["hsic_cka"]) and stats["hsic_cka"] > 0.99
    leak_step_loss(state, outputs, batch, 1)
    assert state.queue.style.shape[0] == 8
    late = LeakState(leak_settings_from_config(config(leak_method="hsic", leak_weight=2.0, leak_until_step=5)))
    assert float(leak_step_loss(late, outputs, batch, 5)[0]) == 0.0


def probe_state(weight: float = 1.0, warmup: int = 0) -> LeakState:
    settings = leak_settings_from_config(config(leak_method="adversary", leak_weight=weight, leak_probe_warmup=warmup))
    probe = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8))
    return LeakState(settings, probe=probe)


def test_adversary_probe_learns_twins_and_only_the_encoder_term_reaches_style() -> None:
    from leak import adversary_loss

    torch.manual_seed(0)
    state = probe_state()
    twins = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    style = twins.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(state.probe.parameters(), lr=0.02)
    for step in range(200):
        probe_ce, confusion, stats = adversary_loss(state.probe, style, twins, 16, state.settings, step)
        optimizer.zero_grad()
        probe_ce.backward()
        optimizer.step()
    assert stats["probe_acc"] > 0.9 and style.grad is None
    _, confusion, _ = adversary_loss(state.probe, style, twins, 16, state.settings, 0)
    confusion.backward()
    assert style.grad.abs().sum() > 0  # the frozen probe's weights get nothing (below)
    before = [p.grad.clone() for p in state.probe.parameters()]
    _, confusion, _ = adversary_loss(state.probe, style, twins, 16, state.settings, 0)
    confusion.backward()
    assert all(torch.equal(b, p.grad) for b, p in zip(before, state.probe.parameters()))


def test_adversary_ramp_and_config_checks() -> None:
    from leak import adversary_loss

    state = probe_state(warmup=10)
    twins = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    _, early, stats = adversary_loss(state.probe, twins.clone(), twins, 6, state.settings, 0)
    assert float(stats["ramp"]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        validate_leak_config(config(leak_method="adversary", leak_weight=1.0))
    with pytest.raises(ValueError):
        LeakState(leak_settings_from_config(config(leak_method="adversary", leak_weight=1.0)))

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
