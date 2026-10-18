from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from train import accumulation_window


def test_accumulation_window_full_windows() -> None:
    assert accumulation_window(0, 10, 4) == 4
    assert accumulation_window(1, 10, 4) == 4
    assert accumulation_window(3, 10, 4) == 4


def test_accumulation_window_partial_tail() -> None:
    assert accumulation_window(8, 10, 4) == 2
    assert accumulation_window(9, 10, 4) == 2


def test_accumulation_window_smaller_than_epoch() -> None:
    assert accumulation_window(0, 2, 8) == 2
    assert accumulation_window(1, 2, 8) == 2


def test_accumulation_window_disabled() -> None:
    assert accumulation_window(0, 10, 1) == 1
    assert accumulation_window(7, 10, 0) == 1


def test_weights_snapshot_restores_every_trained_parameter(tmp_path: Path) -> None:
    from model import trainable_state_dict
    from train import save_weights, snapshot_steps
    from validate import load_checkpoint_for_validation

    trained = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    save_weights(trainable_state_dict(trained), tmp_path / "step_00010", 10, "h")
    fresh = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    # Only weights.pt, as after --prune or in a step snapshot: no training_state.pt beside it.
    load_checkpoint_for_validation(fresh, tmp_path / "step_00010")
    for (name, a), (_, b) in zip(trained.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), name
    assert snapshot_steps({"train": {"keep_weights_at": [0.35, 0.70, 1.0]}}, 3000) == {1050, 2100}
    assert snapshot_steps({"train": {}}, 3000) == set()


def test_untrained_baseline_projection_is_fixed_and_bias_free() -> None:
    from validate import untrained_projection

    model = torch.nn.Module()
    model.projection = torch.nn.Linear(16, 4)
    untrained_projection(model)
    first = model.projection.weight.clone()
    untrained_projection(model)
    assert torch.equal(first, model.projection.weight)
    assert torch.count_nonzero(model.projection.bias) == 0
