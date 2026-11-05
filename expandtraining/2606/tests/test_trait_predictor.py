import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from trait_predictor import NAMES, AttentionReadout, is_clean, predicted_change, trait_vector, weight_entropy_share


def test_trait_vector_keeps_the_score_and_the_trait_order():
    scores = {name: [float(i), 0.9] for i, name in enumerate(reversed(NAMES))}
    vector = trait_vector(scores)
    assert len(vector) == 17
    assert vector[0] == len(NAMES) - 1 and vector[-1] == 0


def test_predicted_change_is_the_mean_absolute_difference():
    a = np.array([[1.0, 3.0], [0.0, 0.0]])
    b = np.array([[2.0, 1.0], [0.0, 0.0]])
    assert predicted_change(a, b) == pytest.approx([1.5, 0.0])


def test_is_clean_drops_held_lines_and_lines_held_speakers_said():
    speakers = {"shared": {"kept", "held"}, "other": {"kept"}}
    assert is_clean("other", speakers, {"held"}, set())
    assert is_clean("never in the db", speakers, {"held"}, set())
    assert not is_clean("shared", speakers, {"held"}, set())
    assert not is_clean("other", speakers, {"held"}, {"other"})


def test_attention_readout_starts_as_mean_pooling_and_skips_padding():
    readout = AttentionReadout(4, 2, torch.tensor([1.0, 2.0])).eval()
    readout.values.data = torch.ones(2, 4)
    hidden = torch.tensor([[[1.0, 0, 0, 0], [3.0, 0, 0, 0], [99.0, 0, 0, 0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert readout.token_weights(hidden, mask)[0, 0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert readout(hidden, mask)[0].tolist() == pytest.approx([3.0, 4.0])


def test_attention_readout_can_silence_unchanged_tokens():
    readout = AttentionReadout(2, 1, torch.zeros(1)).eval()
    readout.values.data = torch.tensor([[1.0, 0.0]])
    readout.queries.data = torch.tensor([[0.0, 100.0]])
    same = [[0.0, -1.0]] * 9
    mask = torch.ones(1, 10, dtype=torch.long)
    a = torch.tensor([same + [[1.0, 1.0]]])
    b = torch.tensor([same + [[-1.0, 1.0]]])
    assert (readout(a, mask) - readout(b, mask)).abs().item() > 1.9


def test_attention_readout_queries_start_learning_on_the_second_step():
    for scorer_dim in (0, 3):
        torch.manual_seed(0)
        readout = AttentionReadout(4, 2, torch.zeros(2), scorer_dim)
        readout.dropout.p = 0.0
        optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
        hidden = torch.randn(3, 5, 4)
        mask = torch.ones(3, 5, dtype=torch.long)
        target = torch.randn(3, 2)
        (readout(hidden, mask) - target).pow(2).sum().backward()
        assert readout.values.grad.abs().max() > 0
        assert readout.queries.grad.abs().max() == 0
        optimizer.step()
        optimizer.zero_grad()
        (readout(hidden, mask) - target).pow(2).sum().backward()
        assert readout.queries.grad.abs().max() > 0


def test_weight_entropy_share_is_one_for_uniform_and_small_for_one_token():
    mask = torch.ones(1, 4, dtype=torch.long)
    assert weight_entropy_share(torch.full((1, 2, 4), 0.25), mask) == pytest.approx(1.0)
    assert weight_entropy_share(torch.tensor([[[1.0, 0, 0, 0]]]), mask) == pytest.approx(0.0)
