import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import pandas as pd

from trait_predictor import NAMES, AttentionReadout, agreement, is_clean, rank_gain, scan_targets, predicted_change, trait_vector, weight_entropy_share


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


def test_agreement_counts_exact_and_within_one_and_the_shift():
    result = agreement([0, 1, 2, 3], [0, 2, 2, 1])
    assert result["exact"] == 0.5 and result["within_one"] == 0.75 and result["mean_second_minus_first"] == -0.25


def test_agreement_has_no_spearman_on_a_constant_rating():
    assert agreement([1, 1, 1], [0, 1, 2])["spearman"] is None


def test_scan_targets_drops_a_pair_with_a_broken_run_and_averages_the_rest():
    key = pd.DataFrame({"id": [1, 2]})
    answers = [{"id": 1, "run": r, "answer": a} for r, a in (("1", "1y"), ("2", "3y"), ("careful", "2p"))]
    answers += [{"id": 2, "run": r, "answer": a} for r, a in (("1", "1y"), ("2", "x"), ("careful", "2y"))]
    kept = scan_targets(key, answers)
    assert kept["id"].tolist() == [1] and kept["scan"][0] == 2 and kept["plain"][0] == 2


def test_rank_gain_is_positive_when_the_score_ranks_better_than_the_base():
    target = np.arange(50.0)
    noisy = np.random.default_rng(0).permutation(target)
    result = rank_gain(target, noisy, target, boots=50)
    assert result["rho"] == 1 and result["gain"] > 0.5 and result["ci99"][0] > 0
