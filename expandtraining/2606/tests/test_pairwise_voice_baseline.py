import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pairwise_voice_baseline import changed_parts, diff_counts, scan_average, untie_label_one


def test_changed_parts_finds_a_dropped_particle():
    assert changed_parts("没有哦？", "没有。") == ("哦？", "。")


def test_diff_counts_separates_removed_and_added_classes():
    counts = diff_counts("奴家没有哦", "我没有")
    assert counts[0] == 3 and counts[1] == 1  # 奴家哦 removed, 我 added
    assert counts[5] == 1 and counts[6] == 0  # particle removed, none added
    assert counts[9] == 1 and counts[10] == 1  # 奴 removed, 我 added


def test_untie_keeps_the_rule_order_and_spreads_only_label_one():
    labels = np.array([0, 1, 1, 1, 2])
    out = untie_label_one(labels, np.array([0.0, 0.0, 1.0, 2.0, 3.0]), np.zeros(5, bool))
    assert out[0] == 0 and out[4] == 2
    assert out[1] == pytest.approx(0.8) and out[3] == pytest.approx(1.2) and out[1] < out[2] < out[3]


def test_scan_average_skips_x_and_missing():
    assert scan_average(["1y", "x", None, "2p"]) == 1.5
