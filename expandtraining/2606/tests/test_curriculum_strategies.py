from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curriculum import (
    DEFAULT_STAGES,
    curriculum_order,
    curriculum_settings,
    expected_exposure,
    rule_tier,
    schedule_summary,
    stage_weights,
)
from records import TripletRecord
from train import train_loader_shuffles

RULES = ("same_speaker_real", "origin_vs_para", "same_speaker_topic_matched")  # easy, mid, hard


def rows(rules: tuple[str, ...] = RULES, n: int = 50) -> list[TripletRecord]:
    return [TripletRecord(f"a{i}", "p", "n", rule, 1.0, 0.1) for rule in rules for i in range(n)]


def share(chunk: list[TripletRecord], tier: str, overrides: dict[str, str] | None = None) -> float:
    return float(np.mean([rule_tier(r.rule, overrides) == tier for r in chunk]))


def mode(name: str, **extra) -> dict:
    return {"curriculum": {"mode": name, **extra}}


def test_staged_anti_and_mixed_spend_identical_exposure() -> None:
    counts = {rule: 50 for rule in RULES}
    draws = {m: {r: v["draws"] for r, v in expected_exposure(counts, mode(m), 30000)["rules"].items()}
             for m in ("staged", "anti", "mixed")}
    assert draws["staged"] == draws["anti"] == draws["mixed"]
    order = curriculum_order(rows(), mode("anti"), seed=0, total=3000)
    assert share(order[:300], "hard") > 0.4 > 0.15 > share(order[-300:], "hard")


def test_mixed_matches_staged_exposure_when_a_tier_is_empty() -> None:
    easy_mid = ("same_speaker_real", "origin_vs_para")
    counts = {rule: 50 for rule in easy_mid}
    staged = expected_exposure(counts, mode("staged"), 20000)["rules"]
    mixed = expected_exposure(counts, mode("mixed"), 20000)["rules"]
    assert {r: v["draws"] for r, v in staged.items()} == {r: v["draws"] for r, v in mixed.items()}
    realized = {m: Counter(r.rule for r in curriculum_order(rows(easy_mid), mode(m), 0, 20000)) for m in ("staged", "mixed")}
    assert abs(realized["staged"]["origin_vs_para"] - realized["mixed"]["origin_vs_para"]) < 400


def test_linear_pacing_removes_the_jump_at_stage_boundaries() -> None:
    step_jump = stage_weights(DEFAULT_STAGES, 0.351)["hard"] - stage_weights(DEFAULT_STAGES, 0.349)["hard"]
    linear_jump = stage_weights(DEFAULT_STAGES, 0.351, "linear")["hard"] - stage_weights(DEFAULT_STAGES, 0.349, "linear")["hard"]
    assert step_jump > 0.15 and abs(linear_jump) < 0.01
    hard = [stage_weights(DEFAULT_STAGES, p / 10 + 0.05, "linear")["hard"] for p in range(10)]
    assert hard == sorted(hard) and hard[0] == pytest.approx(0.05) and hard[-1] == pytest.approx(0.5)


def test_uniform_ignores_tiers() -> None:
    counts = Counter(r.rule for r in curriculum_order(rows(), mode("uniform"), seed=0, total=3000))
    assert all(abs(counts[rule] - 1000) < 100 for rule in RULES)
    exposure = expected_exposure({rule: 50 for rule in RULES}, mode("uniform"), 3000)
    assert {v["draws"] for v in exposure["rules"].values()} == {1000.0}


def test_tier_overrides_move_and_drop_rules() -> None:
    overrides = {"same_speaker_topic_matched": "drop", "origin_vs_para": "hard"}
    order = curriculum_order(rows(), mode("staged", tiers=overrides), seed=0, total=3000)
    assert not any(r.rule == "same_speaker_topic_matched" for r in order)
    assert share(order[-300:], "hard", overrides) > 0.4
    with pytest.raises(ValueError, match="curriculum.tiers"):
        curriculum_settings(mode("staged", tiers={"origin_vs_para": "hrad"}))
    with pytest.raises(ValueError, match="unknown tiers"):
        curriculum_settings({"curriculum": {"stages": [{"until": 1.0, "tiers": {"hrad": 1.0}}]}})
    with pytest.raises(ValueError, match="no rules left"):
        curriculum_order(rows(("origin_vs_para",)), mode("staged", tiers={"origin_vs_para": "drop"}), 0, 10)


def test_rule_shares_weight_rules_within_their_tier() -> None:
    easy = ("same_speaker_real", "length_course_genshin", "length_course_msynthstel")
    cfg = mode("mixed", rule_shares={"length_course_genshin": 0.5, "length_course_msynthstel": 0.5})
    exposure = expected_exposure({rule: 50 for rule in easy}, cfg, 4000)["rules"]
    assert exposure["same_speaker_real"]["draws"] == 2000.0
    assert exposure["length_course_genshin"]["draws"] == 1000.0
    counts = Counter(r.rule for r in curriculum_order(rows(easy), cfg, seed=0, total=4000))
    assert abs(counts["same_speaker_real"] - 2000) < 150
    with pytest.raises(ValueError, match="rule_shares"):
        curriculum_settings(mode("mixed", rule_shares={"same_speaker_real": 0}))


def test_schedule_summary_reports_realized_mix_and_repeats() -> None:
    order = curriculum_order(rows(), mode("staged"), seed=0, total=3000)
    summary = schedule_summary(order, mode("staged"), {rule: 50 for rule in RULES})
    assert len(summary["tier_share_by_phase"]) == 10
    assert summary["tier_share_by_phase"][0].get("hard", 0) < summary["tier_share_by_phase"][-1]["hard"]
    assert sum(v["draws"] for v in summary["rules"].values()) == 3000
    for v in summary["rules"].values():
        assert v["repeats"] == pytest.approx(v["draws"] / 50, abs=0.01)


def test_curriculum_disables_loader_shuffle() -> None:
    assert not train_loader_shuffles({"curriculum": {"enabled": True}, "train": {"rule_balanced_batches": False}})
    assert not train_loader_shuffles({"train": {"rule_balanced_batches": True}})
    assert train_loader_shuffles({"train": {}})
