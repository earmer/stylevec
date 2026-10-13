from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablation import SpecError, compare, measured_tiers, plan, render_compare, render_plan, run_name

SPECS = sorted((Path(__file__).resolve().parents[1] / "configs" / "ablations").glob("*.yaml"))


def base_config() -> dict:
    return {
        "runtime": {"seed": 42, "device": "cpu"},
        "paths": {"base_model": "m"},
        "data": {"speakers": 10, "heldout_speakers": 2, "genshin_min_id_gap": 150},
        "model": {"base_model_path": "m", "max_length": 192},
        "train": {"batch_size": 4, "grad_accum": 2, "max_steps": 10, "max_seconds": 0, "combine_triplet_forward": True},
        "validation": {"per_rule": 8},
        "regularization": {"variance_weight": 0.01, "matryoshka_weight": 0.2, "matryoshka_dims": [16]},
        "curriculum": {"enabled": True, "mode": "staged"},
    }


def write_spec(tmp_path: Path, arms: dict, comparisons: list, **extra) -> Path:
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(base_config()))
    spec = {"name": "t", "base": str(base), "seeds": [1, 2], "arms": arms, "comparisons": comparisons, **extra}
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_checked_in_specs_are_valid_against_their_base(spec: Path) -> None:
    result = plan(spec, check_paths=False, write=False)
    assert result["runs"] and all(c["keys"] for c in result["comparisons"])


def test_plan_materializes_runs_and_shares_one_eval_set(tmp_path: Path) -> None:
    spec = write_spec(
        tmp_path,
        arms={"staged": {}, "mixed": {"curriculum.mode": "mixed"}, "long": {"model.max_length": 384}},
        comparisons=[{"arm": "mixed", "against": "staged"}, {"arm": "long", "against": "staged"}],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"counts": {"triplets/origin_vs_para/train": 40, "triplets/all/train": 40,
                                               "triplets/same_speaker_real/train": 20}}))
    result = plan(spec, manifest=manifest, check_paths=False, write=False)
    assert [r["run"] for r in result["runs"]][:2] == [run_name("t", "staged", 1), run_name("t", "staged", 2)]
    assert result["comparisons"][0]["keys"] == ["curriculum.mode"]
    evals = {r["arm"]: r["eval_config"] for r in result["runs"]}
    assert evals["staged"] == evals["mixed"] and evals["staged"].endswith("/eval.yaml")
    assert evals["long"].endswith("/eval__long.yaml")
    assert result["exposure"]["mixed"]["rules"]["origin_vs_para"]["repeats"] > 0
    assert "Repeats per distinct triplet" in render_plan(result)


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        ({"train.max_seconds": 3600}, "max_seconds"),
        ({"train.max_steps": 20}, "different budgets"),
        ({"data.heldout_speakers": 5}, "leak identities"),
        ({"validation.per_rule": 4}, "measured identically"),
        ({"train.max_step": 5}, "typo"),
        ({"curriculum.mode": "staged"}, "identical"),
        ({"curriculum.mode": "mixed", "regularization.matryoshka_weight": 0.0}, "declare `keys`"),
    ],
)
def test_plan_rejects_unfair_or_confounded_arms(tmp_path: Path, arm: dict, message: str) -> None:
    spec = write_spec(tmp_path, arms={"a": {}, "b": arm}, comparisons=[{"arm": "b", "against": "a"}])
    with pytest.raises(SpecError, match=message):
        plan(spec, check_paths=False, write=False)


def test_declared_keys_allow_a_multi_key_factor(tmp_path: Path) -> None:
    keys = ["regularization.matryoshka_weight", "regularization.variance_weight"]
    spec = write_spec(
        tmp_path,
        arms={"a": {}, "b": {"regularization.variance_weight": 0.0, "regularization.matryoshka_weight": 0.0}},
        comparisons=[{"arm": "b", "against": "a", "keys": keys}],
    )
    assert plan(spec, check_paths=False, write=False)["comparisons"][0]["keys"] == keys


def write_validation(root: Path, name: str, top1: float, provenance: float, gap: float) -> None:
    validation = root / name / "validation"
    validation.mkdir(parents=True)
    pd.DataFrame([
        {"rule": "same_speaker_topic_matched", "heldout": True, "n": 10, "pos": 0.5, "neg": 0.4, "gap": gap,
         "cgap": gap, "violation": 0.3},
        {"rule": "same_speaker_topic_matched", "heldout": False, "n": 10, "pos": 0.5, "neg": 0.1, "gap": 0.4,
         "cgap": 0.4, "violation": 0.0},
    ]).to_csv(validation / "per_rule.csv", index=False)
    retrieval = {"heldout speakers": {"top1": top1, "chance_top1": 0.05, "top5": top1 + 0.2},
                 "seen speakers": {"top1": 0.9, "chance_top1": 0.0125, "top5": 0.95}}
    probe = {"retrieval": retrieval, "provenance": {"accuracy": provenance, "chance": 0.4}}
    (validation / "retrieval_probe.json").write_text(json.dumps(probe))


def test_compare_pairs_by_seed_and_calls_verdicts(tmp_path: Path) -> None:
    spec = write_spec(tmp_path, arms={"a": {}, "b": {"curriculum.mode": "mixed"}},
                      comparisons=[{"arm": "b", "against": "a"}], seeds=[1, 2, 3])
    runs = tmp_path / "runs"
    for seed, noise in ((1, 0.00), (2, 0.01), (3, -0.01)):
        write_validation(runs, run_name("t", "b", seed), 0.20 + noise, 0.70 + noise, 0.10)
        if seed != 3:  # a's third seed never finished: b/s3 must not be paired
            write_validation(runs, run_name("t", "a", seed), 0.08 + noise, 0.90, 0.10 + noise)
    result = compare(spec, runs_root=runs, write=False)
    assert result["missing_runs"] == [run_name("t", "a", 3)]
    effects = result["comparisons"][0]["effects"]
    assert effects["heldout_top1_lift"]["n"] == 2 and effects["heldout_top1_lift"]["verdict"] == "better"
    assert effects["provenance_acc"]["verdict"] == "better"
    assert effects["gap:same_speaker_topic_matched"]["verdict"] == "within noise"
    assert result["arms"]["a"]["gap:same_speaker_topic_matched"]["mean"] == pytest.approx(0.105)  # held-out rows only
    assert result["kill_flagged"] == ["a"]
    assert "| b-vs-a |" in render_compare(result)


def test_interaction_is_the_difference_of_paired_deltas(tmp_path: Path) -> None:
    arms = {"s": {}, "m": {"curriculum.mode": "mixed"}, "s2": {"curriculum.pacing": "linear"},
            "m2": {"curriculum.mode": "mixed", "curriculum.pacing": "linear"}}
    comparisons = [{"name": "o1", "arm": "s", "against": "m"}, {"name": "o2", "arm": "s2", "against": "m2"}]
    spec = write_spec(tmp_path, arms=arms, comparisons=comparisons, interactions=[{"name": "ix", "of": ["o2", "o1"]}])
    runs = tmp_path / "runs"
    for seed in (1, 2):
        for arm, top1 in (("s", 0.3), ("m", 0.2), ("s2", 0.5), ("m2", 0.2)):
            write_validation(runs, run_name("t", arm, seed), top1, 0.7, 0.1)
    effect = compare(spec, runs_root=runs, write=False)["interactions"][0]["effects"]["heldout_top1"]
    assert effect["mean"] == pytest.approx(0.2) and effect["verdict"] == "better"


def test_measured_tiers_prefer_heldout_rows() -> None:
    table = pd.DataFrame([
        {"rule": "x", "heldout": False, "n": 10, "violation": 0.05},
        {"rule": "x", "heldout": True, "n": 10, "violation": 0.50},
        {"rule": "y", "heldout": False, "n": 5, "violation": 0.20},
        {"rule": "z", "heldout": False, "n": 3, "violation": 0.00},
    ])
    assert measured_tiers(table) == {"x": "hard", "y": "mid", "z": "easy"}
