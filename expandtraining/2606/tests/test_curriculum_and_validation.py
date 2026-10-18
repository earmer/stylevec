from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_triplets import build_genshin_triplets
from curriculum import curriculum_order, rule_tier
from import_paratarget import paratarget_job_split
from records import SampleRecord, TripletRecord, stable_id
from report import per_rule_table, speaker_retrieval
from validate import select_validation_triplets


def genshin(speaker: str, order: int, text: str, split: str = "train") -> SampleRecord:
    return SampleRecord(
        sample_id=stable_id("t", speaker, order), text=text, style_id=speaker, style_type="speaker",
        domain="genshin", content_id="c", source_id=str(order), language="zh", split=split, quality=1.0,
        meta={"order": order, "db_id": order, "scene_neighbors": [["Z", "邻居的一句话好长啊好长", 1]], "length": len(text), "para_text": "neutral " + text, "heldout": False},
    )


def test_genshin_positive_is_never_adjacent() -> None:
    config = {"data": {"genshin_min_id_gap": 5}, "weights": {"genshin_real": 1.0},
              "margins": {"medium": 0.1, "hard": 0.2}}
    samples = [genshin(s, i, f"{s}说第{i}句话然后{i*7}") for s in "AB" for i in range(20)]
    rows = build_genshin_triplets(config, samples, "train")
    assert {r.rule for r in rows} == {"same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"}
    by_text = {s.text: s.meta["order"] for s in samples}
    for r in rows:
        assert abs(by_text[r.anchor_text] - by_text[r.positive_text]) >= 5
    topic = [r for r in rows if r.rule == "same_speaker_topic_matched"]
    assert topic and all(r.negative_text == "邻居的一句话好长啊好长" for r in topic)


def test_curriculum_hard_share_rises_and_mixed_is_flat() -> None:
    rows = [TripletRecord("a", "p", "n", rule, 1.0, 0.1)
            for rule in ("same_speaker_real", "origin_vs_para", "same_speaker_topic_matched") for _ in range(50)]
    cfg = {"curriculum": {"mode": "staged"}}
    order = curriculum_order(rows, cfg, seed=0, total=3000)
    hard = lambda chunk: np.mean([rule_tier(r.rule) == "hard" for r in chunk])
    assert hard(order[:300]) < 0.15 < 0.4 < hard(order[-300:])
    assert hard(order[-300:]) < 0.8  # never all-hard
    mixed = curriculum_order(rows, {"curriculum": {"mode": "mixed"}}, seed=0, total=3000)
    assert abs(hard(mixed[:300]) - hard(mixed[-300:])) < 0.15


def test_job_split_keeps_job_together_and_heldout_in_val() -> None:
    assert paratarget_job_split("j", "A", "H", {"H"}) == "val"
    assert paratarget_job_split("j", "A", "B", set()) == paratarget_job_split("j", "A", "B", set())


def test_select_validation_triplets_fixed_n_per_rule() -> None:
    rows = [TripletRecord(f"a{i}", "p", "n", r, 1.0, 0.1, split="val") for r in ("x", "y") for i in range(30)]
    chosen = select_validation_triplets(rows, 8)
    assert sorted({t.rule for t in chosen}) == ["x", "y"]
    assert sum(t.rule == "x" for t in chosen) == 8


def test_per_rule_table_and_retrieval_on_synthetic_embeddings() -> None:
    rng = np.random.default_rng(0)
    centers = {s: rng.normal(size=8) for s in "ABCDEF"}
    recs, vecs = [], []
    for i in range(36):
        sp = "ABCDEF"[i % 6]
        for role, spk in (("anchor", sp), ("positive", sp), ("negative", "ABCDEF"[(i + 1) % 6])):
            recs.append({"triplet_id": f"t{i}", "role": role, "rule": "same_speaker_real", "heldout": True,
                         "speaker": spk, "text_preview": f"{role}{i}"})
            vecs.append(centers[spk] + 0.1 * rng.normal(size=8))
    rows, emb = pd.DataFrame(recs), np.stack(vecs)
    table = per_rule_table(rows, emb)
    assert table.iloc[0]["violation"] == 0.0 and table.iloc[0]["n"] == 36
    stats = speaker_retrieval(rows, emb, heldout=True)
    assert stats["top1"] > 0.9 and stats["chance_top1"] < 0.2


def test_cross_scene_retrieval_drops_same_scene_lines_from_own_centroid() -> None:
    rng = np.random.default_rng(0)
    recs, vecs = [], []
    for sp_index, sp in enumerate("ABCD"):
        center = rng.normal(size=8)
        for j in range(4):  # two scenes per speaker, two lines each
            recs.append({"triplet_id": f"{sp}{j}", "role": "anchor", "rule": "same_speaker_real", "heldout": True,
                         "speaker": sp, "text_preview": f"{sp}{j}", "db_id": sp_index * 10_000 + (j // 2) * 1000 + j})
            vecs.append(center + 0.1 * rng.normal(size=8))
    rows, emb = pd.DataFrame(recs), np.stack(vecs)
    plain = speaker_retrieval(rows, emb, heldout=True)
    cross = speaker_retrieval(rows, emb, heldout=True, min_gap=150)
    assert plain["n_scored"] == cross["n_scored"] == 16 and cross["top1"] > 0.9
    rows["db_id"] = [i for i in range(16)]  # every line in one scene: nothing left to compare against
    assert "top1" not in speaker_retrieval(rows, emb, heldout=True, min_gap=150)
    assert "top1" not in speaker_retrieval(rows.drop(columns="db_id"), emb, heldout=True, min_gap=150)


def test_diagnostics_split_topic_gap_and_whiten() -> None:
    from report import diagnostics, whitened

    rng = np.random.default_rng(1)
    recs, vecs = [], []
    for i in range(8):
        for role, spk in (("anchor", "A"), ("positive", "A"), ("negative", "派蒙" if i < 5 else "C")):
            recs.append({"triplet_id": f"t{i}", "role": role, "rule": "same_speaker_topic_matched",
                         "heldout": True, "speaker": spk, "text_preview": f"{role}{i}"})
            vecs.append(rng.normal(size=6))
    for i in range(20):
        recs.append({"triplet_id": f"s{i}", "role": "anchor", "rule": "same_speaker_real", "heldout": False,
                     "speaker": "S", "text_preview": f"s{i}"})
        vecs.append(rng.normal(size=6))
    rows, emb = pd.DataFrame(recs), np.stack(vecs)
    out = diagnostics(rows, emb, frozenset({"派蒙"}))
    assert out["n_topic_hub_neg"] == 5 and out["n_topic_other_neg"] == 3
    white = whitened(rows, emb)
    assert white.shape == (len(rows), 6) and np.allclose(np.linalg.norm(white, axis=1), 1.0)


def test_validation_selection_interleaves_buckets_and_speaker_of_covers_paratarget() -> None:
    from validate import speaker_of

    rows = [TripletRecord(f"{r}{i}", "p", "n", r, 1.0, 0.1, split="val") for r in ("x", "y", "z") for i in range(5)]
    chosen = select_validation_triplets(rows, 4)
    assert {t.rule for t in chosen[:3]} == {"x", "y", "z"} and len(chosen) == 12
    meta = {"target_speaker": "甲", "source_speaker": "乙"}
    assert speaker_of(meta, "anchor") == "甲" and speaker_of(meta, "negative") == "乙"
    assert speaker_of({**meta, "negative_target": "丙"}, "negative") == "丙"
    assert speaker_of({"feature": "f"}, "anchor") == ""
