from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_triplets import length_tolerance, pick_similar_length_negative, build_length_negative_index, build_bookraw_triplets, build_genshin_triplets, build_length_course_triplets, build_paratarget_triplets, cap_by_rule
from losses import embedding_spread_loss, matryoshka_triplet_margin_loss, weighted_triplet_margin_loss
from probes import length_leakage_metrics
from records import SampleRecord, TripletRecord, normalize_text, stable_id
from train import balanced_triplet_order, dataloader_worker_kwargs


def test_normalize_text_collapses_spaces() -> None:
    assert normalize_text("  a\n b\tc  ") == "a b c"


def test_triplet_record_has_stable_id() -> None:
    row = TripletRecord(
        anchor_text="a",
        positive_text="b",
        negative_text="c",
        rule="unit",
        weight=1.0,
        margin=0.1,
    )
    assert row.triplet_id == row.triplet_id
    assert row.to_json_dict()["triplet_id"] == row.triplet_id


def test_weighted_triplet_margin_loss_prefers_positive() -> None:
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [0.9, 0.1]])
    negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    margins = torch.tensor([0.2, 0.2])
    weights = torch.tensor([1.0, 1.0])
    loss, stats = weighted_triplet_margin_loss(anchor, positive, negative, margins, weights)
    assert loss.item() > 0.0
    assert stats["violations"].shape == (2,)


def test_cap_by_rule_keeps_rule_coverage() -> None:
    rows = [
        TripletRecord("a1", "p1", "n1", "z_rule", 1.0, 0.1),
        TripletRecord("a2", "p2", "n2", "z_rule", 1.0, 0.1),
        TripletRecord("a3", "p3", "n3", "a_rule", 1.0, 0.1),
    ]
    capped = cap_by_rule(rows, 2)
    assert {row.rule for row in capped} == {"a_rule", "z_rule"}


def test_embedding_spread_loss_reports_rank_stats() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(12, 8), dim=-1)
    loss, stats = embedding_spread_loss(
        embeddings,
        min_dim_std=0.03,
        variance_weight=0.01,
        covariance_weight=0.005,
    )
    assert loss.item() >= 0.0
    assert stats["top_eigen_ratio"].item() > 0.0
    assert stats["effective_rank"].item() > 0.0


def test_matryoshka_triplet_margin_loss_reports_prefix_stats() -> None:
    anchor = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    positive = torch.nn.functional.normalize(anchor + 0.05 * torch.randn(6, 8), dim=-1)
    negative = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    margins = torch.full((6,), 0.1)
    weights = torch.ones(6)
    loss, stats = matryoshka_triplet_margin_loss(anchor, positive, negative, margins, weights, [2, 4])
    assert loss.item() >= 0.0
    assert "loss_dim_2" in stats
    assert "violation_dim_4" in stats


def test_length_leakage_probe_detects_length_axis() -> None:
    lengths = np.arange(20, 40)
    embeddings = np.stack([np.log1p(lengths), np.linspace(0.0, 1.0, len(lengths))], axis=1)
    metrics = length_leakage_metrics(embeddings, lengths, lengths // 4)
    assert metrics["length_count"] == float(len(lengths))
    assert metrics["length_pc1_abs_corr"] > 0.95


def test_balanced_triplet_order_interleaves_rules() -> None:
    rows = [
        TripletRecord(f"a{index}", "p", "n", "a_rule", 1.0, 0.1)
        for index in range(3)
    ] + [
        TripletRecord(f"b{index}", "p", "n", "b_rule", 1.0, 0.1)
        for index in range(3)
    ]
    ordered = balanced_triplet_order(rows, seed=1)
    assert len(ordered) == len(rows)
    assert [row.rule for row in ordered[:4]] == ["a_rule", "b_rule", "a_rule", "b_rule"]


def test_dataloader_worker_kwargs_disable_persistent_workers_when_epoch_rotates() -> None:
    kwargs = dataloader_worker_kwargs(
        {
            "num_workers": 4,
            "pin_memory": True,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "rule_balance_rotate_each_epoch": True,
        }
    )
    assert kwargs["num_workers"] == 4
    assert kwargs["pin_memory"] is True
    assert kwargs["prefetch_factor"] == 4
    assert kwargs["persistent_workers"] is False


def style_sample(domain: str, style: str, index: int, text: str, split: str = "train") -> SampleRecord:
    return SampleRecord(
        sample_id=stable_id("unit", domain, style, index),
        text=text,
        style_id=style,
        style_type="speaker",
        domain=domain,
        content_id=stable_id("unit_content", domain, style, index),
        source_id=f"{domain}:{style}:{index}",
        language="zh",
        split=split,
        quality=1.0,
        meta={"speaker": style, "length": len(text), "length_bucket": len(text) // 20},
    )


def test_length_course_triplets_use_different_lengths_same_source() -> None:
    config = {
        "data": {
            "length_course_enabled": True,
            "length_course_triplets_per_domain": 4,
            "length_course_parts": 2,
            "length_course_max_chars": 500,
            "max_triplets": 100,
        },
        "validation": {"max_triplets": 20},
        "weights": {"length_course": 0.08},
        "margins": {"easy": 0.08},
    }
    samples = [
        style_sample("genshin", "alice", 0, "short alice line"),
        style_sample("genshin", "alice", 1, "longer alice line " + "x" * 30),
        style_sample("genshin", "bob", 0, "short bob line"),
        style_sample("genshin", "bob", 1, "longer bob line " + "y" * 30),
    ]
    rows = build_length_course_triplets(config, samples, "train")
    assert rows
    assert {row.rule for row in rows} == {"length_course_genshin"}
    assert any(len(row.positive_text) > len(row.anchor_text) for row in rows)
    assert all(row.weight == 0.08 for row in rows)


def test_length_course_parts_one_disables_concatenation() -> None:
    config = {
        "data": {
            "length_course_enabled": True,
            "length_course_triplets_per_domain": 4,
            "length_course_parts": 1,
            "length_course_max_chars": 500,
            "max_triplets": 100,
        },
        "validation": {"max_triplets": 20},
        "weights": {"length_course": 0.08},
        "margins": {"easy": 0.08},
    }
    samples = [
        style_sample("genshin", "alice", 0, "short alice line"),
        style_sample("genshin", "alice", 1, "longer alice line " + "x" * 30),
        style_sample("genshin", "bob", 0, "short bob line"),
        style_sample("genshin", "bob", 1, "longer bob line " + "y" * 30),
    ]
    assert build_length_course_triplets(config, samples, "train") == []


def test_paratarget_original_push_rule_is_low_weight() -> None:
    config = {
        "data": {"paratarget_original_push_enabled": True},
        "weights": {"paratarget": 0.75, "paratarget_original_push": 0.08},
        "margins": {"easy": 0.08, "hard": 0.18, "adversarial": 0.22},
    }
    genshin = [
        style_sample("genshin", "source", 0, "source real one"),
        style_sample("genshin", "source", 1, "source real two"),
        style_sample("genshin", "target", 0, "target real one"),
        style_sample("genshin", "target", 1, "target real two"),
    ]
    generated = SampleRecord(
        sample_id="generated",
        text="target style generated",
        style_id="target",
        style_type="generated_target",
        domain="paratarget",
        content_id="content",
        source_id="job:0",
        language="zh",
        split="train",
        quality=0.75,
        meta={"source_speaker": "source", "target_speaker": "target", "source_text": "source original"},
    )
    rows = build_paratarget_triplets(config, genshin, [generated], "train")
    original_push = [row for row in rows if row.rule == "paratarget_original_push"]
    assert len(original_push) == 1
    assert original_push[0].negative_text == "source original"
    assert original_push[0].weight == 0.08


def bookraw_sample(author: str, work: str, index: int, split: str = "train") -> SampleRecord:
    text = f"{author} {work} sample passage {index} " + ("x" * 180)
    return SampleRecord(
        sample_id=stable_id("unit_bookraw", author, work, index),
        text=text,
        style_id=author,
        style_type="author",
        domain="bookraw",
        content_id=stable_id("unit_bookraw_work", author, work),
        source_id=f"{author}:{work}:{index}",
        language="zh",
        split=split,
        quality=0.85,
        meta={
            "author": author,
            "work": work,
            "lang": "zh",
            "length": len(text),
            "length_bucket": len(text) // 200,
        },
    )


def test_bookraw_triplets_cover_three_author_rules() -> None:
    config = {
        "data": {"max_triplets": 100, "bookraw_triplets_per_rule": 10},
        "validation": {"max_triplets": 50},
        "weights": {
            "bookraw_same_work": 0.35,
            "bookraw_cross_work": 0.45,
            "bookraw_author_negative": 0.40,
        },
        "margins": {"medium": 0.12, "hard": 0.18},
    }
    samples = [
        bookraw_sample("author_a", "work_1", 0),
        bookraw_sample("author_a", "work_1", 1),
        bookraw_sample("author_a", "work_2", 0),
        bookraw_sample("author_a", "work_2", 1),
        bookraw_sample("author_b", "work_3", 0),
        bookraw_sample("author_b", "work_3", 1),
        bookraw_sample("author_b", "work_4", 0),
        bookraw_sample("author_b", "work_4", 1),
    ]
    rows = build_bookraw_triplets(config, samples, "train")
    rules = {row.rule for row in rows}
    assert "bookraw_same_work" in rules
    assert "bookraw_same_author_cross_work" in rules
    assert "bookraw_same_language_author_negative" in rules


def genshin_line(speaker: str, index: int, text: str, para: str | None = None) -> SampleRecord:
    meta = {"speaker": speaker, "db_id": index * 1000, "length": len(text), "length_bucket": len(text) // 20}
    if para is not None:
        meta["para_text"] = para
    return SampleRecord(
        sample_id=stable_id("unit_genshin", speaker, index),
        text=text,
        style_id=speaker,
        style_type="speaker",
        domain="genshin",
        content_id=stable_id("unit_genshin_content", speaker, index),
        source_id=f"genshin:{speaker}:{index}",
        language="zh",
        split="train",
        quality=1.0,
        meta=meta,
    )


def test_origin_vs_para_drops_only_identical_paraphrases() -> None:
    config = {
        "data": {"genshin_min_id_gap": 150, "genshin_candidates": 24},
        "weights": {"genshin_real": 1.0},
        "margins": {"medium": 0.12, "hard": 0.18},
    }
    lines = {
        "identical": ("所以只要找到正确的材料，就能唤醒它。", " 所以只要找到正确的材料，就能唤醒它。"),
        "small_edit": ("请问您会如何评价这场比赛呢？", "请问您如何评价这场比赛？"),
        "punctuation": ("事不宜迟！得赶紧动起来，我们现在就出发吧！", "事不宜迟…得赶紧动起来 我们现在就出发吧。"),
        "paraphrase": ("难道，是哪位路过的稻妻客商顺手买走了书，然后带回了稻妻…", "莫非是某位途经的稻妻客商顺便购买了书，然后带回了稻妻…"),
    }
    samples = [genshin_line("a", i, origin, para) for i, (origin, para) in enumerate(lines.values())]
    samples += [genshin_line("b", i, f"另一个角色的第{i}句台词，长度差不多。") for i in range(3)]
    rows = [row for row in build_genshin_triplets(config, samples, "train") if row.rule == "origin_vs_para"]
    kept = [lines[key][1] for key in ("small_edit", "punctuation", "paraphrase")]
    assert sorted(row.negative_text for row in rows) == sorted(kept)


def test_target_hard_negative_cycles_through_other_target_lines() -> None:
    config = {
        "data": {"paratarget_original_push_enabled": False, "same_content_diff_style_enabled": False},
        "weights": {"paratarget": 0.75},
        "margins": {"easy": 0.08, "hard": 0.18, "adversarial": 0.22},
    }
    genshin = [style_sample("genshin", name, i, f"{name} real {i}") for name in ("src", "t1", "t2") for i in range(2)]

    def generated(target: str, index: int, text: str) -> SampleRecord:
        return SampleRecord(
            sample_id=stable_id("unit_generated", target, index),
            text=text,
            style_id=target,
            style_type="generated_target",
            domain="paratarget",
            content_id=stable_id("unit_generated_content", target, index),
            source_id=f"job:{target}:{index}",
            language="zh",
            split="train",
            quality=0.75,
            meta={"source_speaker": "src", "target_speaker": target, "source_text": f"src original {index}"},
        )

    # t2's first line repeats a t1 anchor, so that triplet must advance past it.
    t1 = [generated("t1", i, f"t1 generated {i}") for i in range(4)]
    t2 = [generated("t2", 0, "t1 generated 0"), generated("t2", 1, "t2 generated 1"), generated("t2", 2, "t2 generated 2")]
    rows = build_paratarget_triplets(config, genshin, t1 + t2, "train")
    hard = [row for row in rows if row.rule == "target_hard_negative"]
    assert len(hard) == len(t1) + len(t2)
    to_t2 = [row.negative_text for row in hard if row.meta["negative_target"] == "t2"]
    # visit 0 would be t2[0] == its own anchor, so it advances to t2[1]
    assert to_t2 == ["t2 generated 1", "t2 generated 1", "t2 generated 2", "t1 generated 0"]
    to_t1 = [row.negative_text for row in hard if row.meta["negative_target"] == "t1"]
    assert to_t1 == ["t1 generated 1", "t1 generated 1", "t1 generated 2"]  # t2[0] repeats t1[0]: advance
    assert all(row.negative_text not in (row.anchor_text, row.positive_text) for row in hard)


def test_length_course_negative_matches_anchor_length_by_default() -> None:
    config = {
        "data": {"length_course_enabled": True, "length_course_triplets_per_domain": 4,
                 "length_course_parts": 2, "length_course_max_chars": 500, "max_triplets": 100},
        "validation": {"max_triplets": 20},
        "weights": {"length_course": 0.08},
        "margins": {"easy": 0.08},
    }
    samples = [
        style_sample("genshin", "alice", 0, "short alice"),
        style_sample("genshin", "alice", 1, "alice again " + "x" * 30),
        style_sample("genshin", "bob", 0, "short bobby"),
        style_sample("genshin", "bob", 1, "longer bob line " + "y" * 60),
    ]
    anchored = build_length_course_triplets(config, samples, "train")
    assert anchored
    for row in anchored:
        best = min(abs(len(s.text) - row.meta["anchor_length"]) for s in samples if s.style_id != row.meta["source"])
        assert abs(row.meta["negative_length"] - row.meta["anchor_length"]) <= best + length_tolerance(row.meta["anchor_length"])
    config["data"]["length_course_negative_length"] = "positive"
    legacy = build_length_course_triplets(config, samples, "train")
    assert [row.negative_text for row in legacy] != [row.negative_text for row in anchored]
    config["data"]["length_course_negative_length"] = "nope"
    with pytest.raises(ValueError):
        build_length_course_triplets(config, samples, "train")


def test_topic_matched_variants_widen_train_only() -> None:
    def scene_line(speaker: str, index: int, text: str, neighbors: list[list]) -> SampleRecord:
        sample = genshin_line(speaker, index, text)
        sample.meta["scene_neighbors"] = neighbors
        return sample

    config = {
        "data": {"genshin_min_id_gap": 150, "genshin_candidates": 24},
        "weights": {"genshin_real": 1.0},
        "margins": {"medium": 0.12, "hard": 0.18},
    }
    neighbors = [[f"other{i}", f"同一场景里别人说的第{i}句话。", i + 1] for i in range(4)]
    samples = [scene_line("a", i, f"角色甲在第{i}个场景说的一句台词。", neighbors) for i in range(5)]
    samples += [genshin_line("b", i, f"另一个角色的第{i}句台词，长度差不多。") for i in range(3)]

    def topic(rows: list[TripletRecord]) -> list[TripletRecord]:
        return [row for row in rows if row.rule == "same_speaker_topic_matched"]

    single = topic(build_genshin_triplets(config, samples, "train"))
    config["data"]["genshin_topic_matched_variants"] = 3
    widened = topic(build_genshin_triplets(config, samples, "train"))
    assert len(widened) == 3 * len(single)
    by_anchor: dict[str, list[TripletRecord]] = {}
    for row in widened:
        by_anchor.setdefault(row.anchor_text, []).append(row)
    for row in single:  # variant 0 is the single-variant triplet, unchanged
        assert (row.positive_text, row.negative_text) == (
            by_anchor[row.anchor_text][0].positive_text, by_anchor[row.anchor_text][0].negative_text)
    for rows in by_anchor.values():
        assert len({row.positive_text for row in rows}) == 3 and len({row.negative_text for row in rows}) == 3

    val = [SampleRecord(**{**s.__dict__, "split": "val"}) for s in samples]
    assert len(topic(build_genshin_triplets(config, val, "val"))) == len(single)


def test_length_probe_separates_group_level_from_within_group_length() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 40)
    lengths = (20 + 30 * groups + rng.integers(0, 10, size=groups.size)).astype(float)
    # The embedding knows only the group (one-hot plus noise), not the length inside a group.
    embeddings = np.eye(8)[groups] + 0.01 * rng.normal(size=(groups.size, 8))
    metrics = length_leakage_metrics(embeddings, lengths, lengths // 20, groups=groups.tolist(), speakers=groups.tolist())
    assert metrics["length_linear_r2"] > 0.8
    assert metrics["length_r2_within_group"] < 0.2 and metrics["length_r2_within_speaker"] < 0.2
    # An embedding that carries length itself is caught within groups too.
    leaky = np.concatenate([embeddings, np.log1p(lengths)[:, None]], axis=1)
    assert length_leakage_metrics(leaky, lengths, lengths // 20, groups=groups.tolist())["length_r2_within_group"] > 0.8


def test_scene_neighbors_prefer_non_hub_speakers() -> None:
    from import_genshin import attach_scene_neighbors

    anchor = genshin_line("a", 0, "角色甲的一句台词。")
    anchor.meta["db_id"] = 100
    scene = [(101, "派蒙", "派蒙的台词。"), (102, "旅行者", "旅行者的台词。"), (105, "c", "角色丙的台词。"),
             (100, "a", "角色甲的一句台词。")]
    attach_scene_neighbors([anchor], scene, set(), window=12, limit=1, hubs=frozenset({"派蒙", "旅行者"}))
    assert [n[0] for n in anchor.meta["scene_neighbors"]] == ["c", "派蒙"]  # nearest non-hub, then nearest hub
    attach_scene_neighbors([anchor], scene, set(), window=12, limit=1)
    assert [n[0] for n in anchor.meta["scene_neighbors"]] == ["派蒙"]  # no hubs: plain nearest


def test_topic_matched_negative_prefers_non_hub_and_real_positive_matches_length() -> None:
    config = {
        "data": {"genshin_min_id_gap": 150, "genshin_candidates": 24},
        "weights": {"genshin_real": 1.0},
        "margins": {"medium": 0.12, "hard": 0.18},
    }
    texts = ["短句。", "这是一句中等长度的台词。", "这是一句明显要长得多的台词，后面还接着很多很多的字。"]
    samples = [genshin_line("a", i, text) for i, text in enumerate(texts * 2)]
    samples += [genshin_line("b", i, f"另一个角色的第{i}句台词，长度差不多。") for i in range(6)]
    for sample in samples:
        sample.meta["scene_neighbors"] = [["c", "角色丙在同一场景说的话。", 3], ["派蒙", "派蒙在同一场景说的话。", 1]]
    rows = build_genshin_triplets(config, samples, "train")
    topic = [row for row in rows if row.rule == "same_speaker_topic_matched"]
    assert topic and all(row.meta["negative_speaker"] == "c" for row in topic)
    for sample in samples:  # only hubs in the scene: still used, so no anchor is lost
        sample.meta["scene_neighbors"] = [["派蒙", "派蒙在同一场景说的话。", 1]]
    assert len([r for r in build_genshin_triplets(config, samples, "train") if r.rule == "same_speaker_topic_matched"]) == len(topic)
    real = [row for row in rows if row.rule == "same_speaker_real" and row.meta["speaker"] == "a"]
    # Every text of "a" occurs twice in different scenes, so the length-matched positive is its twin.
    assert real and all(len(row.positive_text) == len(row.anchor_text) for row in real)


def test_length_negatives_cycle_within_tolerance_and_anchors_spread_over_sources() -> None:
    lines = [genshin_line(f"s{k}", i, "台" * (10 + i % 5)) for k in range(6) for i in range(20)]
    index = build_length_negative_index(lines)
    picks = [pick_similar_length_negative(index, "s0", 12, turn) for turn in range(16)]
    assert all(p is not None and p.style_id != "s0" and abs(len(p.text) - 12) <= 2 for p in picks)
    assert len({p.style_id for p in picks}) >= 3  # cycles over speakers instead of one row per length
    assert pick_similar_length_negative(index, "s0", 12, 0) is not None
    config = {
        "data": {"length_course_enabled": True, "length_course_triplets_per_domain": 12, "max_triplets": 1000},
        "weights": {"length_course": 0.08},
        "margins": {"easy": 0.08},
    }
    rows = build_length_course_triplets(config, lines, "train")
    assert len(rows) == 12 and len({row.meta["source"] for row in rows}) == 6


def test_msynthstel_near_duplicate_features() -> None:
    from import_msynthstel import near_duplicate_feature

    assert near_duplicate_feature("Usage of Words Indicating Affective Processes", ["Usage of Words Indicating Affective Process"])
    assert near_duplicate_feature("Usage of Present-Focused Tense and Words", ["Usage of Present Tense and Present-Focused Words"])
    assert not near_duplicate_feature("Usage of Words Indicating Cognitive Processes", ["Usage of Words Indicating Affective Processes"])
    assert not near_duplicate_feature("Usage of Polite Tone", ["Usage of Formal Tone"])


def test_paratarget_interleave_round_robin() -> None:
    from import_paratarget import interleave

    assert interleave([[1, 2, 3], [4], [5, 6]]) == [1, 4, 5, 2, 6, 3]
