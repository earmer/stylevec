from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_triplets import build_bookraw_triplets, build_length_course_triplets, build_paratarget_triplets, cap_by_rule
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
