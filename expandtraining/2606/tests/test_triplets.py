from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from losses import embedding_spread_loss, weighted_triplet_margin_loss
from records import SampleRecord, TripletRecord, normalize_text, stable_id
from build_triplets import build_bookraw_triplets, cap_by_rule
from train import balanced_triplet_order


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
