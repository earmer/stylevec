"""Tests for import_authorcorpora.py and the same_author_real triplet rule.

All fixtures are synthetic: no real corpus files are read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from import_authorcorpora import (
    author_split,
    clean_text,
    is_bot_like,
    load_authorcorpora_samples,
)
from build_triplets import build_authorcorpora_triplets
from records import SampleRecord, stable_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(parquet_path: str | None = None, enabled: bool = True) -> dict:
    base: dict = {
        "data": {
            "min_chars": 4,
            "max_chars": 500,
            "max_triplets": 100,
            "authorcorpora_enabled": enabled,
            "authorcorpora_min_texts": 3,
            "authorcorpora_texts_per_author": 10,
        },
        "validation": {"max_triplets": 50},
        "weights": {},
        "margins": {"easy": 0.08, "medium": 0.12, "hard": 0.18, "adversarial": 0.22},
    }
    if parquet_path is not None:
        base["paths"] = {"authorcorpora_parquet": parquet_path}
    else:
        base["paths"] = {}
    return base


def _make_sample(style_id: str, text: str, split: str = "train") -> SampleRecord:
    sid = stable_id("authorcorpora", style_id, text)
    return SampleRecord(
        sample_id=sid,
        text=text,
        style_id=style_id,
        style_type="author_id",
        domain="authorcorpora",
        content_id=stable_id(text),
        source_id=sid,
        language="zh",
        split=split,
        quality=1.0,
        meta={},
    )


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a synthetic unified.parquet and return its path."""
    df = pd.DataFrame(rows, columns=["source", "author_id", "text", "lang", "n_chars", "created", "meta", "split"])
    path = tmp_path / "unified.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_strips_html() -> None:
    assert "<b>" not in clean_text("<b>bold</b> text")
    assert "bold" in clean_text("<b>bold</b> text")


def test_clean_text_strips_url() -> None:
    cleaned = clean_text("visit https://example.com/path?q=1 for details")
    assert "https" not in cleaned
    assert "details" in cleaned


def test_clean_text_strips_at_mention() -> None:
    cleaned = clean_text("@user1 hello @user2")
    assert "@user1" not in cleaned
    assert "@user2" not in cleaned
    assert "hello" in cleaned


def test_clean_text_strips_repost_prefix() -> None:
    cleaned = clean_text("转发微博 some content")
    assert "转发微博" not in cleaned
    assert "some content" in cleaned


def test_clean_text_collapses_whitespace() -> None:
    assert "  " not in clean_text("a   b\n\tc")


def test_clean_text_unescape_html_entities() -> None:
    assert "&amp;" not in clean_text("a &amp; b")
    assert "&" in clean_text("a &amp; b")


# ---------------------------------------------------------------------------
# is_bot_like
# ---------------------------------------------------------------------------

def test_is_bot_like_empty() -> None:
    assert is_bot_like("") is True
    assert is_bot_like("   ") is True


def test_is_bot_like_highly_repetitive() -> None:
    # Only 2 unique chars in 20-char string → highly repetitive
    assert is_bot_like("aaaaaaaaabbbbbbbbbbb") is True


def test_is_bot_like_normal_text() -> None:
    assert is_bot_like("这是一段正常的中文文本，内容比较丰富。") is False


def test_is_bot_like_short_text_passes() -> None:
    # ≤ 6 chars skips the diversity check
    assert is_bot_like("aaa") is False


# ---------------------------------------------------------------------------
# author_split
# ---------------------------------------------------------------------------

def test_author_split_deterministic() -> None:
    a = author_split("weibo:12345")
    b = author_split("weibo:12345")
    assert a == b


def test_author_split_returns_valid_splits() -> None:
    for i in range(50):
        split = author_split(f"test_author_{i}")
        assert split in ("train", "val", "heldout"), f"unexpected split {split!r}"


def test_author_split_covers_all_classes() -> None:
    # With 200 authors we expect all three splits to appear
    splits = {author_split(f"author_{i}") for i in range(200)}
    assert splits == {"train", "val", "heldout"}


def test_author_split_different_ids_can_differ() -> None:
    results = [author_split(f"a_{i}") for i in range(100)]
    assert len(set(results)) > 1


# ---------------------------------------------------------------------------
# load_authorcorpora_samples: disabled by default
# ---------------------------------------------------------------------------

def test_load_returns_empty_when_disabled(tmp_path: Path) -> None:
    config = _make_config(enabled=False)
    samples, filters = load_authorcorpora_samples(config)
    assert samples == []
    assert filters == {}


# ---------------------------------------------------------------------------
# load_authorcorpora_samples: synthetic parquet
# ---------------------------------------------------------------------------

def _synthetic_rows(author_ids: list[str], texts_per_author: int, split: str = "train") -> list[dict]:
    rows = []
    for author_id in author_ids:
        for i in range(texts_per_author):
            rows.append({
                "source":    "weibo",
                "author_id": author_id,
                "text":      f"{author_id} 文本内容 {i} 这是一段足够长的测试文本",
                "lang":      "zh",
                "n_chars":   30,
                "created":   "2024-01-01",
                "meta":      "{}",
                "split":     split,
            })
    return rows


def test_load_samples_basic(tmp_path: Path) -> None:
    rows = _synthetic_rows(["weibo:a", "weibo:b", "weibo:c"], texts_per_author=5)
    path = _make_parquet(tmp_path, rows)
    config = _make_config(str(path), enabled=True)
    samples, filters = load_authorcorpora_samples(config)
    assert len(samples) > 0
    assert all(s.domain == "authorcorpora" for s in samples)
    assert all(s.split == "train" for s in samples)


def test_load_samples_style_id_is_namespaced(tmp_path: Path) -> None:
    rows = _synthetic_rows(["weibo:user42"], texts_per_author=5)
    path = _make_parquet(tmp_path, rows)
    config = _make_config(str(path), enabled=True)
    samples, _ = load_authorcorpora_samples(config)
    assert all(s.style_id == "weibo:user42" for s in samples)


def test_load_samples_respects_min_texts(tmp_path: Path) -> None:
    # Author "weibo:few" has only 2 texts < min_texts=3 → dropped
    rows = _synthetic_rows(["weibo:many"], texts_per_author=5)
    rows += _synthetic_rows(["weibo:few"], texts_per_author=2)
    path = _make_parquet(tmp_path, rows)
    config = _make_config(str(path), enabled=True)
    samples, filters = load_authorcorpora_samples(config)
    assert not any(s.style_id == "weibo:few" for s in samples)
    assert filters.get("below_min_texts", 0) >= 2


def test_load_samples_deduplicates_exact_texts(tmp_path: Path) -> None:
    # Two authors share an exact text → second occurrence is deduplicated
    repeated = "这是完全一样的文字内容不会被保留两次"
    rows = [
        {"source": "weibo", "author_id": "weibo:a", "text": repeated,
         "lang": "zh", "n_chars": len(repeated), "created": "", "meta": "{}", "split": "train"},
        {"source": "weibo", "author_id": "weibo:a", "text": f"{repeated} text1",
         "lang": "zh", "n_chars": 30, "created": "", "meta": "{}", "split": "train"},
        {"source": "weibo", "author_id": "weibo:a", "text": "独特文本内容三",
         "lang": "zh", "n_chars": 30, "created": "", "meta": "{}", "split": "train"},
        {"source": "weibo", "author_id": "weibo:b", "text": repeated,  # exact dup
         "lang": "zh", "n_chars": len(repeated), "created": "", "meta": "{}", "split": "train"},
        {"source": "weibo", "author_id": "weibo:b", "text": "独特文本内容四",
         "lang": "zh", "n_chars": 30, "created": "", "meta": "{}", "split": "train"},
        {"source": "weibo", "author_id": "weibo:b", "text": "独特文本内容五",
         "lang": "zh", "n_chars": 30, "created": "", "meta": "{}", "split": "train"},
    ]
    path = _make_parquet(tmp_path, rows)
    config = _make_config(str(path), enabled=True)
    samples, filters = load_authorcorpora_samples(config)
    texts = [s.text for s in samples]
    assert texts.count(repeated) <= 1
    assert filters.get("duplicate_text", 0) >= 1


def test_load_samples_missing_parquet_raises(tmp_path: Path) -> None:
    config = _make_config(str(tmp_path / "does_not_exist.parquet"), enabled=True)
    with pytest.raises(FileNotFoundError, match="authorcorpora parquet"):
        load_authorcorpora_samples(config)


def test_load_samples_sample_ids_stable(tmp_path: Path) -> None:
    rows = _synthetic_rows(["weibo:u1", "weibo:u2"], texts_per_author=3)
    path = _make_parquet(tmp_path, rows)
    config = _make_config(str(path), enabled=True)
    s1, _ = load_authorcorpora_samples(config)
    s2, _ = load_authorcorpora_samples(config)
    assert [s.sample_id for s in s1] == [s.sample_id for s in s2]


# ---------------------------------------------------------------------------
# build_authorcorpora_triplets
# ---------------------------------------------------------------------------

def test_build_triplets_empty_samples() -> None:
    config = _make_config(enabled=True)
    result = build_authorcorpora_triplets(config, [], "train")
    assert result == []


def test_build_triplets_single_author() -> None:
    samples = [_make_sample("weibo:a", f"text{i}") for i in range(5)]
    config = _make_config(enabled=True)
    result = build_authorcorpora_triplets(config, samples, "train")
    # Only one author → can't form a same_author_real negative
    assert result == []


def test_build_triplets_same_author_real_rule(tmp_path: Path) -> None:
    # 2 authors, 4 texts each, all in train
    samples_a = [_make_sample("weibo:a", f"作者甲的文本{i}这里有足够的内容") for i in range(4)]
    samples_b = [_make_sample("weibo:b", f"作者乙的文本{i}这里有足够的内容") for i in range(4)]
    samples = samples_a + samples_b
    config = _make_config(enabled=True)
    result = build_authorcorpora_triplets(config, samples, "train")
    assert len(result) > 0
    assert all(t.rule == "same_author_real" for t in result)


def test_build_triplets_positive_same_author_negative_different(tmp_path: Path) -> None:
    a_texts = [f"甲_{i}_足够的文字内容" for i in range(4)]
    b_texts = [f"乙_{i}_足够的文字内容" for i in range(4)]
    samples_a = [_make_sample("weibo:a", t) for t in a_texts]
    samples_b = [_make_sample("weibo:b", t) for t in b_texts]
    config = _make_config(enabled=True)
    triplets = build_authorcorpora_triplets(config, samples_a + samples_b, "train")
    assert len(triplets) > 0
    for t in triplets:
        meta = t.meta
        author = meta.get("author_id")
        neg_author = meta.get("negative_author_id")
        assert author != neg_author, "negative must come from a different author"
        # anchor text must belong to the anchor's author
        anchor_texts = a_texts if author == "weibo:a" else b_texts
        assert t.anchor_text in anchor_texts
        # positive must also belong to the same author
        assert t.positive_text in anchor_texts


def test_build_triplets_split_filter() -> None:
    # Train samples only → val split returns empty
    samples = [_make_sample("weibo:a", f"文本{i}") for i in range(4)]
    samples += [_make_sample("weibo:b", f"其他{i}") for i in range(4)]
    config = _make_config(enabled=True)
    val_result = build_authorcorpora_triplets(config, samples, "val")
    assert val_result == []


def test_build_triplets_meta_keys_present() -> None:
    samples_a = [_make_sample("weibo:a", f"甲{i}文字内容足够") for i in range(3)]
    samples_b = [_make_sample("weibo:b", f"乙{i}文字内容足够") for i in range(3)]
    config = _make_config(enabled=True)
    triplets = build_authorcorpora_triplets(config, samples_a + samples_b, "train")
    for t in triplets:
        assert "author_id" in t.meta
        assert "negative_author_id" in t.meta
