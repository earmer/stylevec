"""Load multi-author corpora into SampleRecords from data/authorcorpora/unified.parquet.

Build the parquet once with:
    uv run python expandtraining/2606/src/import_authorcorpora.py

Sources included:
  weibo        : original Weibo user posts (Chinese)
  commentr     : CommentR user comments (Chinese)
  douban_short : Douban movie reviews by username (Chinese)
  douban_side  : Douban book/movie/music reviews by user_id (Chinese)
  cped         : CPED TV-drama dialogue utterances by speaker (Chinese)
  reddit       : Reddit comments of 10,000 prolific authors, 2015-2019 (English)
  authbench    : AuthBench train split, zh/en/ja documents with 2-4 docs/author (min_texts 2, see _SOURCE_MIN_TEXTS)
  ao3          : AO3 fan-fiction of the most prolific authors, cut into 200-600 char passages (English)

Excluded:
  streamprofilebench: prediction-task benchmark format, not a raw text corpus
"""
from __future__ import annotations

import hashlib
import html
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from config import resolve_repo_path
from records import SampleRecord, normalize_text, stable_id


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_AT_MENTION = re.compile(r"@[一-鿿\w]{1,30}")
_REPOST_PREFIX = re.compile(r"^(转发微博|//\s*@\S+\s*:)")


def clean_text(raw: str) -> str:
    """Unescape HTML, strip tags/URLs/@mentions/repost prefixes; collapse whitespace."""
    text = html.unescape(raw)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _AT_MENTION.sub(" ", text)
    text = _REPOST_PREFIX.sub("", text)
    return normalize_text(text)


def is_bot_like(text: str) -> bool:
    """True when the text looks automated: empty or character-diversity ratio < 0.15 over the first 100 characters
    (an alphabet has ~60 distinct characters, so over a whole long passage the ratio is low for ordinary prose)."""
    stripped = text.replace(" ", "")[:100]
    if not stripped:
        return True
    return len(stripped) > 6 and len(set(stripped)) / len(stripped) < 0.15


# ---------------------------------------------------------------------------
# Deterministic per-author split by SHA-1 hash bucket
# ---------------------------------------------------------------------------

def author_split(author_id: str, heldout_pct: int = 10, val_pct: int = 10) -> str:
    """Assign train / val / heldout deterministically from the author_id hash."""
    bucket = int(hashlib.sha1(author_id.encode()).hexdigest(), 16) % 100
    if bucket < heldout_pct:
        return "heldout"
    if bucket < heldout_pct + val_pct:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Per-source loaders  →  list[dict] with keys: source, author_id, text, lang, created, meta
# ---------------------------------------------------------------------------

def _load_weibo(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """Load original Weibo posts. Skips reposts (是否原创 != 'True')."""
    csv_path = base / "0.数据" / "weibo.csv"
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        if str(row.get("是否原创", "")).strip() != "True":
            continue
        text = clean_text(str(row.get("0", "") or ""))
        if len(text) < min_chars:
            continue
        uid = str(row.get("uid", "")).strip()
        if not uid or uid == "nan":
            continue
        rows.append({
            "source": "weibo",
            "author_id": f"weibo:{uid}",
            "text": text,
            "lang": "zh",
            "created": str(row.get("日期", "") or ""),
            "meta": "{}",
        })
    return rows


def _load_commentr(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """Load CommentR user comments. Comments.json (~1.3 GB) is loaded in one pass."""
    with open(base / "Comments.json", encoding="utf-8") as fh:
        data = json.load(fh)
    rows: list[dict[str, Any]] = []
    for item in data:
        cu = item.get("comment_user")
        if not isinstance(cu, dict):
            continue
        uid = str(cu.get("_id", "")).strip()
        if not uid or uid == "nan":
            continue
        text = clean_text(str(item.get("content", "") or ""))
        if len(text) < min_chars:
            continue
        rows.append({
            "source": "commentr",
            "author_id": f"commentr:{uid}",
            "text": text,
            "lang": "zh",
            "created": str(item.get("created_at", "") or ""),
            "meta": "{}",
        })
    return rows


def _load_douban_short(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """Load Douban movie reviews. Skips deleted accounts ([已注销], 已注销)."""
    df = pd.read_csv(base / "DMSC.csv", encoding="utf-8-sig", low_memory=False)
    _deleted = {"[已注销]", "已注销"}
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        username = str(row.get("Username", "") or "").strip()
        if not username or username == "nan" or username in _deleted:
            continue
        text = clean_text(str(row.get("Comment", "") or ""))
        if len(text) < min_chars:
            continue
        rows.append({
            "source": "douban_short",
            "author_id": f"douban_short:{username}",
            "text": text,
            "lang": "zh",
            "created": str(row.get("Date", "") or ""),
            "meta": json.dumps({"star": str(row.get("Star", ""))}),
        })
    return rows


def _load_douban_side(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """Load Douban book/movie/music reviews from TSV files."""
    txt_dir = base / "douban_dataset(text information)"
    rows: list[dict[str, Any]] = []
    for fname in ("bookreviews_cleaned.txt", "moviereviews_cleaned.txt", "musicreviews_cleaned.txt"):
        fpath = txt_dir / fname
        if not fpath.exists():
            continue
        category = fname.replace("reviews_cleaned.txt", "")
        df = pd.read_csv(fpath, sep="\t", encoding="utf-8-sig", low_memory=False)
        for _, row in df.iterrows():
            uid = str(row.get("user_id", "") or "").strip()
            if not uid or uid == "nan":
                continue
            text = clean_text(str(row.get("comment", "") or ""))
            if len(text) < min_chars:
                continue
            rows.append({
                "source": "douban_side",
                "author_id": f"douban_side:{uid}",
                "text": text,
                "lang": "zh",
                "created": str(row.get("time", "") or ""),
                "meta": json.dumps({"category": category}),
            })
    return rows


def _load_cped(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """Load CPED TV-drama utterances by speaker (character name)."""
    cped_dir = base / "data" / "CPED"
    rows: list[dict[str, Any]] = []
    for split_name in ("train_split", "valid_split", "test_split"):
        fpath = cped_dir / f"{split_name}.csv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False)
        for _, row in df.iterrows():
            speaker = str(row.get("Speaker", "") or "").strip()
            if not speaker or speaker == "nan":
                continue
            text = clean_text(str(row.get("Utterance", "") or ""))
            if len(text) < min_chars:
                continue
            rows.append({
                "source": "cped",
                "author_id": f"cped:{speaker}",
                "text": text,
                "lang": "zh",
                "created": "",
                "meta": json.dumps({"tv_id": str(row.get("TV_ID", ""))}),
            })
    return rows


# ---------------------------------------------------------------------------
# Parquet builder
# ---------------------------------------------------------------------------

_REDDIT_QUOTE = re.compile(r"^\s*(>|&gt;).*$", re.M)  # quoted text is someone else's words
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _load_reddit(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """English Reddit comments of 10,000 prolific authors (2015-2019, 100-5,000 chars, Kaggle/BigQuery export).
    Quoted lines and markdown link targets are dropped; [removed]/[deleted] bodies are skipped."""
    df = pd.read_parquet(base / "reddit_authorship_10000authors.parquet")
    rows: list[dict[str, Any]] = []
    for author, body, subreddit, created in zip(df["author"], df["body"], df["subreddit"], df["created_utc"]):
        body = str(body or "")
        if body.strip() in ("[removed]", "[deleted]"):
            continue
        text = clean_text(_MD_LINK.sub(r"\1", _REDDIT_QUOTE.sub(" ", body)))
        if len(text) < min_chars or not author:
            continue
        rows.append({
            "source": "reddit",
            "author_id": f"reddit:{author}",
            "text": text,
            "lang": "en",
            "created": str(created),
            "meta": json.dumps({"subreddit": str(subreddit)}),
        })
    return rows



_AUTHBENCH_LANGS = {"zh", "en", "ja"}
_AUTHBENCH_MAX_CHARS = 2000  # a few wikisource/arxiv documents run past 10k chars
_WATERMARK = re.compile(r"▬[^▬\n]*▬|◗[^◗◖\n]*◖")  # exorde appends "▬tag▬ ◗id◖" blocks
_WIKI_MARKUP = re.compile(r"={2,}|\{\{|-\{")  # wikisource headings and templates


def _is_markup_noise(text: str) -> bool:
    """True for wiki-markup rows and rows where over 30% of the non-space characters are not letters or digits."""
    if _WIKI_MARKUP.search(text):
        return True
    stripped = re.sub(r"\s", "", text)
    return not stripped or 1 - sum(ch.isalnum() for ch in stripped) / len(stripped) > 0.30


def _load_authbench(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """AuthBench zh/en/ja documents with input_split == "train". Only train/documents.jsonl is opened (never dev/ or
    test/); that file also carries dev/test-tagged rows, and authors who appear in any of them are dropped whole."""
    with open(base / "train" / "documents.jsonl", encoding="utf-8") as fh:
        docs = [json.loads(line) for line in fh]
    held_out = {d["author_id"] for d in docs if d.get("input_split") != "train"}
    rows: list[dict[str, Any]] = []
    for doc in docs:
        if doc.get("input_split") != "train" or doc["author_id"] in held_out or doc.get("lang") not in _AUTHBENCH_LANGS:
            continue
        text = clean_text(_WATERMARK.sub(" ", str(doc.get("content", "") or "")))
        if not min_chars <= len(text) <= _AUTHBENCH_MAX_CHARS or _is_markup_noise(text):
            continue
        rows.append({
            "source": "authbench",
            "author_id": f"authbench:{doc['author_id'][:16]}",
            "text": text,
            "lang": doc["lang"],
            "created": "",
            "meta": json.dumps({"genre": str(doc.get("genre", "")), "origin": str(doc.get("source", ""))}),
        })
    return rows


_AO3_LANGS = {"English": "en", "Italiano": "it", "Español": "es", "Русский": "ru", "Français": "fr"}
_SENTENCE = re.compile(r"[^.!?。！？]+[.!?。！？]+[\"”'’」』]*|[^.!?。！？]+$")


def split_passages(text: str, low: int = 200, high: int = 600) -> list[str]:
    """Pack consecutive sentences into passages of low..high chars; a short tail is dropped.
    Many AO3 dumps have no paragraph breaks, so sentences (not paragraphs) are the unit."""
    units: list[str] = []
    for sentence in (s.strip() for s in _SENTENCE.findall(text)):
        while len(sentence) > high:  # a run-on sentence is wrapped at the last space before high
            cut = sentence.rfind(" ", 0, high) + 1 or high
            units.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            units.append(sentence)
    passages: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > high:
            if len(current) >= low:
                passages.append(current)
            current = unit
        else:
            current = f"{current} {unit}".strip()
    if len(current) >= low:
        passages.append(current)
    return passages


def _load_ao3(base: Path, min_chars: int) -> list[dict[str, Any]]:
    """AO3 works of the top-150 authors as 200-600 char passages. Each author's passages are interleaved across
    works, so the per-author cap spreads over every work instead of keeping only the first one."""
    df = pd.read_parquet(base / "top150_authors.parquet")
    per_author: defaultdict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for work_id, title, metadata, text, author in zip(df["id"], df["title"], df["metadata"], df["text"], df["author"]):
        lang = _AO3_LANGS.get(str((metadata or {}).get("Language") or ""))
        author = str(author or "").strip()
        if lang is None or not author:
            continue
        meta = json.dumps({"work_id": str(work_id), "title": str(title or "")}, ensure_ascii=False)
        passages = [clean_text(p) for p in split_passages(str(text or ""))]
        per_author[author].append([
            {"source": "ao3", "author_id": f"ao3:{author}", "text": p, "lang": lang, "created": "", "meta": meta}
            for p in passages if len(p) >= min_chars and not _is_markup_noise(p)
        ])
    rows: list[dict[str, Any]] = []
    for author in sorted(per_author):
        for group in itertools.zip_longest(*per_author[author]):
            rows.extend(row for row in group if row is not None)
    return rows


_SOURCE_SUBDIR: dict[str, str] = {
    "reddit": "reddit_authorship",
    "weibo": "weibo_users",
    "commentr": "commentr",
    "douban_short": "douban_short",
    "douban_side": "douban_side",
    "cped": "CPED",
    "authbench": "authbench",
    "ao3": "corpora/ao3_random_subset",
}
_SOURCE_MIN_TEXTS: dict[str, int] = {"authbench": 2}  # AuthBench has at most 5 docs per author


def build_unified_parquet(
    out_path: Path,
    datasets_dir: Path,
    min_chars: int = 10,
    cped_min_chars: int = 4,
    min_texts: int = 20,
    texts_per_author: int = 256,
    author_cap: int | None = None,
    heldout_pct: int = 10,
    val_pct: int = 10,
) -> dict[str, int]:
    """Normalise all sources → unified parquet; return per-source counts.

    Cleaning applied:
      - HTML / URL / @mention / repost-prefix stripping
      - Global exact-duplicate text removal (case-folded)
      - Bot-like filter (char-diversity < 0.15 for long texts)
      - min_chars length floor (cped_min_chars for CPED)
      - Authors with < min_texts retained texts are dropped (_SOURCE_MIN_TEXTS overrides it per source)
      - At most texts_per_author texts kept per author (deterministic: first N in source order)
    """
    loaders: list[tuple[str, Any, int]] = [
        ("weibo",        _load_weibo,        min_chars),
        ("commentr",     _load_commentr,     min_chars),
        ("douban_short", _load_douban_short, min_chars),
        ("douban_side",  _load_douban_side,  min_chars),
        ("cped",         _load_cped,         cped_min_chars),
        ("reddit",       _load_reddit,       min_chars),
        ("authbench",    _load_authbench,    min_chars),
        ("ao3",          _load_ao3,          min_chars),
    ]

    all_rows: list[dict[str, Any]] = []
    raw_counts: dict[str, int] = {}
    for source, fn, mc in loaders:
        src_dir = datasets_dir / _SOURCE_SUBDIR[source]
        rows = fn(src_dir, mc)
        raw_counts[f"raw_{source}"] = len(rows)
        all_rows.extend(rows)

    # Global exact-text deduplication (case-folded), then bot filter
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in all_rows:
        key = row["text"].casefold()
        if key not in seen:
            seen.add(key)
            if not is_bot_like(row["text"]):
                deduped.append(row)

    # Group by author; apply min_texts and per-author cap
    by_author: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        by_author[row["author_id"]].append(row)

    final: list[dict[str, Any]] = []
    authors_used = 0
    for author_id in sorted(by_author):
        if author_cap is not None and authors_used >= author_cap:
            break
        rows = by_author[author_id]
        if len(rows) < _SOURCE_MIN_TEXTS.get(rows[0]["source"], min_texts):
            continue
        split = author_split(author_id, heldout_pct=heldout_pct, val_pct=val_pct)
        for row in rows[:texts_per_author]:
            final.append({
                "source":    row["source"],
                "author_id": author_id,
                "text":      row["text"],
                "lang":      row["lang"],
                "n_chars":   len(row["text"]),
                "created":   row["created"],
                "meta":      row["meta"],
                "split":     split,
            })
        authors_used += 1

    df = pd.DataFrame(
        final,
        columns=["source", "author_id", "text", "lang", "n_chars", "created", "meta", "split"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    result: dict[str, int] = dict(raw_counts)
    result["total_rows"]    = len(final)
    result["total_authors"] = authors_used
    for src in _SOURCE_SUBDIR:
        result[f"authors_{src}"] = len({r["author_id"] for r in final if r["source"] == src})
    return result


# ---------------------------------------------------------------------------
# Importer (called by build_triplets.py)
# ---------------------------------------------------------------------------

def load_authorcorpora_samples(config: dict[str, Any]) -> tuple[list[SampleRecord], dict[str, int]]:
    """Load authorcorpora SampleRecords from the unified parquet.

    Returns ([], {}) when data.authorcorpora_enabled is falsy (default).
    Raises FileNotFoundError if enabled but the parquet is missing.
    """
    if not bool(config["data"].get("authorcorpora_enabled", False)):
        return [], {}

    parquet_path = resolve_repo_path(
        str(config["paths"].get("authorcorpora_parquet", "data/authorcorpora/unified.parquet"))
    )
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"authorcorpora parquet not found: {parquet_path}\n"
            "Build it first: uv run python expandtraining/2606/src/import_authorcorpora.py"
        )

    frame = pd.read_parquet(parquet_path)
    required = {"author_id", "text", "lang", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"authorcorpora parquet missing columns {sorted(missing)}: {parquet_path}")

    min_chars     = int(config["data"].get("authorcorpora_min_chars",     config["data"]["min_chars"]))
    max_chars     = int(config["data"].get("authorcorpora_max_chars",     config["data"]["max_chars"]))
    texts_per     = int(config["data"].get("authorcorpora_texts_per_author", 256))
    min_texts     = int(config["data"].get("authorcorpora_min_texts",     20))
    author_cap_v  = config["data"].get("authorcorpora_author_cap")
    author_cap    = int(author_cap_v) if author_cap_v is not None else None

    filter_counts: defaultdict[str, int] = defaultdict(int)
    by_author: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_texts: set[str] = set()

    for _, row in frame.iterrows():
        author_id = normalize_text(str(row["author_id"]))
        lang      = normalize_text(str(row.get("lang", "zh")))
        text      = normalize_text(str(row["text"]))
        split     = str(row.get("split", "train"))
        if not author_id or not text:
            filter_counts["empty_field"] += 1
            continue
        if not min_chars <= len(text) <= max_chars:
            filter_counts["bad_length"] += 1
            continue
        key = text.casefold()
        if key in seen_texts:
            filter_counts["duplicate_text"] += 1
            continue
        seen_texts.add(key)
        meta_raw = str(row.get("meta", "{}") or "{}")
        by_author[author_id].append({
            "text": text, "lang": lang, "split": split, "meta": meta_raw, "source": str(row.get("source", "")),
        })

    samples: list[SampleRecord] = []
    authors_kept = 0
    for author_id, rows in sorted(by_author.items()):
        if len(rows) < _SOURCE_MIN_TEXTS.get(rows[0]["source"], min_texts):
            filter_counts["below_min_texts"] += len(rows)
            continue
        if author_cap is not None and authors_kept >= author_cap:
            filter_counts["author_cap"] += len(rows)
            continue
        for index, rd in enumerate(rows[:texts_per]):
            try:
                extra: dict[str, Any] = json.loads(rd["meta"]) if rd["meta"] else {}
            except (json.JSONDecodeError, TypeError):
                extra = {}
            sid = stable_id("authorcorpora", author_id, rd["text"])
            samples.append(SampleRecord(
                sample_id=sid,
                text=rd["text"],
                style_id=author_id,
                style_type="author_id",
                domain="authorcorpora",
                content_id=stable_id(rd["text"]),
                source_id=sid,
                language=rd["lang"],
                split=rd["split"],
                quality=1.0,
                meta={"index": index, **extra},
            ))
        authors_kept += 1

    filter_counts["authors_kept"] = authors_kept
    return samples, dict(filter_counts)


# ---------------------------------------------------------------------------
# CLI: build the parquet
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Build data/authorcorpora/unified.parquet")
    parser.add_argument("--datasets-dir",      default="data/datasets",
                        help="Repo-relative path to datasets dir (default: data/datasets)")
    parser.add_argument("--out",               default="data/authorcorpora/unified.parquet",
                        help="Repo-relative output path")
    parser.add_argument("--min-chars",         type=int, default=10)
    parser.add_argument("--cped-min-chars",    type=int, default=4)
    parser.add_argument("--min-texts",         type=int, default=20)
    parser.add_argument("--texts-per-author",  type=int, default=256)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    datasets_dir = (repo_root / args.datasets_dir).resolve()
    out_path     = (repo_root / args.out).resolve()

    print(f"datasets_dir : {datasets_dir}")
    print(f"out          : {out_path}")
    t0 = time.time()
    counts = build_unified_parquet(
        out_path=out_path,
        datasets_dir=datasets_dir,
        min_chars=args.min_chars,
        cped_min_chars=args.cped_min_chars,
        min_texts=args.min_texts,
        texts_per_author=args.texts_per_author,
    )
    elapsed = time.time() - t0
    print(f"\nResults ({elapsed:.0f}s):")
    for key, val in sorted(counts.items()):
        print(f"  {key}: {val:,}")
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\nParquet size: {size_mb:.1f} MB → {out_path}")
