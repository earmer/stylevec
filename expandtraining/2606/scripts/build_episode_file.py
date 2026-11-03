"""Write train.episode_file: Reddit training authors' comments with their subreddit as the topic.

    uv run python expandtraining/2606/scripts/build_episode_file.py [out.jsonl] [--min-topics 8] [--per-topic 3]

Reads data/authorcorpora/unified.parquet (src/import_authorcorpora.py), keeps the train split only (held-out
and val authors never enter), and authors with comments in at least --min-topics subreddits. Each author keeps
up to --per-topic comments per subreddit and --per-author in all, so an episode pair can always take every
line from a different subreddit (src/episodes.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import resolve_repo_path

DEFAULT_OUT = "data/episodes/reddit_train.jsonl"


def episode_rows(frame: pd.DataFrame, min_topics: int, per_topic: int, per_author: int, min_chars: int,
                 max_chars: int, seed: int = 0) -> list[dict[str, str]]:
    frame = frame[(frame["split"] == "train") & frame["n_chars"].between(min_chars, max_chars)].copy()
    frame["topic"] = frame["meta"].map(lambda meta: json.loads(meta or "{}").get("subreddit"))
    frame = frame.dropna(subset=["topic"]).drop_duplicates(subset=["text"])
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for author, group in frame.groupby("author_id", sort=True):
        topics = {topic: list(texts) for topic, texts in group.groupby("topic")["text"]}
        if len(topics) < min_topics:
            continue
        kept: list[dict[str, str]] = []
        for topic in sorted(topics):
            texts = topics[topic]
            rng.shuffle(texts)
            kept += [{"author": str(author), "topic": str(topic), "text": str(text)} for text in texts[:per_topic]]
        rng.shuffle(kept)
        rows += kept[:per_author]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=DEFAULT_OUT)
    parser.add_argument("--corpus", default="data/authorcorpora/unified.parquet")
    parser.add_argument("--min-topics", type=int, default=8)
    parser.add_argument("--per-topic", type=int, default=6)
    parser.add_argument("--per-author", type=int, default=120)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=600)
    args = parser.parse_args()
    frame = pd.read_parquet(resolve_repo_path(args.corpus), filters=[("source", "=", "reddit")],
                            columns=["author_id", "text", "n_chars", "meta", "split"])
    rows = episode_rows(frame, args.min_topics, args.per_topic, args.per_author, args.min_chars, args.max_chars)
    out = resolve_repo_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    authors = {row["author"] for row in rows}
    print(f"{out}: {len(rows)} comments, {len(authors)} authors, {len({r['topic'] for r in rows})} subreddits")


if __name__ == "__main__":
    main()
