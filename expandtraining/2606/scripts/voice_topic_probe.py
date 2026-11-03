"""Voice versus topic on held-out Reddit authors: does the embedding follow the writer or the subreddit?

    uv run python expandtraining/2606/scripts/voice_topic_probe.py build [data/episodes/reddit_probe.jsonl]
    uv run python expandtraining/2606/scripts/voice_topic_probe.py run --run-name <run> [--checkpoint latest]

build: items from unified.parquet's held-out Reddit authors (never trained on): a query comment, the same
author's comment in ANOTHER subreddit (voice, different topic) and another held-out author's comment in the
query's subreddit (topic, different voice; length-matched to the voice comment; catch-all subreddits are never the
topic). run: accuracy = share of items with cos(query, voice) > cos(query, topic), also averaged per author; 0.5
is chance, below it the model sorts by topic. Writes validation/voice_topic_probe.json in the run dir.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from config import resolve_repo_path

DEFAULT_ITEMS = "data/episodes/reddit_probe.jsonl"


# Catch-all subreddits share no topic: two comments there are as unrelated as two random ones, so they never
# serve as the query's topic (they may still be the voice comment's subreddit).
CATCH_ALL = {"AskReddit", "pics", "funny", "WTF", "todayilearned", "IAmA", "AdviceAnimals", "videos", "gifs"}


def build_items(frame, count: int, per_author: int = 4, seed: int = 0) -> list[dict[str, str]]:
    """Up to per_author items per author, each from a different query subreddit. The topic comment is the
    same-subreddit comment whose length is closest to the voice comment's, so length habits cannot decide."""
    frame = frame[(frame["split"] == "heldout") & frame["n_chars"].between(40, 600)].copy()
    frame["topic"] = frame["meta"].map(lambda meta: json.loads(meta or "{}").get("subreddit"))
    frame = frame.dropna(subset=["topic"]).drop_duplicates(subset=["text"])
    by_author = {a: g for a, g in frame.groupby("author_id") if g["topic"].nunique() >= 2}
    by_topic = {t: g for t, g in frame.groupby("topic") if g["author_id"].nunique() >= 2 and t not in CATCH_ALL}
    rng = random.Random(seed)
    items: list[dict[str, str]] = []
    authors = sorted(by_author)
    rng.shuffle(authors)
    for author in authors:
        group = by_author[author]
        topics = sorted(set(group["topic"]) & set(by_topic))
        rng.shuffle(topics)
        for topic_name in topics[:per_author]:
            queries = group[group["topic"] == topic_name]
            query = queries.iloc[rng.randrange(len(queries))]
            voices = group[group["topic"] != topic_name]
            voice = voices.iloc[rng.randrange(len(voices))]
            others = by_topic[topic_name]
            others = others[others["author_id"] != author]
            topic = others.iloc[int((others["n_chars"] - voice["n_chars"]).abs().to_numpy().argmin())]
            items.append({"author": author, "subreddit": topic_name, "query": query["text"],
                          "voice": voice["text"], "topic": topic["text"]})
        if len(items) >= count:
            break
    return items[:count]


def accuracy(query: np.ndarray, voice: np.ndarray, topic: np.ndarray) -> dict[str, float]:
    unit = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in (query, voice, topic)]
    cos_voice, cos_topic = (unit[0] * unit[1]).sum(1), (unit[0] * unit[2]).sum(1)
    return {"items": int(len(query)), "voice_over_topic": float(np.mean(cos_voice > cos_topic)),
            "cos_voice": float(cos_voice.mean()), "cos_topic": float(cos_topic.mean())}


def author_macro(items: list[dict[str, str]], wins: np.ndarray) -> float:
    """Mean over authors of each author's item accuracy, so prolific authors do not dominate."""
    per: dict[str, list[bool]] = {}
    for item, win in zip(items, wins):
        per.setdefault(item["author"], []).append(bool(win))
    return float(np.mean([np.mean(v) for v in per.values()]))


def embed(texts: list[str], run_name: str, checkpoint: str) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    from config import load_config, workspace_dir
    from model import build_qwen_style_encoder
    from validate import load_checkpoint_for_validation, select_device

    run_dir = workspace_dir() / "outputs" / "runs" / run_name
    config = load_config(str(run_dir / "config.yaml"))
    config["runtime"]["device"] = "auto"  # runs trained on CUDA are probed wherever the weights were pulled
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = select_device(config)
    model = build_qwen_style_encoder(config).to(device).eval()
    load_checkpoint_for_validation(model, run_dir / "checkpoints" / checkpoint)
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            tokens = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                               max_length=int(config["model"]["max_length"]), return_tensors="pt")
            output = model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))
            out.append(output["embeddings"].float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("out", nargs="?", default=DEFAULT_ITEMS)
    build.add_argument("--count", type=int, default=4000)
    run = sub.add_parser("run")
    run.add_argument("--run-name", required=True)
    run.add_argument("--checkpoint", default="latest")
    run.add_argument("--items", default=DEFAULT_ITEMS)
    args = parser.parse_args()
    if args.command == "build":
        import pandas as pd

        frame = pd.read_parquet(resolve_repo_path("data/authorcorpora/unified.parquet"),
                                filters=[("source", "=", "reddit")],
                                columns=["author_id", "text", "n_chars", "meta", "split"])
        items = build_items(frame, args.count)
        out = resolve_repo_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
        print(f"{out}: {len(items)} items")
        return
    items = [json.loads(line) for line in resolve_repo_path(args.items).read_text(encoding="utf-8").splitlines()]
    n = len(items)
    vectors = embed([i["query"] for i in items] + [i["voice"] for i in items] + [i["topic"] for i in items],
                    args.run_name, args.checkpoint)
    q, v, t = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (vectors[:n], vectors[n:2 * n], vectors[2 * n:]))
    result = {"checkpoint": args.checkpoint, **accuracy(q, v, t),
              "voice_over_topic_author_macro": author_macro(items, (q * v).sum(1) > (q * t).sum(1)),
              "authors": len({i["author"] for i in items})}
    from config import workspace_dir

    out = workspace_dir() / "outputs" / "runs" / args.run_name / "validation" / "voice_topic_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
