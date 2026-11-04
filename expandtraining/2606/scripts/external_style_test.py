"""Pre-registered test on labels this project did not produce: does a run beat a text-only baseline on new speakers?

    uv run python expandtraining/2606/scripts/external_style_test.py build
    uv run python expandtraining/2606/scripts/external_style_test.py run --items <items.jsonl> --out <result.json>

build: items from held-out speakers. cped: query, the speaker's line in another dialogue (positive), another speaker's
line in the query's dialogue, closest in length to the positive (negative). authbench and aozora: query, the author's
other text (another work on aozora), another author's text closest in length (same genre on authbench). Each set is
also written with speakers shuffled, which removes voice and keeps topic: an encoder's score there is its own null.
run: share of items with cos(query, positive) > cos(query, negative), ties 0.5, for char TF-IDF, the untrained encoder
and each run, with the difference from TF-IDF and a 99% bootstrap interval over speakers.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import resolve_repo_path

SEED = 20261104
RUNS = {"data": "head_f_1gpu__data__s42", "episodes": "head_f_1gpu__episodes__s42",
        "push": "head_f_1gpu__push__s42", "both": "head_f_1gpu__both__s42"}
OUT_DIR = "data/episodes/external"


def closest(pool: list[dict], length: int) -> dict:
    return min(pool, key=lambda row: abs(len(row["text"]) - length))


def cped_rows() -> list[dict]:
    frame = pd.read_parquet(resolve_repo_path("data/authorcorpora/unified.parquet"), filters=[("source", "=", "cped")],
                            columns=["author_id", "split"])
    held = {a.split(":", 1)[1] for a in frame[frame["split"] == "heldout"]["author_id"]}
    rows, base = [], resolve_repo_path("data/datasets/CPED/data/CPED")
    for name in ("train_split", "valid_split", "test_split"):
        with open(base / f"{name}.csv", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row["Utterance"].strip()
                if 6 <= len(text) <= 220:
                    rows.append({"speaker": row["Speaker"], "dialogue": row["Dialogue_ID"], "text": text,
                                 "held": row["Speaker"] in held})
    return rows


def cped_items(rows: list[dict], shuffle: bool, per_speaker: int = 60) -> list[dict]:
    rng = random.Random(SEED)
    held = [dict(row) for row in rows if row["held"]]
    if shuffle:
        labels = [row["speaker"] for row in held]
        rng.shuffle(labels)
        for row, label in zip(held, labels):
            row["speaker"] = label
    by_speaker: dict[str, list[dict]] = {}
    for row in held:
        by_speaker.setdefault(row["speaker"], []).append(row)
    by_dialogue: dict[str, list[dict]] = {}  # the negative may be any other speaker in the scene, held out or not
    for row in held + [row for row in rows if not row["held"]]:
        by_dialogue.setdefault(row["dialogue"], []).append(row)
    items = []
    for speaker in sorted(by_speaker):
        lines = list(by_speaker[speaker])
        rng.shuffle(lines)
        kept = 0
        for query in lines:
            others = [r for r in lines if r["dialogue"] != query["dialogue"] and r["text"] != query["text"]]
            rivals = [r for r in by_dialogue[query["dialogue"]] if r["speaker"] != speaker and r["text"] != query["text"]]
            if not others or not rivals:
                continue
            positive = rng.choice(others)
            negative = closest(rivals, len(positive["text"]))
            items.append({"group": speaker, "query": query["text"], "positive": positive["text"],
                          "negative": negative["text"]})
            kept += 1
            if kept == per_speaker:
                break
    return items


def corpus_items(frame: pd.DataFrame, key: str, per_author: int, count: int, min_chars: int, max_chars: int,
                 shuffle: bool, same_key: bool) -> list[dict]:
    """same_key: the negative shares the query's key (genre). Otherwise the positive must differ from it (work)."""
    rng = random.Random(SEED)
    frame = frame[frame["n_chars"].between(min_chars, max_chars)].drop_duplicates("text").copy()
    frame["key"] = frame["meta"].map(lambda meta: str(json.loads(meta or "{}").get(key)))
    if shuffle:
        frame["author_id"] = np.random.default_rng(SEED).permutation(frame["author_id"].to_numpy())
    records = frame[["author_id", "text", "key"]].to_dict("records")
    by_author: dict[str, list[dict]] = {}
    by_key: dict[str, list[dict]] = {}
    for row in records:
        by_author.setdefault(row["author_id"], []).append(row)
        by_key.setdefault(row["key"], []).append(row)
    items: list[dict] = []
    authors = sorted(by_author)
    rng.shuffle(authors)
    for author in authors:
        lines = list(by_author[author])
        rng.shuffle(lines)
        for query in lines[:per_author]:
            others = [r for r in lines if r is not query and (same_key or r["key"] != query["key"])]
            pool = by_key[query["key"]] if same_key else records
            sample = rng.sample(pool, min(len(pool), 400))
            rivals = [r for r in sample if r["author_id"] != author]
            if not others or not rivals:
                continue
            positive = rng.choice(others)
            negative = closest(rivals, len(positive["text"]))
            items.append({"group": author, "query": query["text"], "positive": positive["text"], "negative": negative["text"]})
        if len(items) >= count:
            break
    return items[:count]


def build() -> None:
    out = resolve_repo_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    unified = "data/authorcorpora/unified.parquet"
    columns = ["author_id", "text", "n_chars", "meta", "split"]
    auth = pd.read_parquet(resolve_repo_path(unified), filters=[("source", "=", "authbench")], columns=columns)
    auth = auth[auth["split"] == "heldout"]
    new = pd.read_parquet(resolve_repo_path("data/authorcorpora/unified_new_sources.parquet"),
                          filters=[("source", "=", "aozora")], columns=columns)
    new = new[new["split"] == "heldout"]
    rows = cped_rows()
    sets = {"cped": lambda s: cped_items(rows, s),
            "authbench": lambda s: corpus_items(auth, "genre", 2, 3000, 20, 600, s, True),
            "aozora": lambda s: corpus_items(new, "work_id", 40, 1500, 40, 2000, s, False)}
    for name, make in sets.items():
        for shuffle in (False, True):
            items = make(shuffle)
            path = out / f"{name}{'_shuffled' if shuffle else ''}.jsonl"
            path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
            print(path.name, len(items), "items", len({i["group"] for i in items}), "speakers")


def wins(q: np.ndarray, p: np.ndarray, n: np.ndarray) -> np.ndarray:
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)  # noqa: E731
    delta = (unit(q) * unit(p)).sum(1) - (unit(q) * unit(n)).sum(1)
    return (delta > 0).astype(float) + 0.5 * (delta == 0)


def untrained(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer

    from config import load_config, workspace_dir
    from model import build_qwen_style_encoder
    from validate import select_device

    config = load_config(str(workspace_dir() / "outputs" / "runs" / RUNS["data"] / "config.yaml"))
    config["runtime"]["device"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    device = select_device(config)
    model = build_qwen_style_encoder(config).to(device).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 64):
            tokens = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                               max_length=int(config["model"]["max_length"]), return_tensors="pt")
            out.append(model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))
                       ["embeddings"].float().cpu().numpy())
    return np.concatenate(out)


def run(items_path: str, out_path: str) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    from voice_topic_probe import embed

    items = [json.loads(line) for line in open(resolve_repo_path(items_path), encoding="utf-8")]
    groups = np.array([i["group"] for i in items])
    texts = sorted({i[k] for i in items for k in ("query", "positive", "negative")})
    index = {t: n for n, t in enumerate(texts)}
    take = [[index[i[k]] for i in items] for k in ("query", "positive", "negative")]
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True, max_features=100000).fit_transform(texts)
    score = {}
    a, b, c = (tfidf[t] for t in take)
    delta = np.asarray(a.multiply(b).sum(1)).ravel() - np.asarray(a.multiply(c).sum(1)).ravel()
    score["tfidf"] = (delta > 0).astype(float) + 0.5 * (delta == 0)
    lengths = np.array([[len(t)] for t in texts], dtype=float)
    q, p, n = (lengths[t] for t in take)
    gap = np.abs(q - n).ravel() - np.abs(q - p).ravel()
    score["length"] = (gap > 0).astype(float) + 0.5 * (gap == 0)
    vectors = untrained(texts)
    score["untrained"] = wins(*(vectors[t] for t in take))
    for name, run_name in RUNS.items():
        vectors = embed(texts, run_name, "latest")
        score[name] = wins(*(vectors[t] for t in take))
        print(name, round(float(score[name].mean()), 4), flush=True)
    rng = np.random.default_rng(SEED)
    speakers = sorted(set(groups))
    rows = {s: np.flatnonzero(groups == s) for s in speakers}
    result = {"items": len(items), "speakers": len(speakers), "scores": {}}
    for name, values in score.items():
        macro = float(np.mean([values[rows[s]].mean() for s in speakers]))
        diff = values - score["tfidf"]
        boots = [diff[np.concatenate([rows[s] for s in rng.choice(speakers, len(speakers))])].mean() for _ in range(2000)]
        result["scores"][name] = {"accuracy": float(values.mean()), "macro": macro, "minus_tfidf": float(diff.mean()),
                                  "ci99": [float(x) for x in np.quantile(boots, [0.005, 0.995])]}
    out = resolve_repo_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    go = sub.add_parser("run")
    go.add_argument("--items", required=True)
    go.add_argument("--out", required=True)
    args = parser.parse_args()
    build() if args.command == "build" else run(args.items, args.out)


if __name__ == "__main__":
    main()
