"""Same-content pairs with Jev scores on both sides, for train.named_trait_pair_files (src/named_traits.py).

    uv run python expandtraining/2606/scripts/trait_pairs.py <out.jsonl> [--max-para N]

Three sources, each row {"a", "b", "scores_a", "scores_b", "source"}:
  para  a Genshin training line and its own neutral paraphrase (outputs/traits/genshin_train.jsonl, d-402f1c32b449)
  tone  a Genshin training line and its kept tone twin (scripts/genshin_tone_rewrite.py, outputs/tonetwin/)
  book  a book passage and its kept style rewrite (scripts/book_style_rewrite.py, outputs/bookstyle/)
Rows missing any trait score are skipped (early book verifies scored only the styled traits).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traits import TRAITS

OUT = Path(__file__).resolve().parents[1] / "outputs"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def complete(scores: dict) -> bool:
    return all(name in scores for name in TRAITS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--max-para", type=int, default=0,
                        help="keep at most N paraphrase pairs (seeded sample): many are near-copies, and they "
                             "would otherwise be ~75%% of the pairs")
    args = parser.parse_args()
    labels = {row["text"]: row["scores"] for row in read(OUT / "traits/genshin_train.jsonl")}
    rows, seen = [], set()
    for row in read(OUT / "datasets/d-402f1c32b449/triplets.jsonl"):
        a, b = row["anchor_text"], row["negative_text"]
        if row["split"] == "train" and row["rule"] == "origin_vs_para" and a in labels and b in labels \
                and a != b and (a, b) not in seen:
            seen.add((a, b))
            rows.append({"a": a, "b": b, "scores_a": labels[a], "scores_b": labels[b], "source": "para"})
    if args.max_para and len(rows) > args.max_para:
        rows = random.Random(0).sample(rows, args.max_para)
    for row in read(OUT / "tonetwin/verify.jsonl"):
        if row["kept"] and row["source"] in labels and complete(row["scores"]):
            rows.append({"a": row["source"], "b": row["text"], "scores_a": labels[row["source"]],
                         "scores_b": row["scores"], "source": "tone"})
    sources = {row["chunk_id"]: row["scores"] for row in read(OUT / "bookstyle/verify_sources.jsonl")}
    for row in read(OUT / "bookstyle/verify.jsonl"):
        before = sources.get(row["chunk_id"], {})
        if row["kept"] and complete(before) and complete(row["scores"]):
            rows.append({"a": row["source"], "b": row["text"], "scores_a": before, "scores_b": row["scores"],
                         "source": "book"})
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {name: sum(row["source"] == name for row in rows) for name in ("para", "tone", "book")}
    print(f"{len(rows)} pairs {counts}")


if __name__ == "__main__":
    main()
