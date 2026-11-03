"""Label texts with the named style traits of src/traits.py through Jev.

    uv run python expandtraining/2606/scripts/jev_label.py <out.jsonl> <dataset triplets.jsonl> [--split val] [--pairs N]

Takes the Genshin speaker rules' texts (anchors, positives, origin_vs_para paraphrases, twins) of the given
split, skips texts already in <out.jsonl>, and appends one {"text", "scores": {trait: [score, confidence]}}
row per text, so a rerun resumes. TYPESAFE_API_KEY comes from the environment and is never printed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traits import judge

RULES = {"same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"}


def texts_of(paths: list[str], split: str, pairs: int = 0) -> list[str]:
    """All speaker-rule texts, or with `pairs` only that many origin_vs_para (line, own paraphrase) pairs,
    sampled with a fixed seed: Jev bills the questions as input too, so a full train split is ~$6."""
    texts: dict[str, None] = {}
    rows = [json.loads(line) for path in paths for line in open(path, encoding="utf-8")]
    if pairs:
        rows = [row for row in rows if row["split"] == split and row["rule"] == "origin_vs_para"]
        random.Random(0).shuffle(rows)
        for row in rows:
            if len(texts) >= 2 * pairs:
                break
            texts.setdefault(row["anchor_text"])
            texts.setdefault(row["negative_text"])
        return list(texts)
    for row in rows:
        if row["split"] != split or row["rule"] not in RULES:
            continue
        for text in (row["anchor_text"], row["positive_text"], row["negative_text"], row["meta"].get("twin_text")):
            if text:
                texts.setdefault(text)
    return list(texts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("triplets", nargs="+")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=0, help="only N (line, own paraphrase) pairs")
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["text"] for line in open(args.out, encoding="utf-8")}
    todo = [text for text in texts_of(args.triplets, args.split, args.pairs) if text not in done]
    print(f"{len(done)} labelled, {len(todo)} to go", flush=True)
    lock, started, tokens, count = threading.Lock(), time.time(), 0, 0
    with open(args.out, "a", encoding="utf-8") as handle, \
            concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(judge, text): text for text in todo}
        for future in concurrent.futures.as_completed(futures):
            text = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - one bad text must not stop the run; a rerun retries it
                print(f"failed: {text[:30]!r}: {error}", flush=True)
                continue
            with lock:
                handle.write(json.dumps({"text": text, "scores": result["scores"]}, ensure_ascii=False) + "\n")
                tokens += result["input_tokens"]
                count += 1
                if count % 1000 == 0:
                    handle.flush()
                    rate = count / (time.time() - started)
                    print(f"{count}/{len(todo)} at {rate:.1f}/s, {tokens / 1e6:.1f}M input tokens", flush=True)
    print(f"done: {count} labelled, {tokens / 1e6:.1f}M input tokens (USD {tokens / 1e6 * 0.042:.2f})", flush=True)


if __name__ == "__main__":
    main()
