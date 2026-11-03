"""Tone twins for Genshin training lines: the same line with its speaker's most distinctive tone inverted.

    uv run python expandtraining/2606/scripts/genshin_tone_rewrite.py pick|rewrite|polish|verify|report [--per-speaker 20]

A neutral paraphrase keeps a line's tone (Jev: irony, warmth, valence, hedging, swagger r 0.8-0.9 line vs
paraphrase) and often copies the line, so it is no contrast for tone. Here each picked line (Jev-labelled in
outputs/traits/genshin_train.jsonl) gets one rewrite toward the opposite end of the tone trait on which it is
furthest from neutral (balanced across styles), keeping content, register and length; the generation LLM polishes it, and Jev verifies the
trait moved (book_style_rewrite.kept). Lines with no tone trait >= 1 level from neutral are skipped. Reuses the
book pipeline's proxy, batching and judge; writes outputs/tonetwin/.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import book_style_rewrite as book

TONE_TRAITS = {"irony": ("sarcastic", "earnest"), "hedging": ("tentative", "categorical"),
               "valence": ("cheerful", "gloomy"), "intensity": ("outburst", "flat"),
               "warmth": ("warm", "cold"), "playfulness": ("teasing", None), "softness": (None, "rough"),
               "swagger": ("arrogant", None)}
book.OUT = book.ROOT / "expandtraining/2606/outputs/tonetwin"
LABELS = book.ROOT / "expandtraining/2606/outputs/traits/genshin_train.jsonl"
TRIPLETS = book.ROOT / "expandtraining/2606/outputs/datasets/d-402f1c32b449/triplets.jsonl"
MAX_RATIO = 2.5  # short lines: a few added characters move the ratio a lot

REWRITE_SYSTEM = (
    "You rewrite single lines of game dialogue into a requested tone. Keep every piece of content: the same "
    "facts, names, events and claims, nothing added, nothing dropped. Keep the line's register (how formal, "
    "classical or colloquial it is), its person and address, and roughly its length. Change only the tone, and "
    "make the requested tone unmistakable. Write in the line's own language. Do not explain. Answer with a JSON "
    'array only: [{"id": <id>, "text": <rewrite>}, ...], one object per input item, same ids.')
POLISH_SYSTEM = book.POLISH_SYSTEM.replace("style rewrites", "tone rewrites of game dialogue lines").replace(
    "the requested style is strong", "the requested tone is strong, and the register and length stay close to the source")


def pick(per_speaker: int) -> None:
    labels = {}
    for line in open(LABELS, encoding="utf-8"):
        row = json.loads(line)
        labels[row["text"]] = {name: score for name, (score, _confidence) in row["scores"].items()}
    by_speaker: dict[str, list[str]] = {}
    for line in open(TRIPLETS, encoding="utf-8"):
        row = json.loads(line)
        if row["split"] == "train" and row["rule"] == "origin_vs_para" and row["anchor_text"] in labels:
            lines = by_speaker.setdefault(row["meta"]["speaker"], [])
            if row["anchor_text"] not in lines:
                lines.append(row["anchor_text"])
    rng, rows, used = random.Random(0), [], {}
    for speaker in sorted(by_speaker):
        lines = by_speaker[speaker][:]
        rng.shuffle(lines)
        for text in lines:
            scores = labels[text]
            options = []
            for trait, (up, down) in TONE_TRAITS.items():
                style = down if scores[trait] > 2 else up  # toward the other end
                if style and abs(scores[trait] - 2) >= 1.0:
                    options.append((abs(scores[trait] - 2), trait, style))
            if not options:
                continue
            # The least-used qualifying style, so a direct-speaking cast does not turn into mostly "tentative".
            _distance, trait, style = min(options, key=lambda option: (used.get(option[2], 0), -option[0]))
            used[style] = used.get(style, 0) + 1
            rows.append({"job_id": f"g{len(rows):05d}:{style}", "chunk_id": f"g{len(rows):05d}", "speaker": speaker,
                         "lang": "zh" if any("一" <= ch <= "鿿" for ch in text) else "en",
                         "source": text, "style": style, "before": scores[trait]})
            if sum(r["speaker"] == speaker for r in rows) == per_speaker:
                break
    book.OUT.mkdir(parents=True, exist_ok=True)
    (book.OUT / "jobs.jsonl").unlink(missing_ok=True)
    book.append("jobs.jsonl", rows)
    styles: dict[str, int] = {}
    for row in rows:
        styles[row["style"]] = styles.get(row["style"], 0) + 1
    print(f"{len(rows)} lines from {len(by_speaker)} speakers; styles {styles}")


def build_rewrite(batch: list[dict]) -> str:
    return json.dumps([{"id": job["job_id"], "tone": book.STYLES[job["style"]][2], "line": job["source"]}
                       for job in batch], ensure_ascii=False)


def build_polish(batch: list[dict]) -> str:
    return json.dumps([{"id": job["job_id"], "tone": book.STYLES[job["style"]][2], "source": job["source"],
                        "draft": job["draft"]} for job in batch], ensure_ascii=False)


def rewrite(limit: int) -> None:
    done = {row["job_id"] for row in book.read("rewrite.jsonl")}
    todo = [job for job in book.read("jobs.jsonl") if job["job_id"] not in done][:limit or None]
    book.batched_step("rewrite", todo, build_rewrite, REWRITE_SYSTEM)


def polish(limit: int) -> None:
    done = {row["job_id"] for row in book.read("polish.jsonl")}
    todo = [{**{k: v for k, v in row.items() if k != "reply"}, "draft": row["reply"].get("text", "")}
            for row in book.read("rewrite.jsonl") if row["job_id"] not in done and row["reply"].get("text")]
    book.batched_step("polish", todo[:limit or None], build_polish, POLISH_SYSTEM)


def verify(limit: int) -> None:
    """Scores the polished rewrites on all 17 traits; the source lines already have Jev scores."""
    done = {row["job_id"] for row in book.read("verify.jsonl")}
    todo = [row for row in book.read("polish.jsonl")
            if row["job_id"] not in done and row["reply"].get("ok") and row["reply"].get("text")][:limit or None]
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        results = list(pool.map(lambda text: book.judge(text, names=book.VERIFY_TRAITS),
                                [row["reply"]["text"] for row in todo]))
    rows = []
    for row, result in zip(todo, results):
        trait, direction, _ = book.STYLES[row["style"]]
        ratio = len(row["reply"]["text"]) / max(1, len(row["source"]))
        after = result["scores"][trait][0]
        keep = book.kept(direction, row["before"], after, 1.0) and 0.5 <= ratio <= MAX_RATIO
        rows.append({"job_id": row["job_id"], "speaker": row["speaker"], "lang": row["lang"], "style": row["style"],
                     "source": row["source"], "text": row["reply"]["text"], "scores": result["scores"],
                     "before": row["before"], "shift": direction * (after - row["before"]),
                     "length_ratio": round(ratio, 2), "tokens": result["input_tokens"], "kept": keep})
    book.append("verify.jsonl", rows)
    print(f"verify: {sum(r['kept'] for r in rows)}/{len(rows)} kept", flush=True)


def report() -> None:
    usage = book.read("usage.jsonl")
    for step in ("rewrite", "polish"):
        rows = [u for u in usage if u["step"] == step]
        print(f"{step}: {len(rows)} requests, {sum(u['prompt'] for u in rows) / 1e6:.2f}M prompt, "
              f"{sum(u['completion'] for u in rows) / 1e6:.2f}M completion tokens")
    verified = book.read("verify.jsonl")
    jev = sum(r["tokens"] for r in verified)
    print(f"verify: {sum(r['kept'] for r in verified)}/{len(verified)} kept; Jev {jev / 1e6:.2f}M tokens "
          f"(USD {jev / 1e6 * 0.042:.2f})")
    by_style: dict[str, list] = {}
    for row in verified:
        by_style.setdefault(row["style"], []).append(row)
    for style, rows in sorted(by_style.items()):
        print(f"  {style:12s} kept {sum(r['kept'] for r in rows)}/{len(rows)}  mean shift "
              f"{sum(r['shift'] for r in rows) / len(rows):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["pick", "rewrite", "polish", "verify", "report"])
    parser.add_argument("--per-speaker", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.step == "pick":
        pick(args.per_speaker)
    elif args.step == "verify":
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("TYPESAFE_API_KEY is not set")
        verify(args.limit)
    else:
        {"rewrite": rewrite, "polish": polish, "report": lambda _limit: report()}[args.step](args.limit)


if __name__ == "__main__":
    main()
