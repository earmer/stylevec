"""Rate a frozen key's pairs with Argon: plain runs and the careful prompt, one line of talk on each side, resumable.

    uv run python expandtraining/2606/scripts/argon_scan.py <key.jsonl> <answers.jsonl> [--runs 1 2 careful] [--max-tokens 200000]

The key comes from GKEY in the environment and is never printed. --effort sets the careful run's reasoning effort.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / "data/genshin/genshin_v2.db"
BRIEF = ROOT / "data/human_ratings/voice_change_full_for_llms_v2.md"
URL = "http://127.0.0.1:23333/v1/chat/completions"
MODEL = "vertexai:gemini-4-argon"
SEED = 20261105
TALK_LINES = 1
BATCH = 40
CAREFUL = ("in the character's speech habits. Be much more careful and more sensitive than you would normally be: read A and B slowly "
           "as spoken lines, and give 0 or 1 only if this character truly sounds the same in B.")


def briefs() -> tuple[str, str]:
    text = BRIEF.read_text(encoding="utf-8")
    plain = text[:text.index("## Pair 1\n")].replace("# Voice change: 100 pairs", "# Voice change: {n} pairs")
    careful = plain.replace("in the character's speech habits.", CAREFUL)
    assert careful != plain
    return plain, careful


def parse_answers(text: str, size: int) -> dict[str, str]:
    got = dict(re.findall(r"(?m)^\s*(\d+)\s+(x|[0-3][ypn])\s*$", text))
    if sorted(map(int, got)) != list(range(1, size + 1)):
        raise ValueError(f"parsed {len(got)} of {size}")
    return got


def pair_text(n: int, row: dict, talk: list[tuple], lines: int = TALK_LINES) -> str:
    at = next(i for i, line in enumerate(talk) if line[0] == row["id"])
    out = [f"## Pair {n}", "", f"Speaker: {row['speaker']}", ""]
    before, after = talk[max(0, at - lines):at], talk[at + 1:at + 1 + lines]
    if before:
        out += ["Before:"] + [f"> {s or '—'}：{u}  " for _, s, u in before] + [""]
    out += [f"**A** {row['A']}", "", f"**B** {row['B']}", ""]
    if after:
        out += ["After:"] + [f"> {s or '—'}：{u}  " for _, s, u in after] + [""]
    return "\n".join(out)


def ask(prompt: str, effort: str) -> tuple[str, dict]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    if effort:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["GKEY"]})
    reply = json.load(urllib.request.urlopen(request, timeout=600))
    return reply["choices"][0]["message"]["content"], reply.get("usage") or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", type=Path)
    parser.add_argument("answers", type=Path)
    parser.add_argument("--runs", nargs="+", default=["1", "2", "careful"])
    parser.add_argument("--effort", default="", help="the careful run's reasoning effort; empty is the API's default")
    parser.add_argument("--max-tokens", type=int, default=200000, help="stop once this run has used this many")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.key, encoding="utf-8")]
    talks: dict[int, list[tuple]] = defaultdict(list)
    for id_, talk, speaker, text in sqlite3.connect(f"file:{DB}?mode=ro", uri=True).execute("select id,talk_id,speaker,origin_text from dialogues order by id"):
        talks[talk].append((id_, speaker, text))
    done = {(r["run"], r["id"]) for r in map(json.loads, open(args.answers, encoding="utf-8"))} if args.answers.exists() else set()
    plain, careful = briefs()
    spent = {"in": 0, "out": 0}
    for run in args.runs:
        todo = [row for row in rows if (run, row["id"]) not in done]
        random.Random(f"{SEED}/{run}").shuffle(todo)
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            head = careful if run == "careful" else plain
            prompt = head.format(n=len(batch)) + "\n".join(pair_text(i + 1, row, talks[row["talk_id"]]) for i, row in enumerate(batch))
            for attempt in range(8):
                try:
                    text, usage = ask(prompt, args.effort if run == "careful" else "")
                    got = parse_answers(text, len(batch))
                    break
                except Exception as error:  # 429s and short replies are routine here
                    print(run, "retry", attempt, type(error).__name__, str(error)[:100], flush=True)
                    time.sleep(20 * (attempt + 1))
            else:
                raise SystemExit(f"run {run}: gave up on a batch")
            with open(args.answers, "a", encoding="utf-8") as handle:
                for i, row in enumerate(batch):
                    handle.write(json.dumps({"run": run, "model": MODEL, "id": row["id"], "answer": got[str(i + 1)]}) + "\n")
            spent["in"] += usage.get("prompt_tokens", 0)
            spent["out"] += usage.get("completion_tokens", 0)
            print(run, start + len(batch), "of", len(todo), spent, flush=True)
            if spent["in"] + spent["out"] >= args.max_tokens:
                raise SystemExit(f"stopped at the token limit: {spent}")
    print("SCAN_DONE", spent, flush=True)


if __name__ == "__main__":
    main()
