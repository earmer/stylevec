"""Rewrite book chunks into named styles, polish them, and keep only rewrites Jev verifies.

    uv run python expandtraining/2606/scripts/book_style_rewrite.py chunks  --zh 1500 --en 300
    uv run python expandtraining/2606/scripts/book_style_rewrite.py rewrite [--limit N]
    uv run python expandtraining/2606/scripts/book_style_rewrite.py polish  [--limit N]
    uv run python expandtraining/2606/scripts/book_style_rewrite.py verify  [--limit N]
    uv run python expandtraining/2606/scripts/book_style_rewrite.py report

chunks: sentence-aligned 60-150 character (Chinese) or 120-320 character (English) chunks from
data/bookraw/books_cleaned.parquet, at most a few per work, each assigned STYLES_PER_CHUNK styles at random.
rewrite / polish: batched JSON requests to the paraphrase proxy of genshin/conf.json (its key is read from
the file and never printed). Polish compares each rewrite with its source, fixes meaning drift or weak style,
or rejects it. verify: Jev scores source and polished rewrite on the style traits; a rewrite is kept when its
target trait moved >= MIN_SHIFT levels the right way and the length ratio is sane. Every step appends to its
own JSONL under outputs/bookstyle/, skips what is done, and logs token usage, so reruns resume.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traits import TRAITS, judge

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "expandtraining/2606/outputs/bookstyle"
BOOKS = ROOT / "data/bookraw/books_cleaned.parquet"
CONF = ROOT / "genshin/conf.json"
STYLES_PER_CHUNK = 3
MIN_SHIFT = 1.0
MAX_RATIO = 2.0


def kept(direction: int, before: float, after: float, ratio: float) -> bool:
    """The target trait moved >= MIN_SHIFT the right way, or reached its end of the 0..4 scale (>= 3 or <= 1)
    with at least half that shift, which a source already near that end cannot exceed; and the length is sane."""
    shift = direction * (after - before)
    at_end = after >= 3.0 if direction > 0 else after <= 1.0
    return (shift >= MIN_SHIFT or (at_end and shift >= MIN_SHIFT / 2)) and 0.5 <= ratio <= MAX_RATIO
BATCH = 12

# name: (trait, direction, instruction). The trait is the Jev trait (src/traits.py) the style must move.
STYLES: dict[str, tuple[str, int, str]] = {
    "ceremonial": ("formality", +1, "solemn, ceremonial or official register, as in a proclamation or formal address"),
    "casual_chat": ("formality", -1, "very casual everyday chat between friends: relaxed, clipped, with casual particles"),
    "classical": ("archaic", +1, "classical literary language (文言 for Chinese; archaic diction such as thee/thou, 'tis for English)"),
    "trendy": ("archaic", -1, "contemporary, slightly trendy modern language of a young speaker today"),
    "spontaneous": ("spokenness", +1, "spontaneous speech: fillers, hesitations, self-corrections, trailing off, fragments"),
    "ornate": ("ornament", +1, "heavily ornamented prose: imagery, parallelism, metaphor, idioms"),
    "terse": ("verbosity", -1, "terse and curt: as few words as possible, no qualifiers"),
    "childlike": ("age_voice", +1, "the voice of a cute young child: simple words, reduplication, endearing particles"),
    "elderly_sage": ("age_voice", -1, "the voice of a venerable old sage: measured, old-fashioned self-address and sayings"),
    "arrogant": ("swagger", +1, "an arrogant, boastful speaker who talks themselves up (e.g. 本大爷 / 'the great me')"),
    "humble": ("deference", +1, "a humble, self-lowering speaker addressing a superior (e.g. 在下, 小的 / 'your humble servant')"),
    "warm": ("warmth", +1, "a very warm, affectionate, caring speaker"),
    "cold": ("warmth", -1, "a cold, aloof, dismissive speaker"),
    "rough": ("softness", -1, "a rough, gruff, coarse speaker"),
    "teasing": ("playfulness", +1, "a playful, teasing, mischievous speaker"),
    "theatrical": ("theatricality", +1, "theatrical, grandiloquent stage declamation"),
    "deadpan": ("theatricality", -1, "flat deadpan understatement"),
    "slang": ("nonstandard", +1, "heavy slang, internet speak or dialect words"),
    # Tone styles: the traits a neutral paraphrase keeps (line-vs-paraphrase r 0.8-0.9), so only an explicit
    # rewrite gives a same-content contrast on them. Not drawn for book chunks (make_chunks uses CHUNK_STYLES).
    "sarcastic": ("irony", +1, "openly sarcastic and mocking: say it with a sneer, ironic praise, eye-rolling asides"),
    "earnest": ("irony", -1, "completely earnest and heartfelt, with no irony or sarcasm at all"),
    "tentative": ("hedging", +1, "tentative and evasive: hedge everything (maybe, perhaps, I guess / 或许, 可能, 吧), never commit"),
    "categorical": ("hedging", -1, "absolutely categorical: flat assertions and commands, no softening or hedges"),
    "cheerful": ("valence", +1, "exuberant and delighted, bubbling with positive feeling"),
    "gloomy": ("valence", -1, "bitter, gloomy or despairing"),
    "outburst": ("intensity", +1, "an emotional outburst: very intense, exclamations, repetition"),
    "flat": ("intensity", -1, "flat and emotionless, detached, no feeling in the delivery"),
}
TONE_STYLES = ["sarcastic", "earnest", "tentative", "categorical", "cheerful", "gloomy", "outburst", "flat"]
CHUNK_STYLES = [name for name in STYLES if name not in TONE_STYLES]
# Every trait is scored, so a rewrite's off-target drift is measured too; Jev bills the questions as input,
# so this is ~30% more per request than scoring only the styled traits.
VERIFY_TRAITS = list(TRAITS)

REWRITE_SYSTEM = (
    "You rewrite short passages into a requested speaking or writing style. Keep every piece of content: the same "
    "events, facts, names, numbers and claims, nothing added, nothing dropped, roughly the same length (0.6x to 1.8x). "
    "Change only the style, and make the requested style unmistakable. Write in the passage's own language "
    "(Simplified Chinese for Chinese passages, English for English ones). Do not explain. Answer with a JSON array "
    'only: [{"id": <id>, "text": <rewrite>}, ...], one object per input item, same ids.')
POLISH_SYSTEM = (
    "You are an exacting editor checking style rewrites. For each item you get the source passage, the requested "
    "style, and a draft rewrite. Check (1) content: every fact, name, event and claim of the source is present and "
    "nothing new is invented (remove any framing, address, moral or commentary the source does not have); (2) style: the requested style is strong and consistent; (3) fluency: it reads like a "
    "natural native text in the source's language. If the draft passes, return it unchanged or lightly polished. If "
    "it can be fixed, return the fixed text. If it cannot (e.g. the style would require changing the content), "
    'reject it. Answer with a JSON array only: [{"id": <id>, "ok": true|false, "text": <final text or "">}, ...].')


def proxy(system: str, user: str, retries: int = 6) -> tuple[str, dict]:
    api = json.load(open(CONF, encoding="utf-8"))["api"]
    body = json.dumps({"model": api["model"], "temperature": 0.7, "max_tokens": 32768,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    for attempt in range(retries):
        try:
            request = urllib.request.Request(api["base_url"], body, {"Content-Type": "application/json",
                                                                     "Authorization": "Bearer " + api["api_key"]})
            reply = json.load(urllib.request.urlopen(request, timeout=300))
            return reply["choices"][0]["message"]["content"], reply.get("usage", {})
        except Exception:  # noqa: BLE001 - rate limits and dropped connections; the caller logs a final failure
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))
    raise AssertionError("unreachable")


def parse_array(content: str) -> list[dict]:
    match = re.search(r"\[.*\]", content, re.S)
    return json.loads(match.group(0)) if match else []


def read(name: str) -> list[dict]:
    path = OUT / name
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def append(name: str, rows: list[dict]) -> None:
    with open(OUT / name, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_usage(step: str, usage: dict) -> None:
    append("usage.jsonl", [{"step": step, "prompt": usage.get("prompt_tokens", 0),
                            "completion": usage.get("completion_tokens", 0), "time": time.time()}])


def split_chunks(text: str, lang: str) -> list[str]:
    low, high = (60, 150) if lang == "zh" else (120, 320)
    sentences = re.findall(r"[^。！？!?…]+[。！？!?…”」』]*" if lang == "zh" else r"[^.!?]+[.!?\"”']*", text)
    chunks, current = [], ""
    noise = re.compile(r"\[\d+\]|〔\d+〕|［\d+］|【\d+】|\(\d+\)|（\d+）|注[:：〕]|\d{3,}|http")
    for sentence in (s.strip() for s in sentences):
        if not sentence:
            continue
        current = f"{current}{'' if lang == 'zh' else ' '}{sentence}".strip()
        if len(current) >= low:
            if len(current) <= high and not noise.search(current):  # footnote markers, editorial notes
                chunks.append(current)
            current = ""
    return chunks


def make_chunks(zh: int, en: int) -> None:
    import pandas as pd
    books = pd.read_parquet(BOOKS)
    rng = random.Random(0)
    rows = []
    for lang, total in (("zh", zh), ("en", en)):
        part = books[books["lang"] == lang]
        authors = sorted(part["author"].unique())  # one-work modern authors would drown under many small classics
        per_work = max(1, 2 * total // len(authors) + 1)  # small authors run short
        for author in authors:
            passages = part[part["author"] == author]
            picked = []
            for index in rng.sample(range(len(passages)), min(len(passages), 4 * per_work)):
                options = split_chunks(passages.iloc[index]["text"], lang)
                if options:
                    picked.append((passages.iloc[index], rng.choice(options)))
                if len(picked) == per_work:
                    break
            for row, chunk in picked:
                rows.append({"lang": lang, "author": author, "work": row["work"], "source": chunk})
        rows_lang = [r for r in rows if r["lang"] == lang]
        rng.shuffle(rows_lang)
        rows = [r for r in rows if r["lang"] != lang] + rows_lang[:total]
    names = CHUNK_STYLES
    for number, row in enumerate(rows):
        row["chunk_id"] = f"c{number:05d}"
        row["styles"] = rng.sample(names, STYLES_PER_CHUNK)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chunks.jsonl").unlink(missing_ok=True)
    append("chunks.jsonl", rows)
    print(f"{len(rows)} chunks ({sum(r['lang'] == 'zh' for r in rows)} zh) from {len({r['work'] for r in rows})} works")


def batched_step(step: str, todo: list[dict], build, system: str, workers: int = 4) -> None:
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

    def run(batch: list[dict]) -> list[dict]:
        content, usage = proxy(system, build(batch))
        log_usage(step, usage)
        by_id = {str(item.get("id")): item for item in parse_array(content) if isinstance(item, dict)}
        return [{**job, "reply": by_id[job["job_id"]]} for job in batch if job["job_id"] in by_id]

    done = 0
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run, batch) for batch in batches]):
            try:
                rows = future.result()
            except Exception as error:  # noqa: BLE001 - a failed batch is retried by the next rerun
                print(f"{step}: batch failed: {error}", flush=True)
                continue
            append(f"{step}.jsonl", rows)
            done += len(rows)
    print(f"{step}: {done}/{len(todo)} written", flush=True)


def rewrite(limit: int) -> None:
    done = {row["job_id"] for row in read("rewrite.jsonl")}
    jobs = [{"job_id": f"{c['chunk_id']}:{style}", "chunk_id": c["chunk_id"], "lang": c["lang"], "style": style,
             "source": c["source"]} for c in read("chunks.jsonl") for style in c["styles"]]
    todo = [job for job in jobs if job["job_id"] not in done][:limit or None]

    def build(batch: list[dict]) -> str:
        return json.dumps([{"id": job["job_id"], "style": STYLES[job["style"]][2], "passage": job["source"]}
                           for job in batch], ensure_ascii=False)
    batched_step("rewrite", todo, build, REWRITE_SYSTEM)


def polish(limit: int) -> None:
    done = {row["job_id"] for row in read("polish.jsonl")}
    todo = [{key: row[key] for key in ("job_id", "chunk_id", "lang", "style", "source")} | {"draft": row["reply"]["text"]}
            for row in read("rewrite.jsonl") if row["job_id"] not in done and row["reply"].get("text")][:limit or None]

    def build(batch: list[dict]) -> str:
        return json.dumps([{"id": job["job_id"], "style": STYLES[job["style"]][2], "source": job["source"],
                            "draft": job["draft"]} for job in batch], ensure_ascii=False)
    batched_step("polish", todo, build, POLISH_SYSTEM)


def verify(limit: int) -> None:
    done = {row["job_id"] for row in read("verify.jsonl")}
    sources = {row["chunk_id"]: row for row in read("verify_sources.jsonl")}
    # Whole chunks in chunk order, so a limit keeps every style of the chunks it covers (one source score each).
    todo = sorted((row for row in read("polish.jsonl")
                   if row["job_id"] not in done and row["reply"].get("ok") and row["reply"].get("text")),
                  key=lambda row: row["job_id"])[:limit or None]
    need = {row["chunk_id"]: row["source"] for row in todo if row["chunk_id"] not in sources}

    def score(text: str) -> dict:
        return judge(text, names=VERIFY_TRAITS)
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        for chunk_id, result in zip(need, pool.map(score, need.values())):
            sources[chunk_id] = {"chunk_id": chunk_id, "scores": result["scores"], "tokens": result["input_tokens"]}
            append("verify_sources.jsonl", [sources[chunk_id]])
        results = list(pool.map(score, [row["reply"]["text"] for row in todo]))
    rows = []
    for row, result in zip(todo, results):
        trait, direction, _ = STYLES[row["style"]]
        before, after = sources[row["chunk_id"]]["scores"][trait][0], result["scores"][trait][0]
        ratio = len(row["reply"]["text"]) / max(1, len(row["source"]))
        rows.append({"job_id": row["job_id"], "chunk_id": row["chunk_id"], "lang": row["lang"], "style": row["style"],
                     "source": row["source"], "text": row["reply"]["text"], "scores": result["scores"],
                     "shift": direction * (after - before), "length_ratio": round(ratio, 2), "tokens": result["input_tokens"],
                     "before": before, "kept": kept(direction, before, after, ratio)})
    append("verify.jsonl", rows)
    print(f"verify: {sum(r['kept'] for r in rows)}/{len(rows)} kept", flush=True)


def report() -> None:
    usage = read("usage.jsonl")
    for step in ("rewrite", "polish"):
        rows = [u for u in usage if u["step"] == step]
        print(f"{step}: {len(rows)} requests, {sum(u['prompt'] for u in rows) / 1e6:.2f}M prompt, "
              f"{sum(u['completion'] for u in rows) / 1e6:.2f}M completion tokens")
    polished = read("polish.jsonl")
    print(f"polish: {sum(bool(r['reply'].get('ok')) for r in polished)}/{len(polished)} ok")
    sources = {row["chunk_id"]: row for row in read("verify_sources.jsonl")}
    verified = read("verify.jsonl")
    for row in verified:
        trait, direction, _ = STYLES[row["style"]]
        row["kept"] = kept(direction, sources[row["chunk_id"]]["scores"][trait][0], row["scores"][trait][0],
                           row["length_ratio"])
    jev = sum(r["tokens"] for r in verified) + sum(r["tokens"] for r in read("verify_sources.jsonl"))
    print(f"verify: {sum(r['kept'] for r in verified)}/{len(verified)} kept; Jev {jev / 1e6:.2f}M tokens "
          f"(USD {jev / 1e6 * 0.042:.2f})")
    by_style: dict[str, list] = {}
    for row in verified:
        by_style.setdefault(row["style"], []).append(row)
    for style, rows in sorted(by_style.items()):
        print(f"  {style:13s} kept {sum(r['kept'] for r in rows)}/{len(rows)}  mean shift "
              f"{sum(r['shift'] for r in rows) / len(rows):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["chunks", "rewrite", "polish", "verify", "report"])
    parser.add_argument("--zh", type=int, default=1500)
    parser.add_argument("--en", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="0 = all; verify defaults to 3000 (Jev budget)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.step == "chunks":
        make_chunks(args.zh, args.en)
    elif args.step == "verify":
        if not os.environ.get("TYPESAFE_API_KEY"):
            sys.exit("TYPESAFE_API_KEY is not set")
        verify(args.limit or 3000)
    else:
        {"rewrite": rewrite, "polish": polish, "report": lambda _limit: report()}[args.step](args.limit)


if __name__ == "__main__":
    main()
