"""Local page for rating voice change on (line, paraphrase) pairs; every click is saved to disk at once.

    uv run python expandtraining/2606/scripts/rate_voice_server.py build-extra
    uv run python expandtraining/2606/scripts/rate_voice_server.py build-confirm
    uv run python expandtraining/2606/scripts/rate_voice_server.py serve [--port 8765]

build-extra: freezes the pairs beyond the pre-registered 100 into voice_change_extra_key.jsonl: held-out speakers,
the same filter as the full set, minus the quick and full lines, ordered so that stopping anywhere keeps the full
set's edit-size mix. build-confirm: freezes the confirmation queue of prereg/trait_confirmation.md: fresh pairs of speakers
the trait predictor never saw, re-rates of the 95, and the pairs first rated with Gemini's score on screen. serve: the full 100 come first, then the extra pairs in sets of --set-size, in speaker and
conversation order; the page shows one pair at a time and jumps on when it is rated. The page gets the talk around each line, the speaker's other lines and the official
translations, never edit size, batch or an arm's score; --show-llm adds saved LLM answers, and each rating logs if
they were on screen. Ratings append to voice_change_ratings.jsonl; the latest row
for an id wins, so changing a rating keeps the history. --queue serves only one key file's pairs.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sqlite3
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[3]
RATINGS_DIR = ROOT / "data/human_ratings"
DB = ROOT / "data/genshin/genshin_v2.db"
MANIFEST = ROOT / "expandtraining/2606/outputs/runs/head_f_1gpu__data__s42/data_manifest.json"
PAGE = Path(__file__).with_name("rate_voice_page.html")
SEED = 20261105
BIN_WEIGHTS = {"small": 40, "mid": 30, "large": 20, "rewrite": 20}  # the mix make_ratings gave the full set
CONTEXT_LINES = 12
REFERENCE_LINES = 24
LLM_FILES = {"run": RATINGS_DIR / "voice_change_extra_llm_answers.jsonl",
             "careful": RATINGS_DIR / "voice_change_small_careful_answers.jsonl"}
TRAIT_DIR = ROOT / "expandtraining/2606/outputs/traits"
CONFIRM_MIX = {"small": 60, "mid": 12, "large": 4, "rewrite": 4}
CONFIRM_RETEST = {"small": 12, "other": 6}
CONFIRM_PER_SPEAKER = 3
CONFIRM_MIN_LINES = 20
NOT_A_VOICE = {"留言板"}  # a notice board
SPEAKER_ALIASES = {"留云借风真君": ("闲云",)}  # the database keeps one character under two names


def edit_bin(changed: float) -> str:
    return "small" if changed < 0.15 else "mid" if changed < 0.35 else "large" if changed < 0.6 else "rewrite"


def stratified_order(rows: list[dict], seed: int = SEED) -> list[dict]:
    """Interleave (bin, batch) cells by weight, so any prefix keeps the mix until a thin cell runs out."""
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[row["bin"], row["batch"]].append(row)
    ranked = []
    for (bin_name, _), cell in sorted(cells.items()):
        rng.shuffle(cell)
        ranked += [((i + 1) / (BIN_WEIGHTS[bin_name] / 2), rng.random(), row) for i, row in enumerate(cell)]
    return [row for _, _, row in sorted(ranked, key=lambda r: r[:2])]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def build_extra(out: Path) -> list[dict]:
    held = set(json.loads(MANIFEST.read_text(encoding="utf-8"))["speakers"][-20:])
    used = {row["A"] for name in ("quick", "full") for row in read_jsonl(RATINGS_DIR / f"voice_change_{name}_key.jsonl")}
    query = "select id,talk_id,speaker,origin_text,para_text,similarity from dialogues where dup_of is null order by id"
    rows, seen = [], set()
    for id_, talk, speaker, a, b, similarity in sqlite3.connect(DB).execute(query):
        if speaker not in held or not b or a == b or not 12 <= len(a) <= 60 or a in seen or a in used:
            continue
        seen.add(a)
        changed = 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        rows.append({"id": id_, "talk_id": talk, "speaker": speaker, "batch": "new" if similarity is None else "old",
                     "changed": round(changed, 3), "bin": edit_bin(changed), "A": a, "B": b})
    ordered = [{"item": n, **row} for n, row in enumerate(stratified_order(rows), 1)]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
    return ordered


def pick_fresh(pool: list[dict], mix: dict[str, int], per_speaker: int, seed: int = SEED) -> list[dict]:
    """Random pairs filling each bin's count, with at most per_speaker from one speaker across all bins."""
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    taken: dict[str, int] = defaultdict(int)
    left, picked = dict(mix), []
    for row in shuffled:
        if left.get(row["bin"], 0) and taken[row["speaker"]] < per_speaker:
            left[row["bin"]] -= 1
            taken[row["speaker"]] += 1
            picked.append(row)
    assert not any(left.values()), f"pool too thin: {left}"
    return picked


def pick_retests(full: list[dict], ratings: dict[int, dict], counts: dict[str, int], seed: int = SEED) -> list[dict]:
    """Re-rates of usable rated pairs, chosen by edit size and never by the rating they got."""
    usable = [{**row, "bin": edit_bin(row["changed"])} for row in full
              if row["id"] in ratings and ratings[row["id"]]["voice"] is not None and not ratings[row["id"]]["broken"]]
    rng = random.Random(seed)
    small = [row for row in usable if row["bin"] == "small"]
    other = [row for row in usable if row["bin"] != "small"]
    return rng.sample(small, counts["small"]) + rng.sample(other, counts["other"])


def build_confirm(out: Path) -> list[dict]:
    seen_texts = {json.loads(line)["text"] for name in ("genshin_train", "genshin_val")
                  for line in open(TRAIT_DIR / f"{name}.jsonl", encoding="utf-8")}
    for line in open(TRAIT_DIR / "trait_pairs_train.jsonl", encoding="utf-8"):
        row = json.loads(line)
        seen_texts |= {row["a"], row["b"]}
    full = read_jsonl(RATINGS_DIR / "voice_change_full_key.jsonl")
    extra = read_jsonl(RATINGS_DIR / "voice_change_extra_key.jsonl")
    small_dev = {row["id"] for row in read_jsonl(RATINGS_DIR / "voice_change_small_rule_labels.jsonl")}
    log = read_jsonl(RATINGS_DIR / "voice_change_ratings.jsonl")
    ratings = latest_ratings(log)
    query = "select id,talk_id,speaker,origin_text,para_text,similarity from dialogues where dup_of is null order by id"
    rows = list(sqlite3.connect(DB).execute(query))
    lines: dict[str, set] = defaultdict(set)
    for _, _, speaker, a, _, _ in rows:
        lines[speaker].add(a)
    seen_speakers = {speaker for speaker, said in lines.items() if said & seen_texts} | {row["speaker"] for row in full + extra}
    pool, used = [], set()
    for id_, talk, speaker, a, b, similarity in rows:
        if (not speaker or "?" in speaker or "？" in speaker or speaker in NOT_A_VOICE or speaker in seen_speakers or len(lines[speaker]) < CONFIRM_MIN_LINES
                or not b or a == b or not 12 <= len(a) <= 60 or a in used or b in seen_texts or id_ in small_dev or id_ in ratings):
            continue
        used.add(a)
        changed = 1 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
        pool.append({"id": id_, "talk_id": talk, "speaker": speaker, "batch": "new" if similarity is None else "old",
                     "changed": round(changed, 3), "bin": edit_bin(changed), "A": a, "B": b})
    seen_gemini = {row["id"] for row in ratings.values() if row["views"].get("llm")}
    picked = ([{**row, "block": "fresh"} for row in pick_fresh(pool, CONFIRM_MIX, CONFIRM_PER_SPEAKER)]
              + [{**row, "block": "retest_95"} for row in pick_retests(full, ratings, CONFIRM_RETEST)]
              + [{**row, "block": "retest_gemini"} for row in extra if row["id"] in seen_gemini])
    random.Random(SEED).shuffle(picked)
    ordered = [{**row, "item": n} for n, row in enumerate(picked, 1)]
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
    return ordered


def llm_answers(files: dict[str, Path]) -> dict[int, list[dict]]:
    """Answers per pair id, labelled by file and run."""
    answers: dict[int, list[dict]] = defaultdict(list)
    for name, path in files.items():
        for row in read_jsonl(path):
            label = f"run {row['run']}" if name == "run" else name
            answers[row["id"]].append({"label": label, "answer": row["answer"]})
    return answers


def latest_ratings(rows: list[dict]) -> dict[int, dict]:
    return {row["id"]: row for row in rows}


def clean_rating(body: dict, known_ids: set[int]) -> dict:
    """Raises ValueError on anything but a well-formed rating of a served pair."""
    id_, voice, meaning = body.get("id"), body.get("voice"), body.get("meaning")
    if id_ not in known_ids:
        raise ValueError("unknown id")
    if voice not in (None, 0, 1, 2, 3) or isinstance(voice, bool):
        raise ValueError("voice must be 0-3 or null")
    if meaning not in (None, "y", "p", "n"):
        raise ValueError("meaning must be y, p, n or null")
    note = body.get("note") or ""
    if not isinstance(note, str) or len(note) > 500:
        raise ValueError("note must be text up to 500 characters")
    views = body.get("views") or {}
    return {"id": id_, "voice": voice, "meaning": meaning, "broken": body.get("broken") is True, "note": note,
            "views": {k: views.get(k) is True for k in ("talk", "diff", "translations", "llm")}}


def diff_segments(a: str, b: str) -> tuple[list[list], list[list]]:
    left, right = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if i2 > i1:
            left.append([a[i1:i2], tag != "equal"])
        if j2 > j1:
            right.append([b[j1:j2], tag != "equal"])
    return left, right


class Store:
    """The frozen item sets, the database lookups around them and the append-only rating log."""

    def __init__(self, log: Path, set_size: int, queue: Path | None = None, show_llm: bool = False) -> None:
        self.log, self.lock, self.show_llm = log, threading.Lock(), show_llm
        extra = read_jsonl(queue or RATINGS_DIR / "voice_change_extra_key.jsonl")
        self.sets = [] if queue else [("Full 100", "full", read_jsonl(RATINGS_DIR / "voice_change_full_key.jsonl"))]
        for start in range(0, len(extra), set_size):
            label = f"{'Queue' if queue else 'Extra'} {start // set_size + 1}"
            self.sets.append((label, "extra", extra[start:start + set_size]))
        self.ids = {row["id"] for _, _, rows in self.sets for row in rows}
        self.lines = {row["A"] for _, _, rows in self.sets for row in rows}
        self.ratings = latest_ratings(read_jsonl(log))
        # talk_id and speaker have no index: one pass now saves a table scan per pair on every page load.
        self.talks: dict[int, list[tuple]] = defaultdict(list)
        self.by_speaker: dict[str, list[str]] = defaultdict(list)
        conn = self.connect()
        query = "select id,talk_id,speaker,origin_text,dup_of from dialogues order by id"
        for id_, talk, speaker, text, dup_of in conn.execute(query):
            self.talks[talk].append((id_, speaker, text))
            if dup_of is None and text and 6 <= len(text) <= 80 and text not in self.lines:
                self.by_speaker[speaker].append(text)
        conn.close()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    def summary(self) -> list[dict]:
        return [{"index": i, "label": label, "size": len(rows), "rated": sum(self.is_rated(r["id"]) for r in rows)}
                for i, (label, _, rows) in enumerate(self.sets)]

    def is_rated(self, id_: int) -> bool:
        row = self.ratings.get(id_)
        return bool(row) and (row["voice"] is not None or row["broken"])

    def references(self, speaker: str) -> list[str]:
        names = (speaker, *SPEAKER_ALIASES.get(speaker, ()))
        texts = list(dict.fromkeys(text for name in names for text in self.by_speaker[name]))
        random.Random(f"{SEED}/{speaker}").shuffle(texts)
        return texts[:REFERENCE_LINES]

    def pair(self, conn: sqlite3.Connection, row: dict, llm: dict[int, list[dict]]) -> dict:
        talk = self.talks[row["talk_id"]]
        at = next(i for i, line in enumerate(talk) if line[0] == row["id"])
        lines = lambda part: [{"speaker": s, "text": t} for _, s, t in part]
        en = conn.execute("select origin_text from dialogues_en where id=?", (row["id"],)).fetchone()
        ja = conn.execute("select origin_text from dialogues_ja where id=?", (row["id"],)).fetchone()
        a_diff, b_diff = diff_segments(row["A"], row["B"])
        return {"id": row["id"], "A": row["A"], "B": row["B"], "aDiff": a_diff, "bDiff": b_diff,
                "before": lines(talk[max(0, at - CONTEXT_LINES):at]), "after": lines(talk[at + 1:at + 1 + CONTEXT_LINES]),
                "en": en[0] if en else "", "ja": ja[0] if ja else "", "rating": self.ratings.get(row["id"]),
                "llm": llm.get(row["id"], [])}

    def set_payload(self, index: int) -> dict:
        label, tier, rows = self.sets[index]
        by_speaker: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_speaker[row["speaker"]].append(row)
        speakers = sorted(by_speaker)
        random.Random(f"{SEED}/{index}").shuffle(speakers)
        # Read on each load: the careful scan may still be writing.
        llm = llm_answers(LLM_FILES) if self.show_llm else {}
        conn = self.connect()
        try:
            groups = [{"speaker": " / ".join((speaker, *SPEAKER_ALIASES.get(speaker, ()))), "references": self.references(speaker),
                       "pairs": [self.pair(conn, row, llm) for row in sorted(by_speaker[speaker], key=lambda r: r["id"])]}
                      for speaker in speakers]
        finally:
            conn.close()
        return {"index": index, "label": label, "tier": tier, "groups": groups}

    def save(self, body: dict) -> dict:
        row = clean_rating(body, self.ids)
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self.lock:
            with open(self.log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.ratings[row["id"]] = row
        return {"saved": row["id"], "sets": self.summary()}


def make_handler(store: Store) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: bytes, kind: str = "application/json; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, payload: dict | list, status: int = 200) -> None:
            self.reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/sets":
                self.reply_json(store.summary())
            elif url.path == "/api/set":
                try:
                    index = int(parse_qs(url.query).get("n", ["0"])[0])
                    assert 0 <= index < len(store.sets)
                except (ValueError, AssertionError):
                    self.reply_json({"error": "no such set"}, 404)
                    return
                self.reply_json(store.set_payload(index))
            else:
                self.reply_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            # A JSON content type keeps other sites' plain form posts out of the rating log.
            if self.path != "/api/rate" or "application/json" not in self.headers.get("Content-Type", ""):
                self.reply_json({"error": "not found"}, 404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self.reply_json(store.save(body))
            except (ValueError, AttributeError) as error:
                self.reply_json({"error": str(error)}, 400)

        def log_message(self, *args) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-extra")
    build.add_argument("--out", type=Path, default=RATINGS_DIR / "voice_change_extra_key.jsonl")
    confirm = sub.add_parser("build-confirm")
    confirm.add_argument("--out", type=Path, default=RATINGS_DIR / "voice_change_confirm_key.jsonl")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--set-size", type=int, default=50)
    serve.add_argument("--log", type=Path, default=RATINGS_DIR / "voice_change_ratings.jsonl")
    serve.add_argument("--queue", type=Path, help="serve only this key file's pairs")
    serve.add_argument("--show-llm", action="store_true", help="show saved LLM answers; ratings then are not blind")
    args = parser.parse_args()
    if args.command == "build-extra":
        print(f"{args.out}: {len(build_extra(args.out))} pairs")
        return
    if args.command == "build-confirm":
        print(f"{args.out}: {len(build_confirm(args.out))} pairs")
        return
    store = Store(args.log, args.set_size, args.queue, args.show_llm)
    print(f"http://127.0.0.1:{args.port}  {len(store.ids)} pairs, {len(store.ratings)} rated, log {args.log}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
