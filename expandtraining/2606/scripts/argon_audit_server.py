"""Side-by-side audit of the pairs where the trait model and Argon disagree most; which side is which stays hidden.

    uv run python expandtraining/2606/scripts/argon_audit_server.py build
    uv run python expandtraining/2606/scripts/argon_audit_server.py serve [--port 8767]
    uv run python expandtraining/2606/scripts/argon_audit_server.py score

build: pairs the pairs the model ranks furthest above Argon with those Argon ranks furthest above the model, at similar edit size. serve: the rater
picks the line that changes voice more, or neither, or marks the round broken when an edited line is faulty. score: who the rater sided with. See prereg/trait_argon_test.md.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/human_ratings"
DB = ROOT / "data/genshin/genshin_v2.db"
AUDIT_KEY, CHOICES = D / "argon_audit_key.jsonl", D / "argon_audit_choices.jsonl"
ARM = "lora_r8_delta1.0_s42_attention"
ROUNDS = 25
SHORTLIST = 60
SEED = 20261105


def match_rounds(scores: pd.DataFrame, rounds: int = ROUNDS, shortlist: int = SHORTLIST) -> list[tuple[int, int]]:
    """(model-high id, Argon-high id) pairs: largest rank gaps first, each taking the closest unused edit size on the other side."""
    gap = scores[ARM].rank(pct=True) - scores["scan"].rank(pct=True)
    ordered = scores.assign(gap=gap).sort_values("gap")
    argon_high = ordered.head(shortlist).to_dict("records")
    model_high = ordered.tail(shortlist).iloc[::-1].to_dict("records")
    matched = []
    for row in model_high:
        if len(matched) == rounds or not argon_high:
            break
        other = min(argon_high, key=lambda r: abs(r["changed"] - row["changed"]))
        argon_high.remove(other)
        matched.append((row["id"], other["id"]))
    return matched


def segments(a: str, b: str) -> list[list]:
    """B's text as [piece, changed] runs; a deletion shows as a marked gap."""
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append([b[j1:j2], False])
        elif j2 > j1:
            out.append([b[j1:j2], True])
        else:
            out.append(["∅", True])
    return out


def build() -> int:
    scores = pd.read_json(D / "argon_test_scores.jsonl", lines=True)
    rng = random.Random(SEED)
    rows = [{"round": n, "model_high": m, "argon_high": a, "model_left": rng.random() < .5}
            for n, (m, a) in enumerate(match_rounds(scores), 1)]
    rng.shuffle(rows)
    AUDIT_KEY.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


def tally(rows: list[dict], choices: dict[int, str]) -> dict:
    model = argon = neither = broken = 0
    for row in rows:
        choice = choices.get(row["round"])
        if choice == "neither":
            neither += 1
        elif choice == "broken":
            broken += 1
        elif choice:
            model += (choice == "left") == row["model_left"]
            argon += (choice == "left") != row["model_left"]
    p = binomtest(model, model + argon).pvalue if model + argon else None
    return {"rounds": len(rows), "sided_with_model": model, "sided_with_argon": argon, "neither": neither, "broken": broken, "two_sided_p": p}


def side(pair: dict, conn: sqlite3.Connection) -> dict:
    talk = conn.execute("select id,speaker,origin_text from dialogues where talk_id=? order by id", (pair["talk_id"],)).fetchall()
    at = next(i for i, line in enumerate(talk) if line[0] == pair["id"])
    around = lambda part: [f"{s or '—'}：{t}" for _, s, t in part]  # noqa: E731
    return {"speaker": pair["speaker"], "A": pair["A"], "B": segments(pair["A"], pair["B"]),
            "before": around(talk[max(0, at - 1):at]), "after": around(talk[at + 1:at + 2])}


def payload() -> list[dict]:
    pairs = {r["id"]: r for r in map(json.loads, open(D / "argon_test_key.jsonl", encoding="utf-8"))}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    out = []
    for line in open(AUDIT_KEY, encoding="utf-8"):
        row = json.loads(line)
        left, right = (row["model_high"], row["argon_high"]) if row["model_left"] else (row["argon_high"], row["model_high"])
        out.append({"round": row["round"], "left": side(pairs[left], conn), "right": side(pairs[right], conn)})
    return out


def read_choices() -> dict[int, str]:
    if not CHOICES.exists():
        return {}
    return {r["round"]: r["choice"] for r in map(json.loads, open(CHOICES, encoding="utf-8"))}


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>Voice Audit</title><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#fff;--ink:#222;--mute:#666;--line:#ddd;--mark:#ffe08a}@media(prefers-color-scheme:dark){:root{--bg:#1c1c1e;--ink:#eee;--mute:#aaa;--line:#444;--mark:#7a5c00}}
body{margin:0 auto;max-width:1000px;padding:16px;font:17px/1.6 system-ui;background:var(--bg);color:var(--ink)}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.row{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:8px;padding:12px}.mute{color:var(--mute);font-size:14px}mark{background:var(--mark);color:inherit}button{font:inherit;padding:8px 14px;margin:12px 8px 0 0;cursor:pointer}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px}</style>
<p id="top"></p><p>Which edit changes the speaker's voice more? <kbd>1</kbd> left <kbd>2</kbd> right <kbd>0</kbd> neither or can't tell <kbd>x</kbd> an edited line is broken <kbd>←</kbd> back</p><div class="row" id="cards"></div><div><button data-c="left">1 Left</button><button data-c="right">2 Right</button><button data-c="neither">0 Neither</button><button data-c="broken">x Broken</button><button id="back">← Back</button></div>
<script>let rounds=[],picks={},at=0;const $=i=>document.getElementById(i);
const card=(s,n)=>{const d=document.createElement("div");d.className="card";const p=(t,c)=>{const e=document.createElement("p");e.textContent=t;if(c)e.className=c;d.append(e);return e};p(n+" · "+s.speaker,"mute");s.before.forEach(t=>p(t,"mute"));p("A "+s.A);const b=p("B ");s.B.forEach(([t,m])=>{const e=document.createElement(m?"mark":"span");e.textContent=t;b.append(e)});s.after.forEach(t=>p(t,"mute"));return d};
function draw(){const done=Object.keys(picks).length;if(at>=rounds.length){const open=rounds.findIndex(r=>!picks[r.round]);if(open<0){$("top").textContent="Done: "+done+" of "+rounds.length+" saved.";$("cards").replaceChildren();return}at=open}const r=rounds[at];$("top").textContent="Round "+(at+1)+" of "+rounds.length+" · "+done+" saved"+(picks[r.round]?" · you chose "+picks[r.round]:"");$("cards").replaceChildren(card(r.left,"Left"),card(r.right,"Right"));scrollTo(0,0)}
async function pick(c){if(at>=rounds.length)return;const r=rounds[at];const x=await fetch("/api/choose",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({round:r.round,choice:c})});if(!x.ok){$("top").textContent="NOT SAVED: is the server running?";return}picks[r.round]=c;at++;draw()}
document.querySelectorAll("[data-c]").forEach(b=>b.onclick=()=>pick(b.dataset.c));$("back").onclick=()=>{at=Math.max(0,at-1);draw()};addEventListener("keydown",e=>{const m={"1":"left","2":"right","0":"neither","x":"broken"}[e.key];if(m)pick(m);if(e.key==="ArrowLeft"){at=Math.max(0,at-1);draw()}});
(async()=>{const d=await(await fetch("/api/rounds")).json();rounds=d.rounds;picks=d.picks;at=rounds.findIndex(r=>!picks[r.round]);if(at<0)at=rounds.length;draw()})();</script></html>"""


def serve(port: int) -> None:
    rounds = payload()
    known = {r["round"] for r in rounds}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body: bytes, kind: str = "application/json; charset=utf-8", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self.reply(PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/rounds":
                self.reply(json.dumps({"rounds": rounds, "picks": read_choices()}, ensure_ascii=False).encode())
            else:
                self.reply(b"{}", status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/choose" or "application/json" not in self.headers.get("Content-Type", ""):
                self.reply(b"{}", status=404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if body.get("round") not in known or body.get("choice") not in ("left", "right", "neither", "broken"):
                self.reply(b"{}", status=400)
                return
            with open(CHOICES, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"round": body["round"], "choice": body["choice"]}) + "\n")
            self.reply(b"{}")

        def log_message(self, *args) -> None:
            pass

    print(f"http://127.0.0.1:{port}  {len(rounds)} rounds", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "serve", "score"])
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    if args.command == "build":
        print(build(), "rounds")
    elif args.command == "serve":
        serve(args.port)
    else:
        print(tally([json.loads(line) for line in open(AUDIT_KEY, encoding="utf-8")], read_choices()))


if __name__ == "__main__":
    main()
