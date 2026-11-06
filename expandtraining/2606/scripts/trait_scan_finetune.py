"""Warm-start a saved frozen-head trait run and add Argon's scan on small-dev pairs to the Jev losses.

    uv run python expandtraining/2606/scripts/trait_scan_finetune.py [--steps 800] [--scan-weight 1.0]

The Jev line and pair losses stay on, so the 17 traits keep their meaning. The scan loss fits a learned scale and offset of the equal-weight
predicted change to the scan average of small-dev pairs. A few small-dev speakers are held out to watch overfitting. argon_test_key is the
development set here: it was sealed once and has been scored, so numbers on it are not a new sealed result.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import resolve_repo_path
from pairwise_voice_baseline import load_base, load_pairs, save
from trait_predictor import (NAMES, OUT_DIR, RATINGS_DIR, judge_dev, load_data, make_encode, make_fusion, make_predict, make_readout, predicted_change,
                             rank_gain, read_ratings, scan_targets)

START = "frozen_delta1.0_s42_attention_layers6-12-18-24-28"
HELD_SPEAKERS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--scan-weight", type=float, default=1.0)
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    data = load_data()
    small, rated = load_pairs()
    small = small[small["y_scan"].notna()].reset_index(drop=True)
    speakers = sorted(small["speaker"].unique())
    random.Random(42).shuffle(speakers)
    held = small["speaker"].isin(speakers[:HELD_SPEAKERS])
    fit, check = small[~held].reset_index(drop=True), small[held].reset_index(drop=True)
    key = pd.read_json(resolve_repo_path(f"{RATINGS_DIR}/argon_test_key.jsonl"), lines=True)
    dev = scan_targets(key, [json.loads(line) for line in open(resolve_repo_path(f"{RATINGS_DIR}/argon_test_answers.jsonl"), encoding="utf-8")])
    blind = read_ratings("voice_change_confirm_ratings.jsonl")
    confirm = pd.read_json(resolve_repo_path(f"{RATINGS_DIR}/voice_change_confirm_key.jsonl"), lines=True)
    fresh = confirm[(confirm["block"] == "fresh") & confirm["id"].isin(blind)].reset_index(drop=True)
    print(f"scan pairs {len(fit)}, held {len(check)} of {HELD_SPEAKERS} speakers, dev {len(dev)}", flush=True)

    start = resolve_repo_path(OUT_DIR) / START
    saved = json.loads((start / "dev.json").read_text(encoding="utf-8"))
    tokenizer, base, device = load_base()
    encoder = base.requires_grad_(False)
    head = make_readout("attention", base.config.hidden_size, torch.zeros(len(NAMES)), device, 0)
    head.load_state_dict(torch.load(start / "head.pt", map_location=device))
    fusion = make_fusion(tuple(saved["layers"]), device)
    fusion.load_state_dict(torch.load(start / "fusion.pt", map_location=device))
    encode = make_encode(tokenizer, encoder, head, fusion, device, True)
    predict = make_predict(encode, [encoder, head, fusion])
    change = lambda frame: predicted_change(predict(list(frame["A"])), predict(list(frame["B"])))  # noqa: E731

    begin = change(fit)
    slope, offset = np.polyfit(begin, fit["y_scan"].to_numpy(dtype=float), 1)
    scale = torch.nn.Parameter(torch.tensor(float(slope), device=device))
    shift = torch.nn.Parameter(torch.tensor(float(offset), device=device))
    optimizer = torch.optim.AdamW([*head.parameters(), *fusion.parameters(), scale, shift], lr=2e-4, weight_decay=0.05)
    edit = dev["changed"].to_numpy(dtype=float)

    def probe(step: int) -> dict:
        now = change(dev)
        row = {"step": step, "dev_scan": rank_gain(now, edit, dev["scan"].to_numpy(), boots=1000), "held_small_dev": round(float(spearmanr(change(check), check["y_scan"])[0]), 3),
               "fit_small_dev": round(float(spearmanr(change(fit), fit["y_scan"])[0]), 3)}
        print(json.dumps(row), flush=True)
        return row

    curve = [probe(0)]
    lines = list(data["train"].items())
    rows = fit[["A", "B", "y_scan"]].to_records(index=False).tolist()
    for step in range(1, args.steps + 1):
        chosen, paired, scanned = random.sample(lines, 16), random.sample(data["pairs"], 8), random.sample(rows, 8)
        texts = [t for t, _ in chosen] + [t for row in paired for t in row[:2]] + [t for row in scanned for t in row[:2]]
        y = torch.tensor([v for _, v in chosen] + [v for row in paired for v in row[2:]], dtype=torch.float32, device=device)
        predicted = encode(texts)
        jev, extra = predicted[:32], predicted[32:]
        loss = torch.nn.functional.mse_loss(jev, y) + torch.nn.functional.mse_loss(jev[16::2] - jev[17::2], y[16::2] - y[17::2])
        target = torch.tensor([row[2] for row in scanned], dtype=torch.float32, device=device)
        scan = torch.nn.functional.mse_loss(scale * (extra[0::2] - extra[1::2]).abs().mean(1) + shift, target)
        total = loss + args.scan_weight * scan
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"{time.strftime('%H:%M:%S')} step {step} jev {loss.item():.3f} scan {scan.item():.3f}", flush=True)
        if step % 200 == 0 or step == args.steps:
            curve.append(probe(step))
    out = resolve_repo_path(OUT_DIR) / f"frozen_taps_scan_w{args.scan_weight}_s42"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    torch.save(fusion.state_dict(), out / "fusion.pt")
    final = {"step": args.steps, "readout": "attention", "layers": saved["layers"], "frozen": True, "scorer_dim": 0, **judge_dev(predict, data)}
    (out / "dev.json").write_text(json.dumps(final), encoding="utf-8")
    now = change(dev)
    result = {"curve": curve, "jev_dev": final, "start_jev_dev": saved, "scan": rank_gain(now, edit, dev["scan"].to_numpy()), "plain": rank_gain(now, edit, dev["plain"].to_numpy())}
    small_rated = (rated["bin"] == "small").to_numpy()
    small_fresh = (fresh["bin"] == "small").to_numpy()
    voice = fresh["id"].map(blind).to_numpy(dtype=float)
    result["human"] = {"95_small": round(float(spearmanr(change(rated)[small_rated], rated["voice"][small_rated])[0]), 3),
                       "fresh_small": round(float(spearmanr(change(fresh)[small_fresh], voice[small_fresh])[0]), 3)}
    print("FINAL", json.dumps(result), flush=True)
    save({f"stage4b_w{args.scan_weight}": result})


if __name__ == "__main__":
    main()
