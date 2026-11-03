"""Turn judged trait scores of (line, paraphrase) pairs into train.twin_weight_file weights.

    uv run python expandtraining/2606/scripts/twin_style_delta.py <labels.jsonl> <pairs dataset triplets.jsonl> \
        <out.jsonl> [--low 0.1] [--high 0.75]

labels.jsonl: scripts/jev_label.py output (one {"text", "scores"} row per text, any judge). Each training pair
(anchor, meta twin_text) with both sides labelled gets delta = mean |score difference| over the traits, and
weight = (delta - q_low) / (q_high - q_low) clipped to [0, 1], with q the deltas' quantiles: the most
near-copy pairs are not pushed at all, the most restyled ones fully. A mild surface gate,
clip((d - 0.02) / 0.13, 0, 1) with d = 1 - difflib ratio, also scales down pairs that are character-for-character
almost the same (under 15% changed), whatever the judge says: there a trait delta is mostly judge noise. --mean rescales the weights to that mean
(1.0: the same total push as unweighted twins, only redistributed), so an arm compares where the push goes, not
how much. Unlabelled pairs then take train.twin_weight_default = --mean.
"""
from __future__ import annotations

import argparse
import difflib
import json

import numpy as np


def style_delta(a: dict[str, list], b: dict[str, list]) -> float:
    shared = [name for name in a if name in b]
    return float(np.mean([abs(float(a[name][0]) - float(b[name][0])) for name in shared]))


def surface_gate(a: str, b: str, floor: float = 0.02, span: float = 0.13) -> float:
    changed = 1.0 - difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    return float(np.clip((changed - floor) / span, 0.0, 1.0))


def weights(deltas: np.ndarray, low: float, high: float) -> tuple[np.ndarray, float, float]:
    q_low, q_high = (float(np.quantile(deltas, q)) for q in (low, high))
    return np.clip((deltas - q_low) / max(q_high - q_low, 1e-9), 0.0, 1.0), q_low, q_high


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels")
    parser.add_argument("triplets")
    parser.add_argument("out")
    parser.add_argument("--low", type=float, default=0.1)
    parser.add_argument("--high", type=float, default=0.75)
    parser.add_argument("--mean", type=float, default=0.0, help="rescale weights to this mean (0 = keep [0, 1])")
    args = parser.parse_args()
    scores = {}
    for line in open(args.labels, encoding="utf-8"):
        row = json.loads(line)
        scores[row["text"]] = row["scores"]
    pairs: dict[tuple[str, str], None] = {}
    for line in open(args.triplets, encoding="utf-8"):
        row = json.loads(line)
        twin = row.get("meta", {}).get("twin_text")
        if twin and row["anchor_text"] in scores and twin in scores:
            pairs.setdefault((row["anchor_text"], twin))
    keys = list(pairs)
    deltas = np.array([style_delta(scores[a], scores[b]) for a, b in keys])
    w, q_low, q_high = weights(deltas, args.low, args.high)
    gates = np.array([surface_gate(a, b) for a, b in keys])
    w = w * gates
    if args.mean > 0:
        w = w * (args.mean / w.mean())
    with open(args.out, "w", encoding="utf-8") as handle:
        for (text, twin), delta, weight in zip(keys, deltas, w):
            handle.write(json.dumps({"text": text, "twin_text": twin, "delta": round(float(delta), 4),
                                     "weight": round(float(weight), 4)}, ensure_ascii=False) + "\n")
    print(f"{len(keys)} pairs; delta quantiles 10/25/50/75/90: "
          f"{np.round(np.quantile(deltas, [0.1, 0.25, 0.5, 0.75, 0.9]), 3).tolist()}; "
          f"q_low {q_low:.3f} q_high {q_high:.3f}; mean weight {w.mean():.3f}, max {w.max():.3f}; "
          f"zero {np.mean(w == 0):.2f}; surface gate < 1 on {np.mean(gates < 1):.2f}")


if __name__ == "__main__":
    main()
