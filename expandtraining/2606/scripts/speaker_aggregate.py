"""Speaker retrieval from averaged lines: does a style vector find its speaker once several lines are pooled?

StyleVector (ACL 2025) averages line-minus-rewrite shifts over all of a user's texts; one line's shift is mostly
noise. Here each held-out speaker's real lines are split at random into two disjoint sets of k lines; the mean of
one (query) must find the same speaker's other mean (gallery) among the held-out speakers by cosine. Lines are
unit-normalised after removing the run's mean embedding. Averaged over random splits.

uv run python expandtraining/2606/scripts/speaker_aggregate.py outputs/runs/clean_nce_a_1gpu__nce_clean_twin__s42 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from report import heldout_speaker_set

REAL_RULES = ("same_speaker_real", "same_speaker_topic_matched")  # origin_vs_para holds paraphrases


def speaker_lines(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row indices of each held-out speaker's distinct real lines."""
    held = heldout_speaker_set(rows)
    real = rows[rows["rule"].isin(REAL_RULES) & rows["speaker"].isin(held)]
    real = real.drop_duplicates(["speaker", "text_preview"])
    return {speaker: group.index.to_numpy() for speaker, group in real.groupby("speaker")}


def aggregate_top1(embeddings: np.ndarray, lines: dict[str, np.ndarray], k: int, trials: int = 50,
                   seed: int = 0) -> tuple[float, int]:
    """(mean top-1 over random splits, speakers with at least 2k lines)."""
    x = embeddings.astype(np.float64) - embeddings.mean(0)
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)
    usable = {name: index for name, index in lines.items() if len(index) >= 2 * k}
    if len(usable) < 2:
        return float("nan"), len(usable)
    rng = np.random.default_rng(seed)
    hits = []
    for _ in range(trials):
        query, gallery = [], []
        for index in usable.values():
            pick = rng.permutation(index)
            query.append(x[pick[:k]].mean(0))
            gallery.append(x[pick[k:2 * k]].mean(0))
        query, gallery = np.stack(query), np.stack(gallery)
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
        hits.append(float(((query @ gallery.T).argmax(1) == np.arange(len(usable))).mean()))
    return float(np.mean(hits)), len(usable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories holding validation/{rows.parquet,embeddings.npy}")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--out", help="csv of all runs")
    args = parser.parse_args()
    table = []
    for run in args.runs:
        validation = Path(run) / "validation"
        if not (validation / "embeddings.npy").exists():
            continue
        rows = pd.read_parquet(validation / "rows.parquet")
        lines = speaker_lines(rows)
        embeddings = np.load(validation / "embeddings.npy", mmap_mode="r")
        result = {"run": Path(run).name}
        for k in args.ks:
            result[f"k{k}"], result[f"n{k}"] = aggregate_top1(np.asarray(embeddings), lines, k)
        (validation / "speaker_aggregate.json").write_text(json.dumps(result, indent=1))
        table.append(result)
        print(" ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in result.values()), flush=True)
    if args.out and table:
        pd.DataFrame(table).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
