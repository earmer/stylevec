"""Which embedding dimensions carry the line leak, and which carry speaker identity?

Random masking: each trial zeroes a random half of the dimensions and rescores held-out top-1 (macro) and line
leak. A dimension's effect on a metric is its mean with the dimension kept minus its mean with it dropped
(over trials, every dimension is kept about half the time; --select-on seen measures them on training speakers
so the held-out ones only confirm); ridge and lasso fits of each metric on all masks
at once are reported beside it. Dimensions are ranked by leak removed per top-1 lost (ratio, denominator floored
at the noise) or by the difference in noise units (score); the last step drops the top-ranked ones together and
rescores against as many random ones. --basis pca masks principal axes instead (a rotation: cosines unchanged).

uv run python expandtraining/2606/scripts/dim_attribution.py outputs/runs/clean_nce_a_1gpu__nce_clean_twin__s42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV

from report import heldout_top1_by_speaker, line_leak, speaker_retrieval


def score(rows: pd.DataFrame, embeddings: np.ndarray) -> tuple[float, float]:
    top1 = np.mean(list(heldout_top1_by_speaker(rows, embeddings).values()))
    return float(top1), line_leak(rows, embeddings)["line_leak_heldout"]


def score_seen(rows: pd.DataFrame, embeddings: np.ndarray) -> tuple[float, float]:
    """The same two numbers on speakers seen in training, so dimensions can be picked without the held-out ones."""
    return (speaker_retrieval(rows, embeddings, heldout=False)["top1"],
            line_leak(rows, embeddings, heldout=False)["line_leak_seen"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--drop", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--names", help="optional text file, one dimension name per line")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basis", choices=["axis", "pca"], default="axis")
    parser.add_argument("--select-on", choices=["heldout", "seen"], default="heldout",
                        help="seen: rank dimensions on training speakers, confirm on held-out ones")
    args = parser.parse_args()
    validation = Path(args.run) / "validation"
    rows = pd.read_parquet(validation / "rows.parquet")
    embeddings = np.load(validation / "embeddings.npy").astype(np.float32)
    if args.basis == "pca":  # rotate onto the principal axes of the centred embeddings; norms and cosines stay
        _, _, vt = np.linalg.svd(embeddings - embeddings.mean(0), full_matrices=False)
        embeddings = embeddings @ vt.T
    dims = embeddings.shape[1]
    base = score(rows, embeddings)
    print(f"{Path(args.run).name}: {dims} dims, top1 {base[0]:.3f}, leak {base[1]:.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    masks = np.zeros((args.trials, dims), dtype=bool)
    results = np.zeros((args.trials, 2))
    for trial in range(args.trials):
        masks[trial, rng.permutation(dims)[:dims // 2]] = True  # True = kept
        results[trial] = (score_seen if args.select_on == "seen" else score)(rows, embeddings * masks[trial])
        if (trial + 1) % 50 == 0:
            print(f"  {trial + 1} trials", flush=True)
    kept = masks.sum(0)
    effect = {name: (masks.T @ results[:, i]) / kept - ((~masks).T @ results[:, i]) / (args.trials - kept)
              for i, name in enumerate(("top1", "leak"))}
    noise = {name: results[:, i].std() * np.sqrt(4 / args.trials) for i, name in enumerate(("top1", "leak"))}
    prefix = "pc" if args.basis == "pca" else "d"
    names = (Path(args.names).read_text().splitlines() if args.names and args.basis == "axis"
             else [f"{prefix}{i}" for i in range(dims)])
    table = pd.DataFrame({"dim": names, "leak_effect": effect["leak"], "top1_effect": effect["top1"]})
    x = masks.astype(np.float64)
    for i, name in enumerate(("top1", "leak")):
        table[f"{name}_ridge"] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x, results[:, i]).coef_
        table[f"{name}_lasso"] = LassoCV(cv=5, random_state=0, max_iter=20000).fit(x, results[:, i]).coef_
    # ranked on the ridge fit; keeping a dimension raises leak by leak_ridge and top-1 by top1_ridge
    table["ratio"] = table["leak_ridge"] / table["top1_ridge"].clip(lower=noise["top1"])
    table["score"] = table["leak_ridge"] / noise["leak"] - table["top1_ridge"].clip(lower=0) / noise["top1"]
    table["flagged"] = table["leak_ridge"] > 2 * noise["leak"]
    table = table.sort_values("ratio", ascending=False)
    suffix = "_seen" if args.select_on == "seen" else ""
    out = validation / f"dim_attribution_{args.basis}{suffix}.csv"
    table.to_csv(out, index=False)
    print(f"effect noise (1 se): top1 {noise['top1']:.4f}, leak {noise['leak']:.4f}; "
          f"{int(table['flagged'].sum())} dims flagged, lasso keeps {int((table['leak_lasso'] > 0).sum())} "
          f"leak dims -> {out}", flush=True)
    print(table.head(12).to_string(index=False, float_format="%.4f"), flush=True)

    for rank in ("ratio", "score"):
        order = table.sort_values(rank, ascending=False).index.to_numpy()
        for n in args.drop:
            keep = np.ones(dims, dtype=bool)
            keep[order[:n]] = False
            flagged = score(rows, embeddings * keep)
            keep = np.ones(dims, dtype=bool)
            keep[rng.permutation(dims)[:n]] = False
            random = score(rows, embeddings * keep)
            print(f"{rank:5s} drop {n:4d}: top1 {flagged[0]:.3f} leak {flagged[1]:.3f} | "
                  f"random: top1 {random[0]:.3f} leak {random[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
