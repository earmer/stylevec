"""A readable page on the frozen-LM style neurons (frozen_llm_eval.py --neuron-models must have run).

For each requested model and layer, the top neurons by Fisher ratio: the lines whose activation rises most over
their neutral rewrite (line and rewrite side by side, so the difference is the feature), the lines where it
falls most, and the speakers whose lines shift it most on average.

uv run python expandtraining/2606/scripts/neuron_report.py --tag a --pick q4bi:27 q17b:21 q06b:28
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from config import workspace_dir


def clip(text: str, width: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "/")
    return text if len(text) <= width else text[:width - 1] + "…"


def neuron_section(full: pd.DataFrame, line: np.ndarray, rewrite: np.ndarray, ranked: np.ndarray, score: np.ndarray,
                   column: int, examples: int, min_lines: int) -> list[str]:
    shift = line[:, column] - rewrite[:, column]
    out = [f"### neuron {int(ranked[column])} (rank {column + 1}, Fisher {score[column]:.3f})", ""]
    genshin = full["speaker"].str.contains("[\u4e00-\u9fff]")  # not the synthetic-feature "speakers"
    by_speaker = (pd.DataFrame({"speaker": full["speaker"], "shift": shift})[genshin]
                  .groupby("speaker")["shift"].agg(["mean", "size"]).query("size >= @min_lines"))
    high = ", ".join(f"{s} {m:+.2f}" for s, m in by_speaker["mean"].nlargest(5).items())
    low = ", ".join(f"{s} {m:+.2f}" for s, m in by_speaker["mean"].nsmallest(3).items())
    out += [f"Speakers who raise it most: {high}. Lower it most: {low}.", ""]
    out += ["| shift | speaker | line | neutral rewrite |", "|---|---|---|---|"]
    for i in list(np.argsort(-shift)[:examples]) + list(np.argsort(shift)[:max(1, examples // 3)]):
        out.append(f"| {shift[i]:+.2f} | {full['speaker'][i] or '-'} | {clip(full['text'][i])} | "
                   f"{clip(full['neutral'][i])} |")
    return out + [""]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pick", nargs="+", required=True, help="model_tag:layer, e.g. q4bi:27")
    parser.add_argument("--neurons", type=int, default=12)
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--min-lines", type=int, default=10)
    args = parser.parse_args()
    cache = workspace_dir() / "outputs" / "frozen" / args.tag
    full = pd.read_parquet(cache / "full.parquet")
    page = [f"# Frozen-LM style neurons (frozen_{args.tag})", "",
            "Each neuron is an MLP unit whose activation shift from a line's neutral rewrite to the line itself "
            "best separates 32 training speakers (Fisher ratio). Shifts are in units of the neuron's "
            "within-speaker spread. Lines come from the validation set and the ranking lines.", ""]
    for pick in args.pick:
        tag, layer = pick.split(":")
        npz = next(cache.glob(f"neurons_{tag}_*_top*.npz"))
        stored = np.load(npz)
        line, rewrite = stored[f"p{layer}"].astype(np.float32), stored[f"f{layer}"].astype(np.float32)
        ranked, score = np.load(cache / f"neuron_rank_{tag}_l{layer}.npy")
        page += [f"## {tag}, layer {layer}", ""]
        for column in range(args.neurons):
            page += neuron_section(full, line, rewrite, ranked.astype(int), score, column, args.examples,
                                   args.min_lines)
    out = cache / "neuron_report.md"
    out.write_text("\n".join(page), encoding="utf-8")
    print(f"wrote {out} ({len(page)} lines)")


if __name__ == "__main__":
    main()
