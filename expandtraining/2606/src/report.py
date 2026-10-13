"""Per-rule validation metrics computed from saved rows + embeddings (no model, no GPU)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

RULE_DOMAIN_PREFIXES = (
    ("bookraw_", "bookraw"),
    ("trait_", "trait"),
    ("generated_to_target", "generated"),
    ("target_hard_negative", "generated"),
    ("same_content_diff_style", "genshin_real"),
    ("same_speaker_", "genshin_real"),
    ("origin_vs_para", "genshin_real"),
)
SPEAKER_RULES = {"same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"}


def rule_domain(rule: str) -> str | None:
    for prefix, domain in RULE_DOMAIN_PREFIXES:
        if rule.startswith(prefix):
            return domain
    return None


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def triplet_frame(rows: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """One row per complete triplet: rule, heldout, raw and centered cosines."""
    if len(rows) != len(embeddings):
        raise ValueError("rows and embeddings must align")
    unit = _unit(embeddings.astype(np.float64))
    centered = _unit(embeddings.astype(np.float64) - embeddings.mean(axis=0, keepdims=True))
    index = {(tid, role): i for i, (tid, role) in enumerate(zip(rows["triplet_id"], rows["role"]))}
    records = []
    for tid, rule, heldout in rows[rows["role"] == "anchor"][["triplet_id", "rule", "heldout"]].itertuples(index=False):
        try:
            a, p, n = index[(tid, "anchor")], index[(tid, "positive")], index[(tid, "negative")]
        except KeyError:  # incomplete triplet: never score it
            continue
        records.append(
            {
                "triplet_id": tid, "rule": rule, "heldout": bool(heldout),
                "pos": float(unit[a] @ unit[p]), "neg": float(unit[a] @ unit[n]),
                "cpos": float(centered[a] @ centered[p]), "cneg": float(centered[a] @ centered[n]),
            }
        )
    return pd.DataFrame(records)


def per_rule_table(rows: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    frame = triplet_frame(rows, embeddings)
    if frame.empty:
        return frame
    frame["gap"] = frame["pos"] - frame["neg"]
    frame["cgap"] = frame["cpos"] - frame["cneg"]
    frame["violation"] = (frame["pos"] <= frame["neg"]).astype(float)
    table = frame.groupby(["rule", "heldout"]).agg(
        n=("pos", "size"), pos=("pos", "mean"), neg=("neg", "mean"),
        gap=("gap", "mean"), cgap=("cgap", "mean"), violation=("violation", "mean"),
    )
    return table.reset_index()


def speaker_retrieval(rows: pd.DataFrame, embeddings: np.ndarray, heldout: bool = True) -> dict[str, float]:
    """Leave-one-out nearest-centroid speaker retrieval on distinct lines (anchor/positive roles).

    With heldout=True only speakers never seen in training are used, so the centroids come from
    validation lines alone: this measures transfer, not memory.
    """
    mask = (
        rows["rule"].isin(SPEAKER_RULES) & rows["role"].isin(["anchor", "positive"])
        & (rows["heldout"] == heldout) & (rows["speaker"] != "")
    )
    sub = rows[mask].drop_duplicates(subset=["speaker", "text_preview"])
    out: dict[str, float] = {"n_lines": float(len(sub))}
    if sub.empty:
        return out
    vectors = _unit(embeddings[sub.index.to_numpy()].astype(np.float64))
    labels = sub["speaker"].to_numpy()
    classes = sorted(set(labels))
    out["n_speakers"] = float(len(classes))
    counts = {c: int((labels == c).sum()) for c in classes}
    if len(classes) < 2:
        return out
    sums = np.stack([vectors[labels == c].sum(axis=0) for c in classes])
    top1 = top5 = valid = 0
    for i, label in enumerate(labels):
        if counts[label] < 2:
            continue  # no other line to build the centroid from
        centroids = sums.copy()
        k = classes.index(label)
        centroids[k] = centroids[k] - vectors[i]
        scores = _unit(centroids) @ vectors[i]
        rank = int((scores > scores[k]).sum())
        top1 += rank == 0
        top5 += rank < 5
        valid += 1
    if valid:
        out.update(top1=top1 / valid, top5=min(1.0, top5 / valid), chance_top1=1 / len(classes),
                   chance_top5=min(1.0, 5 / len(classes)), n_scored=float(valid))
    return out


def provenance_probe(rows: pd.DataFrame, embeddings: np.ndarray, max_folds: int = 4) -> dict[str, Any]:
    """Domain recoverable from anchor embeddings (linear probe). High = provenance axis dominates."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    anchors = rows[rows["role"] == "anchor"].copy()
    anchors["domain"] = anchors["rule"].map(rule_domain)
    anchors = anchors.dropna(subset=["domain"])
    out: dict[str, Any] = {"n": int(len(anchors))}
    counts = anchors["domain"].value_counts()
    keep = counts[counts >= 4].index
    anchors = anchors[anchors["domain"].isin(keep)]
    if anchors["domain"].nunique() < 2:
        return out
    x = _unit(embeddings[anchors.index.to_numpy()].astype(np.float64))
    y = anchors["domain"].to_numpy()
    folds = min(max_folds, int(anchors["domain"].value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = cross_val_score(LogisticRegression(max_iter=1000), x, y, cv=cv)
    out.update(accuracy=float(scores.mean()), chance=float(anchors["domain"].value_counts(normalize=True).max()),
               classes=sorted(set(y)), folds=folds)
    return out


def render_report(table: pd.DataFrame, retrieval: dict[str, dict[str, float]], probe: dict[str, Any]) -> list[str]:
    lines = ["## Per-rule (complete triplets only)", "",
             "| rule | heldout speakers | n | pos | neg | gap | centered gap | violation |", "|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        lines.append(f"| {r.rule} | {r.heldout} | {r.n} | {r.pos:.4f} | {r.neg:.4f} | {r.gap:+.4f} | {r.cgap:+.4f} | {r.violation:.1%} |")
    lines += ["", "## Speaker retrieval (leave-one-out nearest centroid)", ""]
    for name, stats in retrieval.items():
        if "top1" in stats:
            lines.append(f"- {name}: top1 {stats['top1']:.3f} (chance {stats['chance_top1']:.3f}), "
                         f"top5 {stats['top5']:.3f} (chance {stats['chance_top5']:.3f}), "
                         f"speakers {int(stats['n_speakers'])}, lines {int(stats['n_scored'])}")
        else:
            lines.append(f"- {name}: not enough data ({stats})")
    lines += ["", "## Provenance probe", ""]
    if "accuracy" in probe:
        lines.append(f"- domain accuracy {probe['accuracy']:.3f} vs majority chance {probe['chance']:.3f} "
                     f"over {probe['classes']} (n={probe['n']}). Progress needs this to FALL while held-out retrieval rises.")
    else:
        lines.append(f"- not enough classes ({probe})")
    return lines
