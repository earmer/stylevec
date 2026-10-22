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


def _retrieval_lines(rows: pd.DataFrame, heldout: bool) -> pd.DataFrame:
    """Distinct speaker-labelled lines (anchor/positive roles of the Genshin speaker rules)."""
    mask = (
        rows["rule"].isin(SPEAKER_RULES) & rows["role"].isin(["anchor", "positive"])
        & (rows["heldout"] == heldout) & (rows["speaker"] != "")
    )
    return rows[mask].drop_duplicates(subset=["speaker", "text_preview"])


def _retrieval_ranks(
    labels: np.ndarray, vectors: np.ndarray, db_ids: np.ndarray | None = None, min_gap: int = 0,
) -> np.ndarray:
    """Rank of each line's own speaker among the centroids, leaving the line (and, with min_gap > 0, its
    own speaker's lines within min_gap DB ids) out of its centroid. -1 where no other line is left."""
    vectors = _unit(vectors.astype(np.float64))
    classes = sorted(set(labels))
    counts = {c: int((labels == c).sum()) for c in classes}
    sums = np.stack([vectors[labels == c].sum(axis=0) for c in classes])
    ranks = np.full(len(labels), -1, dtype=int)
    for i, label in enumerate(labels):
        same = labels == label
        if db_ids is None or min_gap <= 0 or db_ids[i] < 0:
            drop = np.zeros(len(labels), dtype=bool)
            drop[i] = True
        else:
            drop = same & (db_ids >= 0) & (np.abs(db_ids - db_ids[i]) < min_gap)
            drop[i] = True
        if counts[label] - int((drop & same).sum()) < 1:
            continue  # no other line to build the centroid from
        centroids = sums.copy()
        k = classes.index(label)
        centroids[k] = centroids[k] - vectors[drop & same].sum(axis=0)
        scores = _unit(centroids) @ vectors[i]
        ranks[i] = int((scores > scores[k]).sum())
    return ranks


def speaker_retrieval(
    rows: pd.DataFrame, embeddings: np.ndarray, heldout: bool = True, min_gap: int = 0,
) -> dict[str, float]:
    """Leave-one-out nearest-centroid speaker retrieval on distinct lines (anchor/positive roles).

    With heldout=True only speakers never seen in training are used, so the centroids come from
    validation lines alone: this measures transfer, not memory. With min_gap > 0 the query's own
    speaker centroid also leaves out that speaker's lines within `min_gap` DB ids of the query (same
    scene), so a hit cannot come from scene proximity. Needs a `db_id` column; lines without one
    (db_id < 0) exclude nothing.
    """
    if min_gap > 0 and "db_id" not in rows:
        return {"n_lines": 0.0}
    sub = _retrieval_lines(rows, heldout)
    out: dict[str, float] = {"n_lines": float(len(sub))}
    if sub.empty:
        return out
    labels = sub["speaker"].to_numpy()
    classes = sorted(set(labels))
    out["n_speakers"] = float(len(classes))
    if len(classes) < 2:
        return out
    db_ids = sub["db_id"].to_numpy() if min_gap > 0 else None
    ranks = _retrieval_ranks(labels, embeddings[sub.index.to_numpy()], db_ids, min_gap)
    scored = ranks[ranks >= 0]
    if len(scored):
        out.update(top1=float((scored == 0).mean()), top5=min(1.0, float((scored < 5).mean())),
                   chance_top1=1 / len(classes), chance_top5=min(1.0, 5 / len(classes)),
                   n_scored=float(len(scored)))
    return out


def heldout_top1_by_speaker(rows: pd.DataFrame, embeddings: np.ndarray) -> dict[str, float]:
    """Held-out top-1 per speaker. With only ~20 held-out speakers and hits concentrated on a few of
    them, which speakers were held out moves top-1 more than the training seed does, so compare
    pairs arms speaker by speaker (paired standard error across speakers)."""
    sub = _retrieval_lines(rows, True)
    labels = sub["speaker"].to_numpy()
    if len(set(labels)) < 2:
        return {}
    ranks = _retrieval_ranks(labels, embeddings[sub.index.to_numpy()])
    frame = pd.DataFrame({"speaker": labels, "hit": ranks == 0})[ranks >= 0]
    return {str(k): float(v) for k, v in frame.groupby("speaker")["hit"].mean().items()}


def seen_top1_matched(
    rows: pd.DataFrame, embeddings: np.ndarray, repeats: int = 20, seed: int = 0,
) -> float | None:
    """Seen-speaker top-1 on the held-out task's terms: as many seen speakers as there are held-out
    ones, each with a line count drawn from the held-out speakers' counts, averaged over `repeats`
    draws. The raw seen top-1 is a ~4x larger classification with fewer lines per centroid, so
    seen − held-out is not a memorisation gap; this one is."""
    held = _retrieval_lines(rows, True)
    seen = _retrieval_lines(rows, False)
    held_counts = held["speaker"].value_counts().to_numpy()
    n_speakers = len(held_counts)
    seen_counts = seen["speaker"].value_counts()
    pool = seen_counts[seen_counts >= 2].index.to_numpy()
    if n_speakers < 2 or len(pool) < n_speakers:
        return None
    rng = np.random.default_rng(seed)
    by_speaker = {s: seen.index[seen["speaker"].to_numpy() == s].to_numpy() for s in pool}
    scores = []
    for _ in range(repeats):
        picked = []
        for speaker, count in zip(rng.choice(pool, n_speakers, replace=False), rng.permutation(held_counts)):
            lines = by_speaker[speaker]
            picked.append(rng.choice(lines, min(len(lines), max(2, int(count))), replace=False))
        index = np.concatenate(picked)
        ranks = _retrieval_ranks(rows.loc[index, "speaker"].to_numpy(), embeddings[index])
        if (ranks >= 0).any():
            scores.append(float((ranks[ranks >= 0] == 0).mean()))
    return float(np.mean(scores)) if scores else None


def prefix_top1(rows: pd.DataFrame, embeddings: np.ndarray, dims: int) -> float | None:
    """Held-out top-1 on the first `dims` dimensions (renormalised): what the Matryoshka prefix keeps."""
    if dims >= embeddings.shape[1]:
        return None
    retrieval = speaker_retrieval(rows, embeddings[:, :dims], heldout=True)
    return retrieval.get("top1")


CROSS_SCENE_GAP = 150  # = data.genshin_min_id_gap: what the builder counts as a different scene
WHITEN_DIMS = 128
PREFIX_DIMS = (32, 64, 128)


def topic_gap_by_negative(rows: pd.DataFrame, embeddings: np.ndarray, hubs: frozenset[str]) -> dict[str, float]:
    """Held-out same_speaker_topic_matched gap, split by whether the negative is a hub speaker.

    Hub speakers (Paimon, Traveler) are seen in training and in nearly every scene, so a model can
    score on them by learning "not Paimon"; the non-hub split is the cleaner same-scene test.
    """
    frame = triplet_frame(rows, embeddings)
    if frame.empty:
        return {}
    frame = frame[(frame["rule"] == "same_speaker_topic_matched") & frame["heldout"]]
    negatives = rows[rows["role"] == "negative"].set_index("triplet_id")["speaker"]
    is_hub = frame["triplet_id"].map(negatives).isin(hubs).to_numpy()
    gap = (frame["pos"] - frame["neg"]).to_numpy()
    out: dict[str, float] = {}
    for name, mask in (("hub", is_hub), ("other", ~is_hub)):
        if mask.any():
            out[f"topic_gap_{name}_neg"] = float(gap[mask].mean())
            out[f"n_topic_{name}_neg"] = float(mask.sum())
    return out


def heldout_speaker_set(rows: pd.DataFrame) -> set[str]:
    """Speakers of held-out Genshin anchors: the speakers never seen in training."""
    mask = rows["rule"].isin(SPEAKER_RULES) & rows["heldout"].astype(bool) & (rows["role"] == "anchor")
    return set(rows.loc[mask, "speaker"]) - {""}


def real_gap_by_negative(rows: pd.DataFrame, embeddings: np.ndarray) -> dict[str, float]:
    """Held-out same_speaker_real gap, split by whether the negative speaker is also held out (unseen vs
    unseen) or was seen in training. About half of these negatives are seen speakers."""
    frame = triplet_frame(rows, embeddings)
    if frame.empty:
        return {}
    frame = frame[(frame["rule"] == "same_speaker_real") & frame["heldout"]]
    negatives = rows[rows["role"] == "negative"].set_index("triplet_id")["speaker"]
    unseen = frame["triplet_id"].map(negatives).isin(heldout_speaker_set(rows)).to_numpy()
    gap = (frame["pos"] - frame["neg"]).to_numpy()
    out: dict[str, float] = {}
    for name, mask in (("heldout", unseen), ("seen", ~unseen)):
        if mask.any():
            out[f"real_gap_{name}_neg"] = float(gap[mask].mean())
            out[f"n_real_{name}_neg"] = float(mask.sum())
    return out


def whitened(rows: pd.DataFrame, embeddings: np.ndarray, dims: int = WHITEN_DIMS) -> np.ndarray | None:
    """PCA-whiten onto the top `dims` directions, fitted on rows from seen speakers only (out of sample for
    held-out retrieval). Post-hoc decorrelation for free: a training-time spread loss must beat it.

    Not just heldout == False rows: a seen anchor's negative is often a held-out speaker's line. Rows
    with no speaker (traits) are kept; they cannot be a held-out speaker's line."""
    unseen = heldout_speaker_set(rows)
    keep = ~rows["heldout"].astype(bool) & ~rows["speaker"].isin(unseen)
    fit = embeddings[keep.to_numpy(dtype=bool)].astype(np.float64)
    if len(fit) < 3:
        return None
    mean = fit.mean(axis=0)
    _u, singular, vt = np.linalg.svd(fit - mean, full_matrices=False)
    keep = singular > singular[0] * 1e-6
    k = min(dims, int(keep.sum()))
    return _unit((embeddings.astype(np.float64) - mean) @ vt[:k].T / singular[:k])


def diagnostics(rows: pd.DataFrame, embeddings: np.ndarray, hubs: frozenset[str]) -> dict[str, Any]:
    """Scalars for compare (topic gap by negative type, cross-scene, matched-seen, prefix and whitened
    retrieval) plus `heldout_top1_by_speaker`, the per-speaker hits compare pairs arms on."""
    out = topic_gap_by_negative(rows, embeddings, hubs)
    out.update(real_gap_by_negative(rows, embeddings))
    cross = speaker_retrieval(rows, embeddings, heldout=True, min_gap=CROSS_SCENE_GAP)
    if "top1" in cross:
        out["heldout_top1_cross_scene"] = cross["top1"]
    held = speaker_retrieval(rows, embeddings, heldout=True)
    by_speaker = heldout_top1_by_speaker(rows, embeddings)
    if by_speaker:
        out["heldout_top1_by_speaker"] = by_speaker
        values = np.array(list(by_speaker.values()))
        out["heldout_top1_macro"] = float(values.mean())
        if len(values) > 1:
            out["heldout_top1_speaker_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
    matched = seen_top1_matched(rows, embeddings)
    if matched is not None:
        out["seen_top1_matched"] = matched
        if "top1" in held:
            out["memory_gap_matched"] = matched - held["top1"]
    for dims in PREFIX_DIMS:
        top1 = prefix_top1(rows, embeddings, dims)
        if top1 is not None:
            out[f"heldout_top1_d{dims}"] = top1
    white = whitened(rows, embeddings)
    if white is not None:
        retrieval = speaker_retrieval(rows, white, heldout=True)
        if "top1" in retrieval:
            out["whitened_top1"] = retrieval["top1"]
        frame = triplet_frame(rows, white)
        ovp = frame[frame["rule"] == "origin_vs_para"] if not frame.empty else frame
        if len(ovp):
            out["whitened_gap_origin_vs_para"] = float((ovp["pos"] - ovp["neg"]).mean())
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


def render_report(
    table: pd.DataFrame, retrieval: dict[str, dict[str, float]], probe: dict[str, Any],
    extra: dict[str, float] | None = None,
) -> list[str]:
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
    if extra:
        lines += ["", "## Diagnostics", ""]
        lines += [f"- {key}: {value:.4f}" for key, value in sorted(extra.items()) if isinstance(value, (int, float))]
    return lines
