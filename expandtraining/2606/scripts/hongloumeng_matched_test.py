"""红楼梦: the chapter-80 trait shift on passages matched for content across the boundary. See prereg/hongloumeng_matched_test.md.

    uv run python expandtraining/2606/scripts/hongloumeng_matched_test.py

Reads only cached predictions from scripts/hongloumeng_trait_test.py, so no GPU.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import resolve_repo_path
from hongloumeng_probe import mask_names
from hongloumeng_trait_test import PRIMARY, RESULTS, Reanalysis, per_passage, sanitize, split_all
from pairwise_voice_baseline import BASE_MODEL
from trait_predictor import NAMES

OUT = "data/episodes/external/results/hongloumeng_matched_test.json"
FROZEN = "frozen_delta1.0_s42_attention_layers6-12-18-24-28"
KEY = ("swagger", "irony", "playfulness", "hedging", "deference")
SCENES = {"banquet": "宴|席|酒|戲", "poetry": "詩|詞|聯句|吟|韻", "household": "銀子|月錢|帳|管家|當票|租", "illness": "病|藥|大夫|脈"}
BALANCE_MAX = .65
BOOTS = 2000
WINDOW = 30
PLACEBOS = range(31, 51)


def quoted_share(text: str) -> float:
    """Share of characters inside quotes. Passages are cut mid-sentence, so an unopened close counts from the start and an unclosed open to the end."""
    inside, count = False, 0
    first_close = re.search("[”]", text)
    first_open = re.search("[“\"]", text)
    if first_close and (not first_open or first_close.start() < first_open.start()):
        inside = True
    for c in text:
        if c in "“\"":
            inside = True
        elif c == "”":
            inside = False
        elif inside:
            count += 1
    return count / max(len(text), 1)


def content_features(study: Reanalysis) -> np.ndarray:
    idf = np.log(len(study.m) / (1 + (study.m > 0).sum(0)))
    f = study.m * idf
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


def match(early: np.ndarray, late: np.ndarray, f: np.ndarray, strata: np.ndarray, keep: float = .5) -> tuple[np.ndarray, np.ndarray, dict]:
    """(early indices, late indices, stats): one-to-one inside strata, the most similar `keep` of the pairs."""
    pairs = []
    for s in np.unique(strata[np.r_[early, late]]):
        a, b = early[strata[early] == s], late[strata[late] == s]
        if not len(a) or not len(b):
            continue
        sim = f[b] @ f[a].T
        rows, cols = linear_sum_assignment(sim, maximize=True)
        pairs += [(a[c], b[r], sim[r, c]) for r, c in zip(rows, cols)]
    pairs = np.array(pairs)
    cut = np.quantile(pairs[:, 2], 1 - keep)
    chosen = pairs[pairs[:, 2] >= cut]
    return chosen[:, 0].astype(int), chosen[:, 1].astype(int), {"pairs_all": len(pairs), "pairs_kept": len(chosen), "mean_cosine_all": float(pairs[:, 2].mean()), "mean_cosine_kept": float(chosen[:, 2].mean())}


def grouped_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, reduce: bool = False) -> float:
    scores = np.zeros(len(y))
    for train, test in GroupKFold(5).split(x, y, groups):
        steps = [StandardScaler()] + ([PCA(len(NAMES), random_state=0)] if reduce else []) + [LogisticRegression(C=1.0, max_iter=2000)]
        scores[test] = make_pipeline(*steps).fit(x[train], y[train]).decision_function(x[test])
    return float(roc_auc_score(y, scores))


def trait_shift(y: np.ndarray, ch: np.ndarray, a: np.ndarray, b: np.ndarray, seed: int = 42) -> dict:
    """Late minus early in passage SD units, with an interval from resampling early and late chapters independently."""
    sd = y.std(0)
    diff = (y[b] - y[a]) / sd
    ca, cb = ch[a], ch[b]
    early, late = np.unique(ca), np.unique(cb)
    pos_e, pos_l = np.searchsorted(early, ca), np.searchsorted(late, cb)
    rng = np.random.default_rng(seed)
    boots = np.zeros((BOOTS, y.shape[1]))
    for i in range(BOOTS):
        w = np.bincount(rng.integers(0, len(early), len(early)), minlength=len(early))[pos_e] * np.bincount(rng.integers(0, len(late), len(late)), minlength=len(late))[pos_l]
        boots[i] = (w[:, None] * diff).sum(0) / max(w.sum(), 1)
    lo, hi = np.quantile(boots, [.025, .975], axis=0)
    return {name: {"shift": float(diff[:, k].mean()), "ci95": [float(lo[k]), float(hi[k])]} for k, name in enumerate(NAMES)}


def unmatched_shift(y: np.ndarray, early: np.ndarray, late: np.ndarray) -> dict:
    d = (y[late].mean(0) - y[early].mean(0)) / y.std(0)
    return {name: float(d[k]) for k, name in enumerate(NAMES)}


def main() -> None:
    passages = json.loads(resolve_repo_path(f"{RESULTS}/passages.json").read_text(encoding="utf-8"))
    cache = resolve_repo_path(f"{RESULTS}/cache")
    texts = [p["text"] for p in passages]
    ch, lens = np.array([p["chapter"] for p in passages]), np.array([p["length"] for p in passages])
    verse = np.array([p["is_verse"] for p in passages])
    masked = [mask_names(t) for t in texts]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(resolve_repo_path(BASE_MODEL)), local_files_only=True, trust_remote_code=True)
    pieces = {"": split_all(texts, tokenizer), "_masked": split_all(masked, tokenizer)}

    def traits(run: str, variant: str = "") -> np.ndarray:
        return per_passage(pieces[variant], np.load(cache / f"traits_{run}{variant}.npy"), len(texts))

    L = np.c_[np.log(lens.astype(float)), np.ones(len(lens))]
    residual = lambda e: e.astype(float) - L @ np.linalg.lstsq(L, e.astype(float), rcond=None)[0]  # noqa: E731
    study = Reanalysis(np.array(texts, dtype=object), ch, lens)
    f = content_features(study)
    share = np.array([quoted_share(t) for t in texts])
    dialogue = np.digitize(share, [.15, .5])
    length = np.digitize(lens, np.quantile(lens, [.25, .5, .75]))
    strata = dialogue * 10 + length
    ok = ~verse
    models = {"lora_attention": residual(traits(PRIMARY)), "lora_attention_masked": residual(traits(PRIMARY, "_masked")), "frozen_taps": residual(traits(FROZEN))}
    others = {"old_256d": residual(np.load(cache / "trained_full.npy")), "base_qwen": residual(np.load(cache / "base_qwen3.npy")), "function_chars": residual(study.function_chars()), "content_chars": study.m}
    early, late = np.flatnonzero(ok & (ch <= 80)), np.flatnonzero(ok & (ch > 80))

    a, b, stats = match(early, late, f, strata)
    result = {"data": {"early": len(early), "late": len(late), "dialogue_class_counts": np.bincount(dialogue[ok]).tolist(), "match": stats, "early_chapters_used": len(np.unique(ch[a])), "late_chapters_used": len(np.unique(ch[b]))}}
    before = np.r_[early, late]
    index = np.r_[a, b]
    side = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    result["balance_whole_book"] = {"content_auc_unmatched": grouped_auc(study.m[before], (ch[before] > 80).astype(int), ch[before]), "content_auc_matched": grouped_auc(study.m[index], side, ch[index]),
                                    "quoted_share_early_late": [float(share[a].mean()), float(share[b].mean())], "length_early_late": [float(lens[a].mean()), float(lens[b].mean())]}
    result["whole_book"] = {name: {"unmatched": unmatched_shift(y, early, late), "matched": trait_shift(y, ch, a, b), "auc_unmatched": grouped_auc(y[before], (ch[before] > 80).astype(int), ch[before]), "auc_matched": grouped_auc(y[index], side, ch[index])} for name, y in models.items()}
    print(json.dumps(result["data"]), json.dumps(result["balance_whole_book"]), flush=True)

    def window(boundary: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = np.flatnonzero(ok & (ch > boundary - WINDOW) & (ch <= boundary))
        second = np.flatnonzero(ok & (ch > boundary) & (ch <= boundary + WINDOW))
        x, z, _ = match(first, second, f, strata)
        at = np.r_[x, z]
        return at, np.r_[np.zeros(len(x), dtype=int), np.ones(len(z), dtype=int)], ch[at]

    all_methods = {**models, **others}
    curve = {name: {} for name in all_methods}
    for boundary in [80, *PLACEBOS]:
        at, y, groups = window(boundary)
        for name, e in all_methods.items():
            curve[name][boundary] = grouped_auc(e[at], y, groups, reduce=e.shape[1] > len(NAMES) and name != "content_chars")
        print(boundary, len(at), {k: round(v[boundary], 3) for k, v in curve.items()}, flush=True)
    result["separation"] = {name: {"auc_80": c[80], "null_max": max(c[b] for b in PLACEBOS), "null_mean": float(np.mean([c[b] for b in PLACEBOS])), "rank_of_80": 1 + sum(c[b] >= c[80] for b in PLACEBOS), "of": 1 + len(PLACEBOS), "by_boundary": {str(k): v for k, v in c.items()}} for name, c in curve.items()}

    y = models["lora_attention"]
    subsets = {f"scene_{name}": np.array([bool(re.search(pattern, t)) for t in texts]) for name, pattern in SCENES.items()}
    subsets |= {"narration": dialogue == 0, "mixed": dialogue == 1, "dialogue": dialogue == 2}
    result["subsets"] = {}
    for name, keep in subsets.items():
        e, l = np.flatnonzero(ok & keep & (ch <= 80)), np.flatnonzero(ok & keep & (ch > 80))
        x, z, s = match(e, l, f, strata)
        at, side2 = np.r_[x, z], np.r_[np.zeros(len(x), dtype=int), np.ones(len(z), dtype=int)]
        result["subsets"][name] = {"early": len(e), "late": len(l), **s, "unmatched": {k: unmatched_shift(y, e, l)[k] for k in KEY}, "matched": {k: trait_shift(y, ch, x, z)[k] for k in KEY},
                                   "auc_matched": grouped_auc(y[at], side2, ch[at]), "content_auc_matched": grouped_auc(study.m[at], side2, ch[at])}
        print(name, json.dumps(sanitize(result["subsets"][name]))[:400], flush=True)

    primary = result["whole_book"]["lora_attention"]
    kept = {k: np.sign(primary["matched"][k]["shift"]) == np.sign(primary["unmatched"][k]) and (primary["matched"][k]["ci95"][0] > 0 or primary["matched"][k]["ci95"][1] < 0) for k in KEY}
    sep = result["separation"]["lora_attention"]
    result["criteria"] = {"balance_ok": result["balance_whole_book"]["content_auc_matched"] <= BALANCE_MAX, "traits_kept": kept, "traits_ok": sum(kept.values()) >= 4, "separation_ok": sep["auc_80"] > sep["null_max"]}
    result["criteria"]["pass"] = all(result["criteria"][k] for k in ("balance_ok", "traits_ok", "separation_ok"))
    out = resolve_repo_path(OUT)
    out.write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print("CRITERIA", json.dumps(sanitize(result["criteria"])), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
