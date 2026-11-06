"""The 红楼梦 chapter-80 benchmark (issue #4) on the trait predictor, next to the old center model, base Qwen and stylometry.

    uv run python expandtraining/2606/scripts/hongloumeng_trait_test.py [--batch-size 16]

Same passages, chapter vectors, boundaries, statistics and seeds as scripts/hongloumeng_probe.py and
scripts/hongloumeng_style_vs_content.py. The 17 predicted traits stand in for the embedding. predict() truncates at 96
tokens, so a longer passage is cut into pieces of at most 96 tokens, at the last punctuation where it can be, and its
vector is the token-weighted mean of the pieces. Piece predictions are cached in outputs/hongloumeng/results/cache
(traits_<run>[_masked].npy), so a re-run redoes the analysis only.

The target is style, not author: chapter 80 is a boundary between hands and only a noisy proxy for a change of style.
The old methods are recomputed from their cache in the same run, to check this script reproduces the published numbers.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import resolve_repo_path
from hongloumeng_probe import (change_point_test, chapter_vectors, dialogue_share_per_chapter, logistic_probe_range, mask_names,
                               passage_level_bootstrap, placebo_boundaries, placebo_rank_and_zscore, regress_out_covariate,
                               regress_out_length, stylometry_vectors, within_cross_similarity)
from pairwise_voice_baseline import BASE_MODEL
from trait_predictor import NAMES, OUT_DIR, load_run, rows_of, trait_vector

RESULTS = "expandtraining/2606/outputs/hongloumeng/results"
OUT = "data/episodes/external/results/hongloumeng_trait_test.json"
PRIMARY = "lora_r8_delta1.0_s42_attention"
RUNS = (PRIMARY, "frozen_delta1.0_s42_attention_layers6-12-18-24-28", "lora_r8_delta1.0_s42")
MAX_TOKENS = 96
CUT_AT = set("。！？．；，、：\n")
CLEAN_NULLS = range(41, 70)


# --- pieces of at most 96 tokens -------------------------------------------------------------------------------------

def split_passage(text: str, tokenizer) -> list[tuple[str, int]]:
    """(piece, tokens) in order, covering the text. Cut at the last punctuation in the second half of the 96 tokens, else at 96."""
    pieces = []
    rest = text
    while rest:
        encoded = tokenizer(rest, return_offsets_mapping=True, add_special_tokens=False)
        if len(encoded["input_ids"]) <= MAX_TOKENS:
            pieces.append((rest, len(encoded["input_ids"])))
            break
        end = encoded["offset_mapping"][MAX_TOKENS - 1][1]
        start_of_half = encoded["offset_mapping"][MAX_TOKENS // 2][0]
        cuts = [i + 1 for i in range(start_of_half, end) if rest[i] in CUT_AT]
        cut = cuts[-1] if cuts else end
        while len(tokenizer(rest[:cut], add_special_tokens=False)["input_ids"]) > MAX_TOKENS:
            cut -= 1
        pieces.append((rest[:cut], len(tokenizer(rest[:cut], add_special_tokens=False)["input_ids"])))
        rest = rest[cut:]
    return pieces


def split_all(texts: list[str], tokenizer) -> pd.DataFrame:
    rows = [(p, piece, n) for p, text in enumerate(texts) for piece, n in split_passage(text, tokenizer)]
    return pd.DataFrame(rows, columns=["passage", "piece", "tokens"])


def free_gpu() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_pieces(predict, pieces: list[str], batch_size: int) -> np.ndarray:
    """predict() batches at 64 at most, so slices of length-sorted lines set the real batch size. Waits and halves on GPU memory errors."""
    order = np.argsort([len(p) for p in pieces], kind="stable")
    out = np.zeros((len(pieces), len(NAMES)), dtype=np.float32)
    start, began = 0, time.time()
    while start < len(order):
        at = order[start:start + batch_size]
        try:
            out[at] = predict([pieces[i] for i in at])
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            print(f"memory error, waiting a minute, batch size {batch_size}", flush=True)
            free_gpu()
            time.sleep(60)
            continue
        start += len(at)
        if start // batch_size % 100 == 0:
            print(f"{time.strftime('%H:%M:%S')} {start}/{len(order)} lines, {start / (time.time() - began):.1f}/s", flush=True)
    return out


def cached_predictions(run: str, variants: dict[str, pd.DataFrame], batch_size: int) -> dict[str, np.ndarray]:
    cache = resolve_repo_path(f"{RESULTS}/cache")
    paths = {v: cache / f"traits_{run}{v}.npy" for v in variants}
    missing = [v for v, path in paths.items() if not path.exists()]
    if missing:
        print(f"loading {run}", flush=True)
        _, predict = load_run(resolve_repo_path(OUT_DIR) / run)
        for v in missing:
            print(f"predicting {run}{v}: {len(variants[v])} pieces", flush=True)
            np.save(paths[v], predict_pieces(predict, list(variants[v]["piece"]), batch_size))
        del predict
        gc.collect()
        free_gpu()
    return {v: np.load(path) for v, path in paths.items()}


def per_passage(pieces: pd.DataFrame, y: np.ndarray, count: int, first_only: bool = False) -> np.ndarray:
    assert len(pieces) == len(y)
    weight = pieces["tokens"].to_numpy(float)
    index = pieces["passage"].to_numpy()
    if first_only:
        keep = ~pieces["passage"].duplicated().to_numpy()
        weight, index, y = weight[keep], index[keep], y[keep]
    total = np.bincount(index, weight, count)
    return np.stack([np.bincount(index, weight * y[:, k], count) / total for k in range(y.shape[1])], axis=1)


# --- hongloumeng_probe.py: run_analysis and the placebo, without the plots -------------------------------------------

def probe_analysis(embs: np.ndarray, chapters: list[int], lengths: list[int], verse: list[bool], dialogue: np.ndarray) -> dict:
    out = {}
    for suffix, mask in (("", np.ones(len(chapters), dtype=bool)), ("_no_verse", ~np.array(verse))):
        for lc in (False, True):
            e, c, n = embs[mask], list(np.array(chapters)[mask]), list(np.array(lengths)[mask])
            if lc:
                e = regress_out_length(e, n)
            v = chapter_vectors(e, c)
            logistic = logistic_probe_range(v)
            out[("length_corrected" if lc else "raw") + suffix] = {
                "change_point": change_point_test(v), "change_point_no_intro": change_point_test(v, skip_first=5),
                "flip_chapter": logistic["flip_chapter"], "p_late": dict(zip(map(str, logistic["chapter_range"]), logistic["p_late"])),
                "within_cross": within_cross_similarity(v), "passage_bootstrap": passage_level_bootstrap(e, c)}
            if lc and not suffix:
                corrected = logistic_probe_range(regress_out_covariate(v, dialogue))
                out["length_corrected"]["flip_chapter_dialogue_corrected"] = corrected["flip_chapter"]
    return out


def probe_placebo(embs: np.ndarray, chapters: list[int], lengths: list[int], lc: bool) -> dict:
    curve = placebo_boundaries(chapter_vectors(regress_out_length(embs, lengths) if lc else embs, chapters))
    clean = [curve[b]["auc"] for b in curve if b < 70]
    return {"auc": placebo_rank_and_zscore(curve, 80, "auc"), "step_size": placebo_rank_and_zscore(curve, 80, "step_size"),
            "clean_null_max": max(clean), "clean_null_mean": float(np.mean(clean)), "by_boundary": {str(b): curve[b]["auc"] for b in curve}}


def p_late_summary(probe: dict) -> dict:
    p = {int(k): v for k, v in probe["length_corrected"]["p_late"].items()}
    return {"mean_61_80": float(np.mean([p[c] for c in range(61, 81)])), "mean_81_90": float(np.mean([p[c] for c in range(81, 91)])),
            "ch67": p[67], "flip_chapter": probe["length_corrected"]["flip_chapter"]}


# --- hongloumeng_style_vs_content.py, same code ----------------------------------------------------------------------

class Reanalysis:
    def __init__(self, texts: np.ndarray, ch: np.ndarray, lengths: np.ndarray) -> None:
        self.texts, self.ch, self.lengths = texts, ch, lengths
        self.L = np.log(lengths.astype(float))
        chapters = range(1, 121)

        def rate(pattern: str) -> np.ndarray:
            r = re.compile(pattern)
            return np.array([sum(len(r.findall(t)) for t in texts[ch == c]) / sum(len(t) for t in texts[ch == c]) for c in chapters])

        self.feat = pd.DataFrame({"classical": rate("[之乎者也矣焉哉]"), "vernacular": rate("[了的著着呢麼么罷罢]"), "dialogue": rate("道：|說：|“|「"),
                                  "verse_lines": rate(r"(?:^|[，。．])[一-鿿]{5}[，。．]|(?:^|[，。．])[一-鿿]{7}[，。．]"),
                                  "mean_passage_len": np.array([lengths[ch == c].mean() for c in chapters])}, index=chapters)
        self.freg = ((self.feat - self.feat.mean()) / self.feat.std()).to_numpy()
        fcs = set("之其或亦方即皆仍故尚呢了的着著吗嗎么麼罢罷吧便就也矣乎焉哉而以於于乃则則但却卻又且都将將把被叫让讓是不我你他這这那一有在來来去說说道人")
        count = Counter(c for t in texts for c in t if "一" <= c <= "鿿" and c not in fcs)
        index = {c: i for i, (c, _) in enumerate(count.most_common(600))}
        m = np.zeros((len(texts), len(index)))
        for i, t in enumerate(texts):
            for c in t:
                j = index.get(c)
                if j is not None:
                    m[i, j] += 1
        self.m = np.log1p(m / np.maximum(lengths[:, None], 1) * 1000)
        self.content_vectors = self.chap(self.m)
        means = np.stack([self.m[ch == c].mean(0) for c in chapters])
        self.cpc = PCA(10, random_state=0).fit_transform((means - means.mean(0)) / (means.std(0) + 1e-9))

    def chap(self, e: np.ndarray, lc: bool = True) -> np.ndarray:
        e = e.astype(float)
        if lc:
            z = np.c_[self.L, np.ones(len(self.L))]
            e = e - z @ np.linalg.lstsq(z, e, rcond=None)[0]
        v = np.stack([e[self.ch == c].mean(0) for c in range(1, 121)])
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def function_chars(self) -> np.ndarray:
        fc = list("之其或亦方即皆仍故尚呢了的着著吗嗎么麼罢罷吧便就也矣乎焉哉而以於于乃则則但却卻又且都将將把被叫让讓")
        m = np.array([[t.count(f) / max(len(t), 1) for f in fc] for t in self.texts])
        return (m - m.mean(0)) / (m.std(0) + 1e-9)

    def remove(self, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        z = np.c_[z, np.ones(len(z))]
        r = v - z @ np.linalg.lstsq(z, v, rcond=None)[0]
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    def resid(self, v: np.ndarray) -> np.ndarray:
        return self.remove(v, self.freg)

    def variants(self, v: np.ndarray) -> dict[str, np.ndarray]:
        return {"as_is": v, "minus_register": self.remove(v, self.freg), "minus_content": self.remove(v, self.cpc),
                "minus_register_minus_content": self.remove(v, np.c_[self.freg, self.cpc])}


def scan(v: np.ndarray, first: int = 1) -> tuple[int, dict]:
    w = v[first - 1:]
    n = len(w)
    scores = {}
    for k in range(2, n - 1):
        a, b = w[:k].mean(0), w[k:].mean(0)
        scores[first - 1 + k] = 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b)
    return max(scores, key=scores.get), scores


def perm_p(v: np.ndarray, first: int, observed: float, n: int, rng) -> float:
    w = v[first - 1:]
    return float(np.mean([max(scan(w[rng.permutation(len(w))], 1)[1].values()) >= observed for _ in range(n)]))


def placebo_probs(v: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    train = list(range(b - 40, b - 10)) + list(range(b + 11, b + 41))
    held = list(range(b - 10, b + 11))
    model = LogisticRegression(C=1.0, max_iter=2000).fit(v[[c - 1 for c in train]], np.array([c > b for c in train]))
    return np.array([c > b for c in held]), model.predict_proba(v[[c - 1 for c in held]])[:, 1]


def placebo(v: np.ndarray) -> dict:
    curve = {b: float(roc_auc_score(*placebo_probs(v, b))) for b in range(41, 81)}
    nulls = [curve[b] for b in CLEAN_NULLS]
    return {"auc_80": curve[80], "null_max": max(nulls), "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
            "z": (curve[80] - float(np.mean(nulls))) / float(np.std(nulls)), "by_boundary": {str(b): curve[b] for b in curve}}


def run_reanalysis(study: Reanalysis, methods: dict[str, np.ndarray], perm_names: tuple[str, ...], rng) -> dict:
    """The order of the permutation draws follows the old script, so its p-values come back exactly."""
    out = {}
    tops = {}
    for name, e in methods.items():
        v = study.chap(e)
        splits = {}
        for first in (1, 6, 12):
            k, scores = scan(v, first)
            ranked = sorted(scores, key=scores.get, reverse=True)
            splits[str(first)] = {"split_after": k, "p": perm_p(v, first, scores[k], 200, rng), "next_best": ranked[1:4], "rank_of_80": ranked.index(80) + 1, "of": len(ranked)}
        minus = {}
        vr = study.resid(v)
        for first in (1, 6):
            k, scores = scan(vr, first)
            ranked = sorted(scores, key=scores.get, reverse=True)
            minus[str(first)] = {"split_after": k, "rank_of_80": ranked.index(80) + 1}
        out[name] = {"split": splits, "split_minus_register": minus}
    for name in perm_names:
        vr = study.resid(study.chap(methods[name]))
        k, scores = scan(vr, 1)
        out[name]["split_minus_register"]["1"]["p"] = perm_p(vr, 1, scores[k], 500, rng)
    upper = np.triu_indices(120, 1)
    content = (study.content_vectors @ study.content_vectors.T)[upper]
    for name, e in methods.items():
        v = study.chap(e)
        out[name]["placebo"] = {label: placebo(variant) for label, variant in study.variants(v).items()}
        out[name]["content_similarity_r"] = float(np.corrcoef((v @ v.T)[upper], content)[0, 1])
    return out


def auc_boot(a: np.ndarray, b: np.ndarray, y: np.ndarray, seed: int = 42, boots: int = 4000) -> dict:
    """Not in the old test: bootstrap over the 21 scored chapters, classifiers fixed. Reads spread, not significance."""
    rng = np.random.default_rng(seed)
    one, diff = [], []
    while len(one) < boots:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        one.append(roc_auc_score(y[i], a[i]))
        diff.append(one[-1] - roc_auc_score(y[i], b[i]))
    return {"auc": float(roc_auc_score(y, a)), "auc_ci95": [float(q) for q in np.quantile(one, [.025, .975])],
            "difference": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "difference_ci95": [float(q) for q in np.quantile(diff, [.025, .975])]}


# --- how far the book is from the training lines ---------------------------------------------------------------------

def domain_check(e: np.ndarray, ch: np.ndarray, lengths: np.ndarray) -> dict:
    train = np.array([trait_vector(r["scores"]) for r in rows_of("genshin_train.jsonl")])
    early, late = e[ch <= 80], e[ch > 80]
    chapters = np.stack([e[ch == c].mean(0) for c in range(1, 121)])
    out = {}
    for k, name in enumerate(NAMES):
        a, b = chapters[:80, k], chapters[80:, k]
        pooled = np.sqrt((a.var(ddof=1) * 79 + b.var(ddof=1) * 39) / 118)
        out[name] = {"train_mean": float(train[:, k].mean()), "train_sd": float(train[:, k].std()), "book_mean": float(e[:, k].mean()), "book_sd": float(e[:, k].std()),
                     "outside_train_range": float(((e[:, k] < train[:, k].min()) | (e[:, k] > train[:, k].max())).mean()),
                     "passage_mean_1_80": float(early[:, k].mean()), "passage_mean_81_120": float(late[:, k].mean()),
                     "chapter_d_81_vs_1": float((b.mean() - a.mean()) / pooled), "r_log_length": float(np.corrcoef(e[:, k], np.log(lengths))[0, 1])}
    return out


def sanitize(obj):
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else round(float(obj), 5)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    passages = json.loads(resolve_repo_path(f"{RESULTS}/passages.json").read_text(encoding="utf-8"))
    cache = resolve_repo_path(f"{RESULTS}/cache")
    stored = pd.read_parquet(cache / "passages.parquet")
    texts = [p["text"] for p in passages]
    chapters = [p["chapter"] for p in passages]
    lengths = [p["length"] for p in passages]
    verse = [p["is_verse"] for p in passages]
    assert list(stored["chapter"]) == chapters and list(stored["length"]) == lengths and len(texts) == 6851
    masked = [mask_names(t) for t in texts]
    ch, lens = np.array(chapters), np.array(lengths)
    dialogue = dialogue_share_per_chapter(chapters, [p["has_dialogue"] for p in passages])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(resolve_repo_path(BASE_MODEL)), local_files_only=True, trust_remote_code=True)
    variants = {"": split_all(texts, tokenizer), "_masked": split_all(masked, tokenizer)}
    tokens = np.bincount(variants[""]["passage"], variants[""]["tokens"], len(texts))
    assert max(p["tokens"].max() for p in variants.values()) <= MAX_TOKENS
    data = {"passages": len(texts), "chars_min_max": [int(lens.min()), int(lens.max())], "passage_tokens_pct_50_90_99": [float(q) for q in np.percentile(tokens, [50, 90, 99])],
            "passages_over_96_tokens": float((tokens > MAX_TOKENS).mean()), "pieces": len(variants[""]), "pieces_masked": len(variants["_masked"]),
            "chars_lost_by_truncation_if_not_split": float(1 - sum(len(variants[""]["piece"][i]) for i in np.flatnonzero(~variants[""]["passage"].duplicated().to_numpy())) / lens.sum())}
    print(data, flush=True)

    embeddings = {}
    first_piece = {}
    for run in RUNS:
        y = cached_predictions(run, variants, args.batch_size)
        embeddings[run] = per_passage(variants[""], y[""], len(texts))
        embeddings[run + "_masked"] = per_passage(variants["_masked"], y["_masked"], len(texts))
        first_piece[run] = per_passage(variants[""], y[""], len(texts), first_only=True)

    print("old methods", flush=True)
    sty, sty_masked = stylometry_vectors(texts), stylometry_vectors(masked)
    load = lambda name: np.load(cache / f"{name}.npy").astype(np.float32)  # noqa: E731
    old = {"stylometry": sty, "stylometry_masked": sty_masked, "base_qwen3": load("base_qwen3"), "base_qwen3_masked": load("base_qwen3_masked"),
           "trained_full": load("trained_full"), "trained_full_masked": load("trained_full_masked"), "trained_dim_64": load("trained_dim_64"), "trained_dim_64_masked": load("trained_dim_64_masked")}
    result = {"data": data, "primary": PRIMARY, "probe": {}, "probe_placebo": {}, "p_late": {}}
    for name, e in {**old, **embeddings}.items():
        print(f"probe {name}", flush=True)
        probe = probe_analysis(e, chapters, lengths, verse, dialogue)
        result["probe"][name] = probe
        result["p_late"][name] = p_late_summary(probe)
        if not name.endswith("_masked"):
            result["probe_placebo"][name] = {"raw": probe_placebo(e, chapters, lengths, False), "length_corrected": probe_placebo(e, chapters, lengths, True)}

    print("reanalysis", flush=True)
    study = Reanalysis(np.array(texts, dtype=object), ch, lens)
    old_methods = {"trained_256": old["trained_full"], "trained_256_masked": old["trained_full_masked"], "base_qwen": old["base_qwen3"], "function_chars": study.function_chars()}
    result["reanalysis"] = {"old": run_reanalysis(study, old_methods, ("trained_256", "trained_256_masked", "base_qwen"), np.random.default_rng(0))}
    for run in RUNS:
        methods = {run: embeddings[run], run + "_masked": embeddings[run + "_masked"]}
        result["reanalysis"][run] = run_reanalysis(study, methods, tuple(methods), np.random.default_rng(0))
    k, scores = scan(study.content_vectors, 1)
    ranked = sorted(scores, key=scores.get, reverse=True)
    result["reanalysis"]["content_chars_only"] = {"split_after": k, "next_best": ranked[1:4], "rank_of_80": ranked.index(80) + 1, "placebo": placebo(study.content_vectors)}

    y, old_probs = placebo_probs(study.chap(old["trained_full"]), 80)
    _, base_probs = placebo_probs(study.chap(old["base_qwen3"]), 80)
    _, function_probs = placebo_probs(study.chap(old_methods["function_chars"]), 80)
    result["auc_80_bootstrap_not_in_old_test"] = {}
    for run in RUNS:
        for suffix in ("", "_masked"):
            _, probs = placebo_probs(study.chap(embeddings[run + suffix]), 80)
            result["auc_80_bootstrap_not_in_old_test"][run + suffix] = {against: auc_boot(probs, other, y) for against, other in
                                                                        (("old_trained_256", old_probs), ("base_qwen", base_probs), ("function_chars", function_probs))}
    result["sensitivity_first_piece_only"] = {run: {"placebo_as_is": placebo(study.chap(first_piece[run])), "p_late": p_late_summary({"length_corrected": (
        lambda lr: {"p_late": dict(zip(map(str, lr["chapter_range"]), lr["p_late"])), "flip_chapter": lr["flip_chapter"]})(
        logistic_probe_range(chapter_vectors(regress_out_length(first_piece[run], lengths), chapters)))})} for run in RUNS}
    result["domain"] = {run: domain_check(embeddings[run], ch, lens) for run in RUNS}

    out = resolve_repo_path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nreanalysis placebo: AUC at 80 (z against clean nulls, null max)")
    tables = {"old": list(old_methods), **{run: [run, run + "_masked"] for run in RUNS}}
    for group, names in tables.items():
        for name in names:
            row = result["reanalysis"][group][name]["placebo"]
            print(f"  {name:62s} " + "  ".join(f"{label}: {row[label]['auc_80']:.2f} ({row[label]['z']:+.1f}, {row[label]['null_max']:.2f})" for label in row))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
