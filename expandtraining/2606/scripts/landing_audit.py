"""Local audit before any multilingual or real-world training: what the trait predictor trained on, how it behaves zero-shot on en and ja, and on real-world text.

    uv run python expandtraining/2606/scripts/landing_audit.py sample     # writes the Jev input for the Japanese recheck
    uv run python expandtraining/2606/scripts/landing_audit.py run

Fixed before scoring. Health on real text: every trait SD at least half its SD on unseen Genshin lines; formality and archaic higher on aozora than on reddit, weibo and douban;
archaic highest on aozora; spokenness higher on cped than on aozora and authbench. Voice over topic: the primary distance is the predicted change (mean absolute trait difference), centred cosine is
also shown. V is accuracy on an item set, T on its speaker-shuffled twin (voice removed, text kept); data/episodes/external and reddit_probe are the items. Descriptive throughout, one seed.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from config import resolve_repo_path
from hongloumeng_trait_test import free_gpu, predict_pieces, sanitize
from trait_predictor import DB, NAMES, OUT_DIR, TRAIT_DIR, load_run, predicted_change, rows_of, trait_vector

LORA = "lora_r8_delta1.0_s42_attention"
FROZEN = "frozen_delta1.0_s42_attention_layers6-12-18-24-28"
OUT = "data/episodes/external/results/landing_audit.json"
CACHE = "expandtraining/2606/outputs/landing_audit"
JA_IN, JA_OUT = f"{TRAIT_DIR}/extra/ja_recheck_in.jsonl", f"{TRAIT_DIR}/extra/ja_recheck.jsonl"
SEED = 20261106
PARALLEL, PARAS, RETEST, UNSEEN_JA, PER_SOURCE, ITEMS, BOOTS = 3000, 1500, 200, 200, 1500, 1500, 1000
SOURCES = [("cped", "zh"), ("weibo", "zh"), ("douban_short", "zh"), ("douban_side", "zh"), ("commentr", "zh"), ("authbench", "zh"), ("reddit", "en"), ("ao3", "en"), ("authbench", "en"), ("authbench", "ja"), ("aozora", "ja")]
ITEM_SETS = ("cped", "reddit", "authbench", "aozora")
KANA = re.compile("[぀-ヿ]")
CJK = re.compile("[一-鿿]")
LATIN = re.compile("[A-Za-z]")


def language(text: str) -> str:
    if KANA.search(text):
        return "ja"
    latin, cjk = len(LATIN.findall(text)), len(CJK.findall(text))
    return "en" if latin > cjk else "zh" if cjk else "other"


def seen_texts() -> set[str]:
    seen = {r["text"] for name in ("genshin_train.jsonl", "genshin_val.jsonl") for r in rows_of(name)}
    for r in rows_of("trait_pairs_train.jsonl"):
        seen |= {r["a"], r["b"]}
    return seen


def rows_from_db() -> pd.DataFrame:
    query = """select d.id,d.speaker,d.origin_text zh,e.origin_text en,j.origin_text ja,d.para_text zh_para,e.para_text en_para,j.para_text ja_para,d.dup_of from dialogues d
               join dialogues_en e on d.id=e.id join dialogues_ja j on d.id=j.id"""
    return pd.read_sql_query(query, sqlite3.connect(resolve_repo_path(DB)))


def unseen(frame: pd.DataFrame, seen: set[str]) -> pd.DataFrame:
    said = frame.groupby("speaker")["zh"].agg(set)
    known = {s for s, lines in said.items() if lines & seen}
    ratings = resolve_repo_path("data/human_ratings")
    known |= {json.loads(line).get("speaker") for name in ("voice_change_full_key.jsonl", "voice_change_extra_key.jsonl") for line in open(ratings / name, encoding="utf-8")}
    good = frame["dup_of"].isna() & ~frame["speaker"].isin(known) & ~frame["speaker"].str.contains("[?？]", na=True) & frame["zh"].str.len().between(8, 120)
    return frame[good & ~frame["zh"].isin(seen) & frame["en"].fillna("").ne("") & frame["ja"].fillna("").ne("")].reset_index(drop=True)


def samples() -> dict:
    seen = seen_texts()
    frame = rows_from_db()
    pool = unseen(frame, seen)
    parallel = pool.sample(min(PARALLEL, len(pool)), random_state=SEED)
    has_para = pool[pool["zh_para"].fillna("").ne("") & pool["en_para"].fillna("").ne("") & pool["ja_para"].fillna("").ne("") & (pool["zh_para"] != pool["zh"]) & (pool["en_para"] != pool["en"]) & (pool["ja_para"] != pool["ja"]) & ~pool["zh_para"].isin(seen)]
    paras = has_para.sample(min(PARAS, len(has_para)), random_state=SEED)
    return {"frame": frame, "seen": seen, "pool": pool, "parallel": parallel, "paras": paras, "has_para": len(has_para)}


def write_ja_input() -> None:
    made = samples()
    xlang = [r for r in rows_of("extra/xlang.jsonl") if r["lang"] == "ja"]
    random.Random(SEED).shuffle(xlang)
    rows = [{"text": r["text"], "group": "retest"} for r in xlang[:RETEST]]
    rows += [{"text": t, "group": "unseen"} for t in made["parallel"]["ja"].head(UNSEEN_JA)]
    path = resolve_repo_path(JA_IN)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{path}: {len(rows)} rows, unseen pool {len(made['pool'])} lines of {made['pool']['speaker'].nunique()} speakers, parallel paraphrases {made['has_para']}")


def audit(made: dict) -> dict:
    frame = made["frame"]
    origin, para = set(frame["zh"].dropna()), set(frame["zh_para"].dropna())
    en, ja = set(frame["en"].dropna()) | set(frame["en_para"].dropna()), set(frame["ja"].dropna()) | set(frame["ja_para"].dropna())

    def domain(text: str) -> str:
        return "genshin_zh_line" if text in origin else "genshin_zh_paraphrase" if text in para else "genshin_en" if text in en else "genshin_ja" if text in ja else "other"

    train = [r["text"] for r in rows_of("genshin_train.jsonl")]
    pairs = rows_of("trait_pairs_train.jsonl")
    pair_texts = [t for r in pairs for t in (r["a"], r["b"])]
    out = {"train_lines": len(train), "train_pairs": len(pairs), "lines_language": collections.Counter(map(language, train)), "lines_domain": collections.Counter(map(domain, train)),
           "pair_texts_language": collections.Counter(map(language, pair_texts)), "pair_texts_domain": collections.Counter(map(domain, pair_texts)), "pair_source": collections.Counter(r.get("source") for r in pairs),
           "other_examples": [t[:60] for t in train + pair_texts if domain(t) == "other"][:8]}
    out["db"] = {lang: {"lines": int(frame[lang].fillna("").ne("").sum()), "paraphrases": int(frame[f"{lang}_para"].fillna("").ne("").sum()), "speakers": int(frame.loc[frame[lang].fillna("").ne(""), "speaker"].nunique())} for lang in ("zh", "en", "ja")}
    all_three = frame["zh"].fillna("").ne("") & frame["en"].fillna("").ne("") & frame["ja"].fillna("").ne("")
    para_three = frame["zh_para"].fillna("").ne("") & frame["en_para"].fillna("").ne("") & frame["ja_para"].fillna("").ne("")
    out["db"]["aligned_all_three"] = int(all_three.sum())
    out["db"]["aligned_unique_zh"] = int(frame.loc[all_three & frame["dup_of"].isna(), "zh"].nunique())
    out["db"]["paraphrased_in_all_three"] = int(para_three.sum())
    out["unseen_pool"] = {"lines": len(made["pool"]), "speakers": int(made["pool"]["speaker"].nunique())}
    sources = {}
    for name in ("unified.parquet", "unified_new_sources.parquet"):
        d = pd.read_parquet(resolve_repo_path(f"data/authorcorpora/{name}"), columns=["source", "lang", "split", "author_id"])
        sources[name] = {f"{k[0]}/{k[1]}": {"rows": int(len(g)), "authors": int(g["author_id"].nunique()), "heldout_rows": int((g["split"] == "heldout").sum())} for k, g in d.groupby(["source", "lang"])}
    out["sources"] = sources
    return out


class Scores:
    """Predictions by text, per run, saved after each batch of work so a restart reuses them."""

    def __init__(self, run: str) -> None:
        self.run, self.predict = run, None
        self.path = resolve_repo_path(CACHE) / f"{run}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {k: np.array(v) for k, v in json.loads(self.path.read_text(encoding="utf-8")).items()} if self.path.exists() else {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            if self.predict is None:
                _, self.predict = load_run(resolve_repo_path(OUT_DIR) / self.run)
            print(f"{self.run}: predicting {len(missing)} of {len(texts)}", flush=True)
            for text, row in zip(missing, predict_pieces(self.predict, missing, 16)):
                self.cache[text] = row.astype(float)
            self.path.write_text(json.dumps({k: [round(float(x), 4) for x in v] for k, v in self.cache.items()}, ensure_ascii=False), encoding="utf-8")
        return np.stack([self.cache[t] for t in texts])

    def close(self) -> None:
        self.predict = None
        free_gpu()


def per_trait(a: np.ndarray, b: np.ndarray) -> dict:
    """Pearson r, level shift of a from b in b's SD, SD ratio of a to b."""
    rows = {name: {"r": float(np.corrcoef(a[:, k], b[:, k])[0, 1]), "shift_sd": float((a[:, k].mean() - b[:, k].mean()) / b[:, k].std()), "sd_ratio": float(a[:, k].std() / b[:, k].std())} for k, name in enumerate(NAMES)}
    return {"mean_r": float(np.mean([v["r"] for v in rows.values()])), "mean_abs_shift_sd": float(np.mean([abs(v["shift_sd"]) for v in rows.values()])), "mean_sd_ratio": float(np.mean([v["sd_ratio"] for v in rows.values()])), "traits": rows}


def speaker_level(values: dict[str, np.ndarray], speakers: np.ndarray, base: str = "zh", minimum: int = 5) -> dict:
    keep = [s for s, n in collections.Counter(speakers).items() if n >= minimum]
    out = {"speakers": len(keep)}
    means = {lang: np.stack([v[speakers == s].mean(0) for s in keep]) for lang, v in values.items()}
    for lang in means:
        if lang != base:
            out[lang] = float(np.mean([np.corrcoef(means[lang][:, k], means[base][:, k])[0, 1] for k in range(len(NAMES))]))
    return out


def multilingual(made: dict, jev: dict) -> dict:
    parallel, paras = made["parallel"], made["paras"]
    out = {"lines": len(parallel), "speakers": int(parallel["speaker"].nunique()), "paraphrase_lines": len(paras)}
    xlang = rows_of("extra/xlang.jsonl")
    for name, run in (("lora_attention", LORA), ("frozen_taps", FROZEN)):
        scores = Scores(run)
        y = {lang: scores(list(parallel[lang])) for lang in ("zh", "en", "ja")}
        row = {"en_vs_zh": per_trait(y["en"], y["zh"]), "ja_vs_zh": per_trait(y["ja"], y["zh"]), "speaker_means": speaker_level(y, parallel["speaker"].to_numpy())}
        change = {lang: predicted_change(scores(list(paras[lang])), scores(list(paras[f"{lang}_para"]))) for lang in ("zh", "en", "ja")}
        row["paraphrase_change"] = {"mean": {k: float(v.mean()) for k, v in change.items()}, "rho_en_zh": float(spearmanr(change["en"], change["zh"])[0]), "rho_ja_zh": float(spearmanr(change["ja"], change["zh"])[0])}
        for lang in ("en", "ja"):
            part = [r for r in xlang if r["lang"] == lang]
            label = np.array([trait_vector(r["scores"]) for r in part])
            self_y, zh_y = scores([r["text"] for r in part]), scores([r["zh_text"] for r in part])
            row[f"against_jev_{lang}"] = {"n": len(part), "model_on_translation": per_trait(self_y, label), "model_on_zh_line": per_trait(zh_y, label)}
            known = [i for i, r in enumerate(part) if r["zh_text"] in jev]
            if known:
                jev_zh = np.array([trait_vector(jev[part[i]["zh_text"]]) for i in known])
                row[f"against_jev_{lang}"]["jev_zh_vs_jev_translation"] = {"n": len(known), **per_trait(jev_zh, label[known])}
                row[f"against_jev_{lang}"]["model_zh_vs_jev_zh"] = {"n": len(known), **per_trait(zh_y[known], jev_zh)}
        out[name] = row
        scores.close()
    return out


def ja_drift(made: dict) -> dict:
    path = resolve_repo_path(JA_OUT)
    if not path.exists():
        return {"skipped": "no Jev recheck file"}
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [r for r in rows if isinstance(r.get("scores"), dict)]
    out = {"labelled": len(rows)}
    old = {r["text"]: trait_vector(r["scores"]) for r in rows_of("extra/xlang.jsonl") if r["lang"] == "ja"}
    retest = [r for r in rows if r["group"] == "retest" and r["text"] in old]
    a, b = np.array([trait_vector(r["scores"]) for r in retest]), np.array([old[r["text"]] for r in retest])
    out["ja_retest"] = {"n": len(retest), **per_trait(a, b), "mean_abs_difference": float(np.abs(a - b).mean())}
    drift = [r for r in rows_of("extra/drift.jsonl") if isinstance(r.get("scores"), dict) and isinstance(r.get("old_scores"), dict)]
    if drift:
        x, z = np.array([trait_vector(r["scores"]) for r in drift]), np.array([trait_vector(r["old_scores"]) for r in drift])
        out["zh_retest_earlier"] = {"n": len(drift), **per_trait(x, z), "mean_abs_difference": float(np.abs(x - z).mean())}
    fresh = [r for r in rows if r["group"] == "unseen"]
    label = np.array([trait_vector(r["scores"]) for r in fresh])
    zh_of = dict(zip(made["parallel"]["ja"], made["parallel"]["zh"]))
    for name, run in (("lora_attention", LORA), ("frozen_taps", FROZEN)):
        scores = Scores(run)
        out[f"{name}_unseen_ja"] = {"n": len(fresh), "model_on_ja": per_trait(scores([r["text"] for r in fresh]), label), "model_on_zh_line": per_trait(scores([zh_of[r["text"]] for r in fresh]), label)}
        scores.close()
    return out


def real_world(scores: Scores) -> dict:
    out, genshin = {}, np.stack(list(scores.cache.values()))
    frames = {"unified.parquet": pd.read_parquet(resolve_repo_path("data/authorcorpora/unified.parquet"), filters=[("split", "=", "heldout")], columns=["source", "lang", "text", "n_chars"]),
              "unified_new_sources.parquet": pd.read_parquet(resolve_repo_path("data/authorcorpora/unified_new_sources.parquet"), filters=[("split", "=", "heldout"), ("source", "=", "aozora")], columns=["source", "lang", "text", "n_chars"])}
    frame = pd.concat(frames.values(), ignore_index=True)
    reference = scores(list(rows_from_db().pipe(lambda d: d[d["dup_of"].isna()]).sample(PER_SOURCE, random_state=SEED)["zh"]))
    means, spread = {"genshin/zh": reference.mean(0)}, {"genshin/zh": reference.std(0)}
    for source, lang in SOURCES:
        part = frame[(frame["source"] == source) & (frame["lang"] == lang) & frame["n_chars"].between(10, 300)].drop_duplicates("text")
        part = part.sample(min(PER_SOURCE, len(part)), random_state=SEED)
        y = scores(list(part["text"]))
        key = f"{source}/{lang}"
        means[key], spread[key] = y.mean(0), y.std(0)
        out[key] = {"n": len(part), "mean": dict(zip(NAMES, map(float, y.mean(0)))), "sd": dict(zip(NAMES, map(float, y.std(0)))), "min_sd_ratio": float((y.std(0) / reference.std(0)).min()), "worst_trait": NAMES[int((y.std(0) / reference.std(0)).argmin())],
                    "traits_below_half_sd": [NAMES[k] for k in np.flatnonzero(y.std(0) < .5 * reference.std(0))], "r_length": float(np.mean([abs(np.corrcoef(y[:, k], np.log(part["n_chars"].to_numpy(float)))[0, 1]) for k in range(len(NAMES))]))}
    at = {name: k for k, name in enumerate(NAMES)}
    get = lambda key, name: float(means[key][at[name]])  # noqa: E731
    informal = ["reddit/en", "weibo/zh", "douban_short/zh", "douban_side/zh"]
    checks = {"formality_aozora_above_informal": all(get("aozora/ja", "formality") > get(k, "formality") for k in informal), "archaic_aozora_above_informal": all(get("aozora/ja", "archaic") > get(k, "archaic") for k in informal),
              "archaic_aozora_highest": max(means, key=lambda k: means[k][at["archaic"]]) == "aozora/ja", "spokenness_cped_above_books": all(get("cped/zh", "spokenness") > get(k, "spokenness") for k in ("aozora/ja", "authbench/zh", "authbench/en", "authbench/ja"))}
    return {"sources": out, "reference_sd": dict(zip(NAMES, map(float, reference.std(0)))), "reference_mean": dict(zip(NAMES, map(float, reference.mean(0)))), "checks": checks,
            "rankings": {name: sorted(means, key=lambda k: -means[k][at[name]]) for name in NAMES}}


def wins(q: np.ndarray, p: np.ndarray, n: np.ndarray, mode: str) -> np.ndarray:
    if mode == "change":
        gap = predicted_change(q, n) - predicted_change(q, p)
    else:
        centre = np.r_[q, p, n].mean(0)
        unit = [(x - centre) / np.maximum(np.linalg.norm(x - centre, axis=1, keepdims=True), 1e-9) for x in (q, p, n)]
        gap = (unit[0] * unit[1]).sum(1) - (unit[0] * unit[2]).sum(1)
    return (gap > 0) + .5 * (gap == 0)


def tfidf_wins(items: list[dict]) -> np.ndarray:
    texts = [i[k] for i in items for k in ("query", "positive", "negative")]
    x = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True).fit_transform(texts)
    q, p, n = x[0::3], x[1::3], x[2::3]
    gap = np.asarray(q.multiply(p).sum(1)).ravel() - np.asarray(q.multiply(n).sum(1)).ravel()
    return (gap > 0) + .5 * (gap == 0)


def group_ci(values: np.ndarray, groups: list[str]) -> list[float]:
    """99% interval of the mean of values, resampling groups."""
    frame = pd.DataFrame({"v": values, "g": groups}).groupby("g")["v"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(frame), (BOOTS, len(frame)))
    means = frame[picks, 0].sum(1) / frame[picks, 1].sum(1)
    return [float(q) for q in np.quantile(means, [.005, .995])]


def load_items(name: str) -> list[dict]:
    rows = [json.loads(line) for line in open(resolve_repo_path(f"data/episodes/external/{name}.jsonl"), encoding="utf-8")]
    random.Random(SEED).shuffle(rows)
    return rows[:ITEMS]


def score_items(items: list[dict], scores: Scores) -> dict:
    groups = [i["group"] for i in items]
    q, p, n = (scores([i[k] for i in items]) for k in ("query", "positive", "negative"))
    tf = tfidf_wins(items)
    out = {"items": len(items), "groups": len(set(groups)), "tfidf": float(tf.mean())}
    for mode in ("change", "cosine"):
        w = wins(q, p, n, mode)
        out[mode] = {"accuracy": float(w.mean()), "ci99": group_ci(w, groups), "minus_tfidf": float((w - tf).mean()), "minus_tfidf_ci99": group_ci(w - tf, groups)}
    return out


def voice_topic(scores: Scores) -> dict:
    out = {}
    for name in ITEM_SETS:
        real, shuffled = score_items(load_items(name), scores), score_items(load_items(f"{name}_shuffled"), scores)
        out[name] = {"V": real, "T": shuffled, "V_minus_T_change": real["change"]["accuracy"] - shuffled["change"]["accuracy"]}
        print(name, json.dumps(sanitize(out[name]))[:500], flush=True)
    probe = [json.loads(line) for line in open(resolve_repo_path("data/episodes/reddit_probe.jsonl"), encoding="utf-8")]
    random.Random(SEED).shuffle(probe)
    items = [{"group": i["author"], "query": i["query"], "positive": i["voice"], "negative": i["topic"]} for i in probe[:ITEMS]]
    out["reddit_probe_voice_vs_subreddit"] = score_items(items, scores)
    done = {name: json.loads(resolve_repo_path(f"data/episodes/external/results/{name}.json").read_text(encoding="utf-8")) for name in ITEM_SETS}
    out["head_f_earlier_full_sets"] = {name: {arm: d["scores"][arm]["accuracy"] for arm in d["scores"]} for name, d in done.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "run"])
    args = parser.parse_args()
    if args.command == "sample":
        write_ja_input()
        return
    made = samples()
    jev = {}
    for name in ("genshin_val.jsonl", "genshin_train.jsonl"):
        jev |= {r["text"]: r["scores"] for r in rows_of(name) if isinstance(r.get("scores"), dict)}
    result = {"audit": audit(made)}
    print(json.dumps(sanitize(result["audit"]), ensure_ascii=False)[:1500], flush=True)
    result["multilingual"] = multilingual(made, jev)
    result["ja_drift"] = ja_drift(made)
    scores = Scores(LORA)
    result["real_world"] = real_world(scores)
    result["voice_topic"] = voice_topic(scores)
    resolve_repo_path(OUT).write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
