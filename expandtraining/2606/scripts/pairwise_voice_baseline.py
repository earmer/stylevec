"""Can a model that reads both lines of a pair predict the rater's voice-change score? Exploratory.

    uv run python expandtraining/2606/scripts/pairwise_voice_baseline.py features
    uv run python expandtraining/2606/scripts/pairwise_voice_baseline.py finetune | cv95_features | cv95_finetune

Trains on voice_change_small_rule_labels.jsonl (LLM answers through the rater's rule; small edits only) and scores on
the 95 rated pairs of the frozen 100, with the pre-registered statistic. The two sets share no pairs, and they share
speakers, so a test pair's prediction always comes from a model that never saw its speaker.
features: ridge over the changed characters, frozen pair states, or both, leaving one speaker out.
finetune: the last two layers of the base model on "A newline B", five speaker folds.
cv95_features, cv95_finetune: five speaker-grouped folds over the 95 human pairs, trained on the other folds' human pairs,
alone or with those speakers' rule labels (label 1 untied by scan average); a fold's speakers are out of both. These use the 95 as training data, so they
say whether human judgements can be learned at all, and they are not a clean test of any model picked from them.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import resolve_repo_path

BASE_MODEL = "artifacts/base-models/qwen-3-0.6b"
RATINGS = "data/human_ratings/voice_change_ratings.jsonl"
KEY = "data/human_ratings/voice_change_full_key.jsonl"
RULE_LABELS = "data/human_ratings/voice_change_small_rule_labels.jsonl"
OUT = "data/episodes/external/results/pairwise_voice_baseline.json"
CLASSES = {"particle": "啊呀呢吧嘛哦呐哟啦嘞咯哈唔嗯诶欸哎唉呵嘿哼喔噢", "punct": "…！？!?~～—，。、,.",
           "person": "我你您他她它吾汝奴咱俺朕尔君", "classical": "之乎者也其亦已于矣焉哉乃故若"}


def scan_average(answers: list) -> float:
    """Mean of the numeric LLM answers; x and missing answers do not count."""
    values = [int(str(a)[0]) for a in answers if a not in (None, "x") and str(a)[0].isdigit()]
    return float(np.mean(values)) if values else float("nan")


def untie_label_one(labels: np.ndarray, averages: np.ndarray, touches: np.ndarray) -> np.ndarray:
    """Keeps 0 < 1 < 2 from the rule and spreads label 1 over 0.8-1.2 by scan average, then by the edit touching
    particles or punctuation. Ranks are shared by exact ties."""
    from scipy.stats import rankdata

    out = labels.astype(float).copy()
    at = np.flatnonzero(labels == 1)
    if len(at) > 1:
        filled = np.where(np.isnan(averages[at]), np.nanmean(averages[at]), averages[at])
        rank = rankdata(filled + 1e-3 * touches[at].astype(float))
        out[at] = 0.8 + 0.4 * (rank - rank.min()) / max(rank.max() - rank.min(), 1)
    return out


def touches_tone_chars(a: str, b: str) -> bool:
    removed, added = changed_parts(a, b)
    return any(c in CLASSES["particle"] + CLASSES["punct"] for c in removed + added)


def changed_parts(a: str, b: str) -> tuple[str, str]:
    """The characters only a has and the characters only b has, in order."""
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            removed.append(a[i1:i2])
            added.append(b[j1:j2])
    return "".join(removed), "".join(added)


def diff_counts(a: str, b: str) -> list[float]:
    removed, added = changed_parts(a, b)
    counts = [len(removed), len(added), len(a), len(b), abs(np.log(max(len(a), 1) / max(len(b), 1)))]
    for chars in CLASSES.values():
        counts += [sum(c in chars for c in removed), sum(c in chars for c in added)]
    return [float(x) for x in counts]


def jev_scores() -> dict[str, dict]:
    scores = {}
    for path in ["genshin_val.jsonl", "genshin_train.jsonl", "extra/drift.jsonl", "extra/msynth.jsonl", "extra/xlang.jsonl",
                 "extra/voice_pairs.jsonl"]:
        for line in open(resolve_repo_path(f"expandtraining/2606/outputs/traits/{path}"), encoding="utf-8"):
            row = json.loads(line)
            if isinstance(row.get("scores"), dict):
                scores[row["text"]] = row["scores"]
    return scores


def jev_change(a: dict, b: dict) -> float:
    """The mean absolute trait difference, as in scripts/twin_style_delta.py."""
    return float(np.mean([abs(float(a[name][0]) - float(b[name][0])) for name in a if name in b]))


def load_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_json(resolve_repo_path(RULE_LABELS), lines=True, dtype={"label": str})
    train = train[train["label"].isin(["0", "1", "2"])].reset_index(drop=True)
    train["y"] = train["label"].astype(int)
    train["y_untied"] = untie_label_one(train["y"].to_numpy(),
                                        np.array([scan_average([r, q, c]) for r, q, c in zip(train["run_1"], train["run_2"], train["careful"])]),
                                        np.array([touches_tone_chars(a, b) for a, b in zip(train["A"], train["B"])]))
    train["y_scan"] = [scan_average([r, q, c]) for r, q, c in zip(train["run_1"], train["run_2"], train["careful"])]
    train["y_scan"] = train["y_scan"].fillna(train["y_scan"].mean())
    judged = jev_scores()
    change = train.apply(lambda r: jev_change(judged[r["A"]], judged[r["B"]]) if r["A"] in judged and r["B"] in judged else np.nan, axis=1)
    train["y_jev"] = 3 * change.rank(pct=True).fillna(0.5)  # a rank on the raters' 0-3 span, so the same learning rate fits
    last = {}
    for line in open(resolve_repo_path(RATINGS), encoding="utf-8"):
        row = json.loads(line)
        last[row["id"]] = row
    test = pd.read_json(resolve_repo_path(KEY), lines=True)
    test["voice"] = [last.get(i, {}).get("voice") for i in test["id"]]
    test = test[test["voice"].notna() & ~test["id"].map(lambda i: bool(last.get(i, {}).get("broken")))].reset_index(drop=True)
    test["voice"] = test["voice"].astype(int)
    test["bin"] = pd.cut(test["changed"], [0, .15, .35, .6, 1.01], labels=["small", "mid", "large", "rewrite"],
                         right=False).astype(str)
    assert not set(train["id"]) & set(test["id"]), "train and test share a pair"
    return train, test


def within(score: np.ndarray, y: np.ndarray, cells: list[np.ndarray]) -> float:
    total = weight = 0
    for cell in cells:
        if len(cell) < 5 or len(set(y[cell])) < 2 or len(set(score[cell])) < 2:
            continue
        total += spearmanr(score[cell], y[cell])[0] * len(cell)
        weight += len(cell)
    return total / weight if weight else float("nan")


def report(test: pd.DataFrame, scores: dict[str, np.ndarray], boots: int = 4000) -> dict:
    y = test["voice"].to_numpy()
    cells = [np.flatnonzero((test["bin"] == b) & (test["batch"] == x)) for b in ("small", "mid", "large", "rewrite")
             for x in ("old", "new")]
    cells = [c for c in cells if len(c) >= 5]
    small = (test["bin"] == "small").to_numpy()
    edit = test["changed"].to_numpy()
    rng = np.random.default_rng(20261105)
    samples = [[rng.choice(c, len(c)) for c in cells] for _ in range(boots)]
    edit_boot = np.array([within(edit, y, s) for s in samples])
    out = {}
    for name, score in scores.items():
        boot = np.array([within(score, y, s) for s in samples]) - edit_boot
        out[name] = {"within": round(float(within(score, y, cells)), 3), "small": round(float(spearmanr(score[small], y[small])[0]), 3),
                     "overall": round(float(spearmanr(score, y)[0]), 3),
                     "minus_edit": round(float(within(score, y, cells) - within(edit, y, cells)), 3),
                     "ci99": [round(float(q), 3) for q in np.nanquantile(boot, [.005, .995])]}
        print(name, out[name], flush=True)
    return out


def save(results: dict) -> None:
    path = resolve_repo_path(OUT)
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(json.dumps({**done, **results}, ensure_ascii=False, indent=1), encoding="utf-8")


def load_base():
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = str(resolve_repo_path(BASE_MODEL))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=True, dtype=torch.float32).to(device)
    return tokenizer, model, device


def mean_pool(model, tokens):
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    return (model(**tokens).last_hidden_state * mask).sum(1) / mask.sum(1)


def frozen_states(texts: list[str]) -> np.ndarray:
    import torch

    tokenizer, model, device = load_base()
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), 48):
            tokens = tokenizer(texts[start:start + 48], padding=True, truncation=True, max_length=256,
                               return_tensors="pt").to(device)
            out.append(mean_pool(model, tokens).float().cpu().numpy())
    return np.concatenate(out)


def leave_speaker_out(train: pd.DataFrame, test: pd.DataFrame, x_train, x_test) -> np.ndarray:
    """Each test speaker's pairs are predicted by a ridge fitted without that speaker's rows."""
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prediction = np.zeros(len(test))
    for speaker in sorted(set(test["speaker"])):
        keep = (train["speaker"] != speaker).to_numpy()
        at = (test["speaker"] == speaker).to_numpy()
        model = make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(0, 5, 11)))
        model.fit(x_train[keep], train["y"].to_numpy()[keep])
        prediction[at] = model.predict(x_test[at])
    return prediction


def run_features() -> None:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer

    train, test = load_pairs()
    print(f"train {len(train)} pairs, {train['speaker'].nunique()} speakers; test {len(test)}", flush=True)
    parts = {name: frame.apply(lambda r: changed_parts(r["A"], r["B"]), axis=1) for name, frame in (("train", train), ("test", test))}
    removed = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2).fit([p[0] for p in parts["train"]])
    added = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2).fit([p[1] for p in parts["train"]])

    def diff_x(frame, name):
        counts = np.array([diff_counts(a, b) + [c] for a, b, c in zip(frame["A"], frame["B"], frame["changed"])])
        return hstack([csr_matrix(counts), removed.transform([p[0] for p in parts[name]]),
                       added.transform([p[1] for p in parts[name]])]).tocsr()

    def pair_states(frame):
        return frozen_states([f"{a}\n{b}" for a, b in zip(frame["A"], frame["B"])])

    def bi_states(frame):
        states = frozen_states(list(frame["A"]) + list(frame["B"]))
        a, b = states[:len(frame)], states[len(frame):]
        return np.hstack([np.abs(a - b), a * b])

    d_train, d_test = diff_x(train, "train"), diff_x(test, "test")
    p_train, p_test = pair_states(train), pair_states(test)
    b_train, b_test = bi_states(train), bi_states(test)
    scores = {"edit": test["changed"].to_numpy(),
              "diff_ridge": leave_speaker_out(train, test, d_train, d_test),
              "bi_frozen_ridge": leave_speaker_out(train, test, b_train, b_test),
              "cross_frozen_ridge": leave_speaker_out(train, test, p_train, p_test),
              "diff_plus_cross": leave_speaker_out(train, test, hstack([d_train, csr_matrix(p_train)]).tocsr(),
                                                   hstack([d_test, csr_matrix(p_test)]).tocsr())}
    save({"features": report(test, scores)})


def fresh_model(seed: int):
    """The base model with only its last two layers trainable, and a new regression head."""
    import torch

    torch.manual_seed(seed)
    tokenizer, model, device = load_base()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.layers[-2:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
    return tokenizer, model, head, device


def fit(tokenizer, model, head, device, rows: pd.DataFrame, target: str, epochs: int, batch: int, lr: float, seed: int, tag: str) -> None:
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(rows))
        losses = []
        for start in range(0, len(order), batch):
            part = rows.iloc[order[start:start + batch]]
            tokens = tokenizer([f"{a}\n{b}" for a, b in zip(part["A"], part["B"])], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(device)
            y = torch.tensor(part[target].to_numpy(), dtype=torch.float32, device=device)
            loss = torch.nn.functional.mse_loss(head(mean_pool(model, tokens)).squeeze(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"{tag} epoch {epoch} rows {len(rows)} loss {np.mean(losses):.3f}", flush=True)


def predict(tokenizer, model, head, device, rows: pd.DataFrame) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            part = rows.iloc[start:start + 32]
            tokens = tokenizer([f"{a}\n{b}" for a, b in zip(part["A"], part["B"])], padding=True, truncation=True,
                               max_length=256, return_tensors="pt").to(device)
            out.append(head(mean_pool(model, tokens)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def release(*objects) -> None:
    import torch

    del objects
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_finetune(target: str = "y_untied", epochs: int = 3, folds: int = 5, batch: int = 16, seed: int = 0) -> None:
    from sklearn.model_selection import GroupKFold

    train, test = load_pairs()
    prediction = np.zeros(len(test))
    speakers = np.array(sorted(set(train["speaker"]) | set(test["speaker"])))
    fold_of = {s: k for k, (_, at) in enumerate(GroupKFold(folds).split(speakers, groups=speakers)) for s in speakers[at]}
    for fold in range(folds):
        tokenizer, model, head, device = fresh_model(seed + fold)
        fit(tokenizer, model, head, device, train[train["speaker"].map(fold_of) != fold].reset_index(drop=True), target,
            epochs, batch, 2e-5, seed + fold, f"fold {fold}")
        at = np.flatnonzero(test["speaker"].map(fold_of).to_numpy() == fold)
        prediction[at] = predict(tokenizer, model, head, device, test.iloc[at])
        release(model, head)
    save({f"finetune_{target}": report(test, {"edit": test["changed"].to_numpy(), "cross_finetuned": prediction})})


def speaker_folds(speakers: list[str], folds: int, seed: int) -> dict[str, int]:
    """Shuffled round-robin, so each repeat holds out different speaker groups."""
    order = list(np.random.default_rng(seed).permutation(sorted(speakers)))
    return {speaker: k % folds for k, speaker in enumerate(order)}


def run_cv_features(folds: int = 5, repeats: int = 20, human_weight: float = 10.0) -> None:
    """Ridge on the human ratings of the other folds' speakers, alone or with the rule labels of those speakers."""
    from scipy.sparse import csr_matrix, hstack, vstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, test = load_pairs()
    parts = {"rules": [changed_parts(a, b) for a, b in zip(train["A"], train["B"])],
             "human": [changed_parts(a, b) for a, b in zip(test["A"], test["B"])]}
    removed = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2).fit([p[0] for p in parts["rules"]])
    added = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2).fit([p[1] for p in parts["rules"]])

    def diff_x(frame, part):
        counts = np.array([diff_counts(a, b) + [c] for a, b, c in zip(frame["A"], frame["B"], frame["changed"])])
        return hstack([csr_matrix(counts), removed.transform([p[0] for p in part]), added.transform([p[1] for p in part])]).tocsr()

    def cross(frame):
        return csr_matrix(frozen_states([f"{a}\n{b}" for a, b in zip(frame["A"], frame["B"])]))

    d_rules, d_human = diff_x(train, parts["rules"]), diff_x(test, parts["human"])
    c_rules, c_human = cross(train), cross(test)
    sets = {"diff": (d_rules, d_human), "cross": (c_rules, c_human),
            "diff_plus_cross": (hstack([d_rules, c_rules]).tocsr(), hstack([d_human, c_human]).tocsr())}
    y_rules, y_human = train["y_untied"].to_numpy(dtype=float), test["voice"].to_numpy(dtype=float)
    speakers = sorted(set(test["speaker"]))
    cells = [np.flatnonzero((test["bin"] == b) & (test["batch"] == x)) for b in ("small", "mid", "large", "rewrite") for x in ("old", "new")]
    scores, spread = {"edit": test["changed"].to_numpy()}, {}
    for name, (x_rules, x_human) in sets.items():
        for source in ("human_only", "rules_plus_human"):
            runs = []
            for repeat in range(repeats):
                fold_of = speaker_folds(speakers, folds, repeat)
                groups = test["speaker"].map(fold_of).to_numpy()
                prediction = np.zeros(len(test))
                for fold in range(folds):
                    keep, at = groups != fold, groups == fold
                    model = make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(0, 5, 11)))
                    if source == "human_only":
                        model.fit(x_human[keep], y_human[keep])
                    else:
                        held = set(test["speaker"][at])
                        use = (~train["speaker"].isin(held)).to_numpy()
                        weight = np.r_[np.ones(use.sum()), np.full(keep.sum(), human_weight)]
                        model.fit(vstack([x_rules[use], x_human[keep]]).tocsr(), np.r_[y_rules[use], y_human[keep]],
                                  ridgecv__sample_weight=weight)
                    prediction[at] = model.predict(x_human[at])
                runs.append(prediction)
            key = f"{name}/{source}"
            scores[key] = np.mean(runs, axis=0)
            one = [within(r, y_human, cells) for r in runs]
            spread[key] = [round(float(np.mean(one)), 3), round(float(np.std(one)), 3)]
            print(key, "single-repeat within mean, sd", spread[key], flush=True)
    save({"cv95_features": report(test, scores), "cv95_features_single_repeat": spread})


def run_cv_finetune(folds: int = 5, repeats: int = 3) -> None:
    """The last two layers on the other folds' human pairs, alone or after their speakers' rule labels."""
    train, test = load_pairs()
    speakers = sorted(set(test["speaker"]))
    scores = {"edit": test["changed"].to_numpy()}
    y = test["voice"].to_numpy()
    cells = [np.flatnonzero((test["bin"] == b) & (test["batch"] == x)) for b in ("small", "mid", "large", "rewrite") for x in ("old", "new")]
    spread = {}
    for source in ("human_only", "rules_then_human"):
        runs = []
        for repeat in range(repeats):
            fold_of = speaker_folds(speakers, folds, repeat)
            groups = test["speaker"].map(fold_of).to_numpy()
            prediction = np.zeros(len(test))
            for fold in range(folds):
                tag = f"{source} r{repeat} f{fold}"
                tokenizer, model, head, device = fresh_model(100 * repeat + fold)
                if source == "rules_then_human":
                    held = set(test["speaker"][groups == fold])
                    rules = train[~train["speaker"].isin(held)].reset_index(drop=True)
                    fit(tokenizer, model, head, device, rules, "y_untied", 3, 16, 2e-5, 100 * repeat + fold, tag)
                human = test[groups != fold].reset_index(drop=True)
                fit(tokenizer, model, head, device, human, "voice", 6 if source == "human_only" else 3, 8,
                    2e-5 if source == "human_only" else 1e-5, 100 * repeat + fold, tag)
                at = np.flatnonzero(groups == fold)
                prediction[at] = predict(tokenizer, model, head, device, test.iloc[at])
                release(model, head)
            runs.append(prediction)
            one = [within(r, y, cells) for r in runs]
            spread[source] = [round(float(np.mean(one)), 3), round(float(np.std(one)), 3)]
            print(source, "repeats", repeat + 1, "within mean, sd", spread[source], flush=True)
            scores[f"cross_finetuned/{source}"] = np.mean(runs, axis=0)
            save({"cv95_finetune": report(test, scores, boots=1000), "cv95_finetune_single_repeat": spread})


MODES = {"features": run_features, "finetune": run_finetune, "cv95_features": run_cv_features, "cv95_finetune": run_cv_finetune}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=list(MODES))
    parser.add_argument("--target", default="y_untied", choices=["y_untied", "y_scan", "y_jev"], help="finetune only")
    args = parser.parse_args()
    run_finetune(args.target) if args.mode == "finetune" else MODES[args.mode]()


if __name__ == "__main__":
    main()
