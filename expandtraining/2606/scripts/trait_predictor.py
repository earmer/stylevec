"""Predict Jev's 17 style traits from one line, then ask whether the predicted change between two lines follows the rater's ear.

    uv run python expandtraining/2606/scripts/trait_predictor.py frozen
    uv run python expandtraining/2606/scripts/trait_predictor.py lora [--steps 3000] [--delta-weight 1.0] [--readout attention] [--layers 6,12,18,24,28]
    uv run python expandtraining/2606/scripts/trait_predictor.py frozen-head [--readout attention] [--scorer-dim 64] [--layers 6,12,18,24,28]
    uv run python expandtraining/2606/scripts/trait_predictor.py small-dev <run dir> [--score-rated]
    uv run python expandtraining/2606/scripts/trait_predictor.py confirm <run dir> [<run dir> ...]
    uv run python expandtraining/2606/scripts/trait_predictor.py argon <run dir> [<run dir> ...]

No speaker or author labels are trained on: the target is style, not who wrote it. The score of a pair is the mean
absolute predicted trait difference, as for Jev's own scores in pairwise_voice_baseline.jev_change.
Train: outputs/traits/genshin_train.jsonl and trait_pairs_train.jsonl. Dev (picks the step, never the 95): lines of
genshin_val.jsonl and their own paraphrases. Every line of the rated pairs' speakers is out of both, so the 95 pairs are
unseen lines of unseen speakers. extra/voice_pairs.jsonl stays out, since it holds those speakers' lines.
frozen: ridge from the base model's mean-pooled state to the traits.
lora: LoRA r=8 on q_proj and v_proj of every layer and a linear head; lines plus the signed trait difference on pairs.
frozen-head: the same training with the base model untouched, so only the readout learns.
--readout attention gives each trait its own softmax over the tokens, starting as mean pooling; --scorer-dim scores
tokens through a small tanh layer instead of directly. --layers mixes those
hidden states with learned weights (model.LayerFusion: the middle ones go through the backbone's final norm).
small-dev: predicted change against the rule, scan and Jev targets on the small-edit pairs of voice_change_small_rule_labels,
which no run trains on. It chooses between readouts, so the 95 are not used for that.
confirm: the sealed test of prereg/trait_confirmation.md, on the fresh block only, plus how the re-rates agree with the first ratings.
argon: the sealed test of prereg/trait_argon_test.md, against Argon's scan on small edits of unseen speakers.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from config import resolve_repo_path
from pairwise_voice_baseline import frozen_states, jev_change, load_base, load_pairs, mean_pool, report, save
from traits import TRAITS

NAMES = list(TRAITS)
TRAIT_DIR = "expandtraining/2606/outputs/traits"
DB = "data/genshin/genshin_v2.db"
OUT_DIR = "expandtraining/2606/outputs/trait_predictor"
RATINGS_DIR = "data/human_ratings"
LLM_ANSWERS = ("voice_change_extra_llm_answers.jsonl", "voice_change_small_careful_answers.jsonl")


def trait_vector(scores: dict) -> list[float]:
    return [float(scores[name][0]) for name in NAMES]


def predicted_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).mean(axis=1)


def is_clean(text: str, speakers_of: dict[str, set], held_speakers: set, held_lines: set) -> bool:
    return text not in held_lines and not speakers_of.get(text, set()) & held_speakers


class AttentionReadout(torch.nn.Module):
    """One query and one value vector per trait. Zero queries give uniform weights, so training starts at mean pooling.
    With zero values too, the queries get their first gradient on the second step."""

    def __init__(self, hidden: int, traits: int, bias: torch.Tensor, scorer_dim: int = 0) -> None:
        super().__init__()
        self.scorer = torch.nn.Sequential(torch.nn.Linear(hidden, scorer_dim), torch.nn.Tanh()) if scorer_dim else torch.nn.Identity()
        features = scorer_dim or hidden
        self.scale = features ** 0.5
        self.queries = torch.nn.Parameter(torch.zeros(traits, features))
        self.values = torch.nn.Parameter(torch.zeros(traits, hidden))
        self.bias = torch.nn.Parameter(bias.clone())
        self.dropout = torch.nn.Dropout(0.1)
        self.entropy_share = 1.0

    def token_weights(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(batch, traits, tokens), each row summing to 1 over the real tokens."""
        scores = self.scorer(hidden) @ self.queries.T / self.scale
        return torch.softmax(scores.masked_fill(~mask.bool().unsqueeze(-1), float("-inf")), dim=1).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = self.token_weights(hidden, mask)
        self.entropy_share = weight_entropy_share(weights.detach(), mask)
        return (self.dropout(weights @ hidden) * self.values).sum(-1) + self.bias


def weight_entropy_share(weights: torch.Tensor, mask: torch.Tensor) -> float:
    """Entropy of the token weights over its maximum, averaged: 1 is mean pooling, near 0 is one token."""
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    return float((entropy / mask.sum(1, keepdim=True).clamp_min(2).float().log()).mean())


def rows_of(name: str) -> list[dict]:
    return [json.loads(line) for line in open(resolve_repo_path(f"{TRAIT_DIR}/{name}"), encoding="utf-8")]


def speakers_by_text() -> tuple[dict[str, set], list[tuple[str, str, str]]]:
    """Who said each line or its paraphrase, and the (origin, paraphrase, speaker) rows."""
    speakers: dict[str, set] = defaultdict(set)
    rows = list(sqlite3.connect(resolve_repo_path(DB)).execute("select origin_text, para_text, speaker from dialogues"))
    for origin, para, speaker in rows:
        speakers[origin].add(speaker)
        if para:
            speakers[para].add(speaker)
    return speakers, rows


def load_data() -> dict:
    _, test = load_pairs()
    speakers, db_rows = speakers_by_text()
    held = set(test["speaker"])
    lines = set(test["A"]) | set(test["B"])
    clean = lambda text: is_clean(text, speakers, held, lines)  # noqa: E731
    train = {r["text"]: trait_vector(r["scores"]) for r in rows_of("genshin_train.jsonl") if clean(r["text"])}
    pairs = [(r["a"], r["b"], trait_vector(r["scores_a"]), trait_vector(r["scores_b"])) for r in rows_of("trait_pairs_train.jsonl")
             if clean(r["a"]) and clean(r["b"])]
    val = {r["text"]: trait_vector(r["scores"]) for r in rows_of("genshin_val.jsonl")}
    dev = {text: y for text, y in val.items() if text in speakers and clean(text)}
    dev_pairs = [(o, p) for o, p, _ in db_rows if o in dev and p in dev and o != p]
    print(f"train lines {len(train)}, train pairs {len(pairs)}, dev lines {len(dev)}, dev pairs {len(dev_pairs)}", flush=True)
    return {"train": train, "pairs": pairs, "dev": dev, "dev_pairs": dev_pairs, "test": test}


def judge_dev(predict, data: dict) -> dict:
    """Mean per-trait r on dev lines and on dev paraphrase differences."""
    dev = data["dev"]
    texts = list(dev)
    predicted = predict(texts)
    y = np.array([dev[t] for t in texts])
    at = {t: i for i, t in enumerate(texts)}
    a = [at[o] for o, _ in data["dev_pairs"]]
    b = [at[p] for _, p in data["dev_pairs"]]
    trait = lambda p, q: float(np.nanmean([pearsonr(p[:, k], q[:, k])[0] for k in range(len(NAMES))]))  # noqa: E731
    return {"line_r": round(trait(predicted, y), 3), "delta_r": round(trait(predicted[a] - predicted[b], y[a] - y[b]), 3)}


def small_edit_pairs(train_texts: set):
    rules, _ = load_pairs()
    return rules[~rules["A"].isin(train_texts) & ~rules["B"].isin(train_texts)].reset_index(drop=True)


def judge_small(predict, pairs) -> dict:
    change = predicted_change(predict(list(pairs["A"])), predict(list(pairs["B"])))
    rho = lambda score, target: round(float(spearmanr(score, pairs[target])[0]), 3)  # noqa: E731
    lengths = pairs["A"].str.len()
    return {"pairs": len(pairs), "length_rho": {"model": round(float(spearmanr(change, lengths)[0]), 3),
                                                "edit": round(float(spearmanr(pairs["changed"], lengths)[0]), 3)},
            **{target: {"model": rho(change, target), "edit": rho(pairs["changed"], target)}
                                    for target in ("y_untied", "y_scan", "y_jev")}}


def score_95(predict, data: dict, name: str) -> None:
    test = data["test"]
    a, b = predict(list(test["A"])), predict(list(test["B"]))
    change = predicted_change(a, b)
    from pairwise_voice_baseline import jev_scores

    judged = jev_scores()
    jev = np.array([jev_change(judged[x], judged[y]) for x, y in zip(test["A"], test["B"])])
    print("vs Jev's own change on the 95:", round(float(spearmanr(change, jev)[0]), 3), flush=True)
    save({name: report(test, {name: change})})


def run_frozen() -> None:
    from sklearn.linear_model import Ridge

    data = load_data()
    texts = list(data["train"])
    x = frozen_states(texts)
    y = np.array([data["train"][t] for t in texts])
    scale = x.std(axis=0) + 1e-6
    model = Ridge(alpha=100.0).fit(x / scale, y)
    predict = lambda t: model.predict(frozen_states(t) / scale)  # noqa: E731
    print("dev", judge_dev(predict, data), flush=True)
    score_95(predict, data, "trait_predictor_frozen_ridge")


def make_encode(tokenizer, encoder, readout, fusion, device, frozen: bool = False):
    norm = encoder.norm if frozen else encoder.get_base_model().norm

    def encode(texts: list[str]) -> torch.Tensor:
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=96, return_tensors="pt").to(device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            output = encoder(**tokens, output_hidden_states=fusion.mode != "last")
            states = output.hidden_states if fusion.mode != "last" else None
        hidden = fusion(output.last_hidden_state, states, norm).float()
        mask = tokens["attention_mask"]
        if isinstance(readout, AttentionReadout):
            return readout(hidden, mask)
        return readout((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True))

    return encode


def make_predict(encode, modules):
    def predict(texts: list[str]) -> np.ndarray:
        for module in modules:
            module.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), len(NAMES)))
        with torch.no_grad():
            for start in range(0, len(order), 64):
                at = order[start:start + 64]
                out[at] = encode([texts[i] for i in at]).cpu().numpy()
        for module in modules:
            module.train()
        return out

    return predict


def make_readout(kind: str, hidden: int, bias: torch.Tensor, device, scorer_dim: int = 0) -> torch.nn.Module:
    if kind == "attention":
        return AttentionReadout(hidden, len(NAMES), bias, scorer_dim).to(device)
    readout = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(hidden, len(NAMES)))
    torch.nn.init.zeros_(readout[1].weight)
    readout[1].bias.data = bias.clone()
    return readout.to(device)


def make_fusion(layers: tuple[int, ...], device):
    from model import LayerFusion

    return LayerFusion("weighted_sum" if layers else "last", layers).to(device)


def load_run(out: Path):
    from peft import PeftModel

    saved = json.loads((out / "dev.json").read_text(encoding="utf-8"))
    tokenizer, base, device = load_base()
    frozen = saved.get("frozen", False)
    encoder = base if frozen else PeftModel.from_pretrained(base, out).to(device)
    bias = torch.zeros(len(NAMES), dtype=torch.float32)
    readout = make_readout(saved.get("readout", "mean"), base.config.hidden_size, bias, device, saved.get("scorer_dim", 0))
    readout.load_state_dict(torch.load(out / "head.pt", map_location=device))
    fusion = make_fusion(tuple(saved.get("layers", ())), device)
    if saved.get("layers"):
        fusion.load_state_dict(torch.load(out / "fusion.pt", map_location=device))
    return saved, make_predict(make_encode(tokenizer, encoder, readout, fusion, device, frozen), [encoder, readout, fusion])


def run_small_dev(run_dir: str, score_rated: bool = False) -> None:
    out = resolve_repo_path(run_dir)
    data = load_data()
    saved, predict = load_run(out)
    train_texts = set(data["train"]) | {t for row in data["pairs"] for t in row[:2]}
    print(json.dumps({"run": out.name, "step": saved["step"], "small_dev": judge_small(predict, small_edit_pairs(train_texts))}), flush=True)
    if score_rated:
        score_95(predict, data, f"trait_predictor_{out.name}")


def run_lora(steps: int, delta_weight: float, seed: int = 42, kind: str = "mean", layers: tuple[int, ...] = (), score_rated: bool = False,
             frozen: bool = False, scorer_dim: int = 0) -> None:
    from peft import LoraConfig, get_peft_model

    data = load_data()
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer, base, device = load_base()
    if frozen:
        encoder = base.requires_grad_(False)
    else:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # 64 short lines take 12 GB without it
        encoder = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]))
    y_all = np.array(list(data["train"].values()))
    head = make_readout(kind, base.config.hidden_size, torch.tensor(y_all.mean(axis=0), dtype=torch.float32, device=device), device, scorer_dim)
    fusion = make_fusion(layers, device)
    trainable = [p for p in encoder.parameters() if p.requires_grad]
    print("trainable", sum(p.numel() for p in trainable) + sum(p.numel() for p in head.parameters()) + sum(p.numel() for p in fusion.parameters()), flush=True)
    groups = [{"params": [*head.parameters(), *fusion.parameters()], "lr": 1e-3}] + ([{"params": trainable, "lr": 2e-4}] if trainable else [])
    optimizer = torch.optim.AdamW(groups, weight_decay=0.05)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / steps))
    lines = list(data["train"].items())

    encode = make_encode(tokenizer, encoder, head, fusion, device, frozen)
    predict = make_predict(encode, [encoder, head, fusion])
    label = ("frozen" if frozen else "lora_r8") + f"_delta{delta_weight}_s{seed}" + (f"_{kind}" if kind != "mean" else "") + (f"_scorer{scorer_dim}" if scorer_dim else "") + (f"_layers{'-'.join(map(str, layers))}" if layers else "")
    train_texts = set(data["train"]) | {t for row in data["pairs"] for t in row[:2]}

    out = resolve_repo_path(OUT_DIR) / label
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    encoder.train()
    for step in range(1, steps + 1):
        chosen = random.sample(lines, 16)
        paired = random.sample(data["pairs"], 8)
        texts = [t for t, _ in chosen] + [t for row in paired for t in row[:2]]
        y = torch.tensor([v for _, v in chosen] + [v for row in paired for v in row[2:]], dtype=torch.float32, device=device)
        predicted = encode(texts)
        delta = predicted[16::2] - predicted[17::2]
        loss = torch.nn.functional.mse_loss(predicted, y) + delta_weight * torch.nn.functional.mse_loss(delta, y[16::2] - y[17::2])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        if step % 50 == 0:
            sharp = f" entropy {head.entropy_share:.3f}" if kind == "attention" else ""
            print(f"{time.strftime('%H:%M:%S')} step {step} loss {loss.item():.3f}{sharp}", flush=True)
        if step % 500 == 0 or step == steps:
            result = judge_dev(predict, data)
            print(f"step {step} dev {result}", flush=True)
            if result["line_r"] + result["delta_r"] > best:
                best = result["line_r"] + result["delta_r"]
                if not frozen:
                    encoder.save_pretrained(out)
                torch.save(head.state_dict(), out / "head.pt")
                if layers:
                    torch.save(fusion.state_dict(), out / "fusion.pt")
                (out / "dev.json").write_text(json.dumps({"step": step, "readout": kind, "layers": list(layers), "frozen": frozen, "scorer_dim": scorer_dim, **result}), encoding="utf-8")
    if not frozen:
        encoder.load_adapter(out, adapter_name="best")
        encoder.set_adapter("best")
    head.load_state_dict(torch.load(out / "head.pt"))
    if layers:
        fusion.load_state_dict(torch.load(out / "fusion.pt"))
    print("best", (out / "dev.json").read_text(encoding="utf-8"), flush=True)
    print("small dev", json.dumps(judge_small(predict, small_edit_pairs(train_texts))), flush=True)
    if score_rated:
        score_95(predict, data, f"trait_predictor_{label}")


def read_ratings(name: str) -> dict:
    """The latest usable voice rating per pair id."""
    last = {r["id"]: r for r in map(json.loads, open(resolve_repo_path(f"{RATINGS_DIR}/{name}"), encoding="utf-8"))}
    return {i: r["voice"] for i, r in last.items() if r["voice"] is not None and not r["broken"]}


def agreement(first: list[int], second: list[int]) -> dict:
    a, b = np.array(first), np.array(second)
    varied = len(a) > 2 and a.std() > 0 and b.std() > 0
    return {"pairs": len(a), "spearman": round(float(spearmanr(a, b)[0]), 3) if varied else None, "exact": round(float((a == b).mean()), 3),
            "within_one": round(float((np.abs(a - b) <= 1).mean()), 3), "mean_second_minus_first": round(float((b - a).mean()), 3)}


def gemini_mean_answers() -> dict:
    """Mean voice digit of every saved Gemini answer per pair id; broken answers (x) have no digit."""
    answers = defaultdict(list)
    for name in LLM_ANSWERS:
        for line in open(resolve_repo_path(f"{RATINGS_DIR}/{name}"), encoding="utf-8"):
            row = json.loads(line)
            if row["answer"][0].isdigit():
                answers[row["id"]].append(int(row["answer"][0]))
    return {i: float(np.mean(v)) for i, v in answers.items()}


def scan_targets(key: pd.DataFrame, answers: list[dict]) -> pd.DataFrame:
    """The pairs with a digit from all three runs, with the three-run and plain-run means. Others are dropped and counted."""
    digits = defaultdict(dict)
    for row in answers:
        if row["answer"][0].isdigit():
            digits[row["id"]][row["run"]] = int(row["answer"][0])
    kept = key[key["id"].map(lambda i: len(digits[i]) == 3)].reset_index(drop=True)
    return kept.assign(scan=[np.mean(list(digits[i].values())) for i in kept["id"]],
                       plain=[np.mean([digits[i]["1"], digits[i]["2"]]) for i in kept["id"]])


def rank_gain(score: np.ndarray, base: np.ndarray, target: np.ndarray, boots: int = 4000, seed: int = 20261105) -> dict:
    """Spearman of score minus base against target, with a paired 99% bootstrap interval over pairs."""
    rng = np.random.default_rng(seed)
    rho = lambda s, t: float(spearmanr(s, t)[0])  # noqa: E731
    samples = [rng.integers(0, len(target), len(target)) for _ in range(boots)]
    diffs = [rho(score[i], target[i]) - rho(base[i], target[i]) for i in samples]
    return {"rho": round(rho(score, target), 3), "gain": round(rho(score, target) - rho(base, target), 3),
            "ci99": [round(float(q), 3) for q in np.nanquantile(diffs, [.005, .995])]}


def run_argon(run_dirs: list[str]) -> None:
    key = pd.read_json(resolve_repo_path(f"{RATINGS_DIR}/argon_test_key.jsonl"), lines=True)
    answers = [json.loads(line) for line in open(resolve_repo_path(f"{RATINGS_DIR}/argon_test_answers.jsonl"), encoding="utf-8")]
    pairs = scan_targets(key, answers)
    edit = pairs["changed"].to_numpy(dtype=float)
    print("pairs", len(pairs), "of", len(key), "scan", pairs["scan"].round(1).value_counts().sort_index().to_dict(), flush=True)
    outcome = {"pairs": len(pairs), "edit": {t: round(float(spearmanr(edit, pairs[t])[0]), 3) for t in ("scan", "plain")}}
    scored = pairs[["id", "scan", "plain", "changed"]].copy()
    for run_dir in run_dirs:
        out = resolve_repo_path(run_dir)
        _, predict = load_run(out)
        change = predicted_change(predict(list(pairs["A"])), predict(list(pairs["B"])))
        scored[out.name] = change
        outcome[out.name] = {t: rank_gain(change, edit, pairs[t].to_numpy()) for t in ("scan", "plain")}
        print(out.name, outcome[out.name], flush=True)
    print("edit", outcome["edit"], flush=True)
    scored.to_json(resolve_repo_path(f"{RATINGS_DIR}/argon_test_scores.jsonl"), orient="records", lines=True)
    save({"trait_argon_test": outcome})


def run_confirm(run_dirs: list[str]) -> None:
    key = pd.read_json(resolve_repo_path(f"{RATINGS_DIR}/voice_change_confirm_key.jsonl"), lines=True)
    blind = read_ratings("voice_change_confirm_ratings.jsonl")
    first = read_ratings("voice_change_ratings.jsonl")
    rated = key[key["id"].isin(blind)].copy()
    rated["voice"] = rated["id"].map(blind).astype(int)
    fresh = rated[rated["block"] == "fresh"].reset_index(drop=True)
    print("fresh", len(fresh), fresh.groupby(["bin", "batch"]).size().to_dict(), "voice", fresh["voice"].value_counts().sort_index().to_dict(), flush=True)
    results = {"edit_size": fresh["changed"].to_numpy(dtype=float)}
    for run_dir in run_dirs:
        out = resolve_repo_path(run_dir)
        _, predict = load_run(out)
        results[out.name] = predicted_change(predict(list(fresh["A"])), predict(list(fresh["B"])))
    outcome = {"fresh": report(fresh, results)}
    gemini = gemini_mean_answers()
    for block in ("retest_95", "retest_gemini"):
        again = rated[(rated["block"] == block) & rated["id"].isin(first)]
        before, after = [first[i] for i in again["id"]], list(again["voice"])
        outcome[block] = agreement(before, after)
        if block == "retest_gemini":
            scored = [(gemini[i], first[i], blind[i]) for i in again["id"] if i in gemini]
            outcome[block]["distance_to_gemini"] = {"pairs": len(scored), "first": round(float(np.mean([abs(g - x) for g, x, _ in scored])), 3),
                                                    "blind": round(float(np.mean([abs(g - y) for g, _, y in scored])), 3)}
        print(block, outcome[block], flush=True)
    save({"trait_confirmation": outcome})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["frozen", "frozen-head", "lora", "small-dev", "confirm", "argon"])
    parser.add_argument("run_dir", nargs="*")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--delta-weight", type=float, default=1.0)
    parser.add_argument("--readout", choices=["mean", "attention"], default="mean")
    parser.add_argument("--layers", default="", help="hidden-state indices to mix; empty is the last layer alone")
    parser.add_argument("--scorer-dim", type=int, default=0, help="a tanh layer of this size under the attention scores")
    parser.add_argument("--score-rated", action="store_true", help="also score the 95 rated pairs at the end")
    args = parser.parse_args()
    if args.mode == "frozen":
        run_frozen()
    elif args.mode == "small-dev":
        run_small_dev(args.run_dir[0], args.score_rated)
    elif args.mode == "confirm":
        run_confirm(args.run_dir)
    elif args.mode == "argon":
        run_argon(args.run_dir)
    else:
        layers = tuple(int(x) for x in args.layers.split(",") if x)
        run_lora(args.steps, args.delta_weight, kind=args.readout, layers=layers, score_rated=args.score_rated,
                 frozen=args.mode == "frozen-head", scorer_dim=args.scorer_dim)


if __name__ == "__main__":
    main()
