"""Score training-free style vectors from frozen LMs on a trained run's validation rows (see src/frozen_llm.py).

Each method lands in outputs/runs/frozen_<tag>__<method>/validation, read by the same report and ablation code as a
trained run. Rewrites, hidden states and conditional scores are cached under outputs/frozen/<tag>/, so a rerun
only computes what is missing.

uv run python expandtraining/2606/scripts/frozen_llm_eval.py --tag a \
  --config expandtraining/2606/configs/ablations/generated/scene_a_1gpu/eval.yaml --dataset d-e7dd4c973f56 \
  --gen-model artifacts/base-models/qwen-3-4b-instruct --enc-models artifacts/base-models/qwen-3-0.6b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch

from ablation import run_metrics
from config import load_config, workspace_dir
from frozen_llm import (conditional_scores, content_residual, fisher_neurons, hidden_means, language_kept, load_lm, neuron_means,
                        neutralize, persona_coordinates, reference_lines, summary_of, validation_rows)
from prebuilt import load_or_build
from report import heldout_speaker_set
from validate import write_validation_outputs

HEADLINE = ("heldout_top1_macro", "whitened_top1", "line_leak_heldout", "para_beats_own_heldout",
            "topic_gap_hub_neg", "topic_gap_other_neg", "content_rel_line")


def cached_neutrals(cache: Path, texts: list[str], gen_model: str,
                    device: torch.device) -> tuple[dict[str, str], dict[str, str]]:
    """(rewrites now, rewrites as cached before this call). Rewrites that switched language are redone."""
    path = cache / "neutral.parquet"
    known = dict(pd.read_parquet(path).itertuples(index=False)) if path.exists() else {}
    before = dict(known)
    missing = [text for text in dict.fromkeys(texts) if text not in known or not language_kept(text, known[text])]
    if missing:
        tokenizer, model = load_lm(gen_model, device)
        for start in range(0, len(missing), 1024):  # save as it goes: generation is the slow stage
            chunk = missing[start:start + 1024]
            known.update(zip(chunk, neutralize(chunk, tokenizer, model)))
            pd.DataFrame({"text": list(known), "neutral": list(known.values())}).to_parquet(path, index=False)
            print(f"neutral: {len(known)} texts", flush=True)
        del model
        torch.cuda.empty_cache()
        flipped = sum(not language_kept(text, known[text]) for text in missing)
        print(f"neutral: {len(missing)} rewritten, {flipped} still switched language", flush=True)
    return known, before


def text_hash(texts: list[str]) -> np.ndarray:
    return np.array([int(hashlib.blake2b(t.encode(), digest_size=8).hexdigest(), 16) >> 1 for t in texts],
                    dtype=np.int64)


def stale_rows(stored: dict, fallback: np.ndarray, now: np.ndarray) -> np.ndarray:
    """Rows whose neutral rewrite changed since the cache was built (caches without a hash predate the repair)."""
    return np.flatnonzero(stored.pop("nhash", fallback) != now)


def down_proj(path: str, layer: int) -> np.ndarray:
    """Decoder layer `layer` - 1's MLP down_proj weight (hidden_states[layer] is that layer's output), read
    straight from the safetensors shards."""
    from safetensors import safe_open
    name = f"model.layers.{layer - 1}.mlp.down_proj.weight"
    for shard in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).float().numpy()
    raise KeyError(name)


def model_tag(path: str) -> str:
    return Path(path).name.replace("qwen-3-", "q").replace("-instruct", "i").replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gen-model", required=True, help="writes the neutral rewrites")
    parser.add_argument("--enc-models", nargs="*", default=[], help="hidden states (raw, diff, persona)")
    parser.add_argument("--cond-models", nargs="*", default=[], help="conditional likelihood scores")
    parser.add_argument("--layer-fractions", nargs="+", type=float, default=[0.5, 0.75, 1.0])
    parser.add_argument("--ref-speakers", type=int, default=32)
    parser.add_argument("--ref-lines", type=int, default=12)
    parser.add_argument("--neuron-models", nargs="*", default=[], help="MLP neuron shifts, top-k by Fisher ratio")
    parser.add_argument("--neuron-k", nargs="+", type=int, default=[256, 1024])
    parser.add_argument("--select-lines", type=int, default=48, help="lines per reference speaker for ranking neurons")
    parser.add_argument("--only", nargs="*", default=[], help="write only methods with these prefixes")
    parser.add_argument("--per-rule", type=int, default=0, help="0: the config's validation.per_rule")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    triplets, _manifest = load_or_build(config, args.dataset, True)
    rows, texts = validation_rows(triplets, args.per_rule or int(config["validation"].get("per_rule", 64)))
    refs = reference_lines(triplets, args.ref_speakers, args.ref_lines)
    assert not set(refs) & heldout_speaker_set(rows), "a reference speaker is held out"
    ref_texts = [line for lines in refs.values() for line in lines]
    cache = workspace_dir() / "outputs" / "frozen" / args.tag
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "references.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows, {len(set(texts))} texts, {len(refs)} reference speakers", flush=True)

    select = reference_lines(triplets, args.ref_speakers, args.select_lines) if args.neuron_models else {}
    select_texts = [line for lines in select.values() for line in lines]
    neutral, neutral_before = cached_neutrals(cache, texts + ref_texts + select_texts, args.gen_model, device)
    # hidden-state and conditional caches index this list; texts added later go after it, never into it
    unique = list(dict.fromkeys(texts + ref_texts))
    position = {text: i for i, text in enumerate(unique)}
    row_index = np.array([position[text] for text in texts])
    neutral_texts = [neutral[text] for text in unique]
    now_hash = text_hash(neutral_texts)
    old_hash = text_hash([neutral_before.get(text, "") for text in unique])

    held = heldout_speaker_set(rows)
    fit = np.unique(np.concatenate([row_index[~rows["speaker"].isin(held).to_numpy()],
                                    [position[text] for text in ref_texts]]))
    methods: dict[str, np.ndarray] = {}
    hidden_diff: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in args.enc_models:
        tag = model_tag(path)
        npz = cache / f"hidden_{tag}.npz"
        tokenizer, model = load_lm(path, device)
        depth = model.config.num_hidden_layers
        layers = sorted({max(1, round(f * depth)) for f in args.layer_fractions})
        if npz.exists():
            stored = dict(np.load(npz))
            assert len(stored[f"t{layers[0]}"]) == len(unique), f"{npz} was built for other texts"
            stale = stale_rows(stored, old_hash, now_hash)
            if stale.size:
                flat = hidden_means([neutral_texts[i] for i in stale], tokenizer, model, layers)
                for layer in layers:
                    stored[f"n{layer}"][stale] = flat[layer]
                np.savez(npz, nhash=now_hash, **stored)
                print(f"hidden {tag}: {stale.size} rewrites refreshed", flush=True)
        else:
            plain = hidden_means(unique, tokenizer, model, layers)
            flat = hidden_means(neutral_texts, tokenizer, model, layers)
            stored = {f"{kind}{layer}": data[layer] for kind, data in (("t", plain), ("n", flat)) for layer in layers}
            np.savez(npz, nhash=now_hash, **stored)
        del model
        torch.cuda.empty_cache()
        for layer in layers:
            plain, flat = stored[f"t{layer}"], stored[f"n{layer}"]
            diff = plain - flat
            reference = {name: diff[[position[line] for line in lines]] for name, lines in refs.items()}
            methods[f"raw_{tag}_l{layer}"] = plain[row_index]
            methods[f"diff_{tag}_l{layer}"] = diff[row_index]
            methods[f"persona_{tag}_l{layer}"] = persona_coordinates(diff, reference)[row_index]
            methods[f"half_{tag}_l{layer}"] = (plain - 0.5 * flat)[row_index]
            methods[f"resid_{tag}_l{layer}"] = content_residual(plain, flat, fit)[row_index]
            hidden_diff[(tag, layer)] = (diff[row_index], methods[f"resid_{tag}_l{layer}"])

    for path in args.cond_models:
        tag = model_tag(path)
        base = cache / f"cond_{tag}_{len(refs)}x{args.ref_lines}"
        if base.with_suffix(".npz").exists():
            stored = dict(np.load(base.with_suffix(".npz")))
        elif base.with_suffix(".npy").exists():
            stored = {"scores": np.load(base.with_suffix(".npy"))}
        else:
            stored = {"scores": None}
        stale = (np.arange(len(unique)) if stored["scores"] is None
                 else stale_rows(stored, old_hash, now_hash))
        if stale.size:
            tokenizer, model = load_lm(path, device)
            fresh = conditional_scores([unique[i] for i in stale], [neutral_texts[i] for i in stale], refs,
                                       tokenizer, model)
            scores = fresh if stored["scores"] is None else stored["scores"]
            scores[stale] = fresh
            np.savez(base.with_suffix(".npz"), scores=scores, nhash=now_hash)
            del model
            torch.cuda.empty_cache()
        methods[f"cond_{tag}"] = np.load(base.with_suffix(".npz"))["scores"][row_index]

    full = list(dict.fromkeys(unique + select_texts))
    speaker_of_text = {**dict(zip(texts, rows["speaker"])), **{t: name for name, lines in select.items() for t in lines}}
    full_position = {text: i for i, text in enumerate(full)}
    for path in args.neuron_models:
        tag = model_tag(path)
        top = max(args.neuron_k)
        npz = cache / f"neurons_{tag}_{len(select)}x{args.select_lines}_top{top}.npz"
        stored = dict(np.load(npz)) if npz.exists() else {}
        if not any(key.startswith("p") for key in stored):  # all neurons of every text would fill the disk:
            tokenizer, model = load_lm(path, device)        # keep the top-ranked columns, line and rewrite apart
            depth = model.config.num_hidden_layers
            layers = sorted({max(1, round(f * depth)) for f in args.layer_fractions})
            plain = neuron_means(full, tokenizer, model, layers)
            flat = neuron_means([neutral[text] for text in full], tokenizer, model, layers)
            del model
            torch.cuda.empty_cache()
            stored = {}
            for layer in layers:
                line, rewrite = plain.pop(layer).astype(np.float32), flat.pop(layer).astype(np.float32)
                score, scale = fisher_neurons({name: (line - rewrite)[[full_position[t] for t in lines]]
                                               for name, lines in select.items()})
                ranked = np.argsort(-score)
                np.save(cache / f"neuron_rank_{tag}_l{layer}.npy", np.stack([ranked, score[ranked]]))
                keep = ranked[:top]
                stored[f"p{layer}"] = (line[:, keep] / scale[keep]).astype(np.float16)
                stored[f"f{layer}"] = (rewrite[:, keep] / scale[keep]).astype(np.float16)
            np.savez(npz, **stored)
        rows_at = np.array([full_position[text] for text in texts])
        fit_full = np.concatenate([fit, [full_position[t] for t in select_texts]])
        for key in [key for key in stored if key.startswith("p")]:
            layer = key[1:]
            line, rewrite = stored[f"p{layer}"].astype(np.float32), stored[f"f{layer}"].astype(np.float32)
            for k in args.neuron_k:
                methods[f"neuron_{tag}_l{layer}_k{k}"] = (line - rewrite)[rows_at, :k]
                rneuron = content_residual(line[:, :k], rewrite[:, :k], fit_full)[rows_at]
                methods[f"rneuron_{tag}_l{layer}_k{k}"] = rneuron
                if f"resid_{tag}_l{layer}" in methods:  # resid of the hidden state beside the neuron residual
                    hidden = methods[f"resid_{tag}_l{layer}"]
                    methods[f"rboth_{tag}_l{layer}_k{k}"] = np.concatenate(
                        [hidden / hidden.std(), rneuron / rneuron.std()], axis=1)
        for key in [key for key in stored if key.startswith("p")]:
            layer = int(key[1:])
            if (tag, layer) not in hidden_diff:
                continue
            ranked = np.load(cache / f"neuron_rank_{tag}_l{layer}.npy")[0].astype(int)
            out = down_proj(path, layer)  # [d_model, neurons]: each neuron's write direction
            diff, resid = hidden_diff[(tag, layer)]
            for k in args.neuron_k:
                basis = np.linalg.pinv(out[:, ranked[:k]])  # coordinates in the top neurons' output span
                methods[f"wout_{tag}_l{layer}_k{k}"] = diff @ basis.T
                methods[f"wresid_{tag}_l{layer}_k{k}"] = resid @ basis.T
        pd.DataFrame({"text": full, "neutral": [neutral[t] for t in full],
                      "speaker": [speaker_of_text.get(t, "") for t in full]}).to_parquet(cache / "full.parquet")

    table = []
    for name, embeddings in methods.items():
        if args.only and not name.startswith(tuple(args.only)):
            continue
        run_dir = workspace_dir() / "outputs" / "runs" / f"frozen_{args.tag}__{name}__s0"
        out_dir = run_dir / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        embeddings = embeddings.astype(np.float32)
        write_validation_outputs(out_dir, rows, embeddings, summary_of(rows, embeddings), config,
                                 f"frozen LM: {name}", final=False)
        metrics = run_metrics(run_dir, []) or {}
        table.append({"method": name, **{key: metrics.get(key) for key in HEADLINE}})
        print(pd.DataFrame(table[-1:]).to_string(index=False, header=len(table) == 1, float_format="%.3f"), flush=True)
    summary_path = cache / "summary.csv"
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=["method"])
    new = pd.DataFrame(table)
    pd.concat([old[~old["method"].isin(new["method"])], new]).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
