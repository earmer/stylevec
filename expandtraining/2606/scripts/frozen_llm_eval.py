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
from frozen_llm import (conditional_scores, fisher_neurons, hidden_means, load_lm, neuron_means, neutralize,
                        persona_coordinates, reference_lines, summary_of, validation_rows)
from prebuilt import load_or_build
from report import heldout_speaker_set
from validate import write_validation_outputs

HEADLINE = ("heldout_top1_macro", "whitened_top1", "line_leak_heldout", "para_beats_own_heldout",
            "topic_gap_hub_neg", "topic_gap_other_neg", "content_rel_line")


def cached_neutrals(cache: Path, texts: list[str], gen_model: str, device: torch.device) -> dict[str, str]:
    path = cache / "neutral.parquet"
    known = dict(pd.read_parquet(path).itertuples(index=False)) if path.exists() else {}
    missing = [text for text in dict.fromkeys(texts) if text not in known]
    if missing:
        tokenizer, model = load_lm(gen_model, device)
        for start in range(0, len(missing), 1024):  # save as it goes: generation is the slow stage
            chunk = missing[start:start + 1024]
            known.update(zip(chunk, neutralize(chunk, tokenizer, model)))
            pd.DataFrame({"text": list(known), "neutral": list(known.values())}).to_parquet(path, index=False)
            print(f"neutral: {len(known)} texts", flush=True)
        del model
        torch.cuda.empty_cache()
    return known


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
    neutral = cached_neutrals(cache, texts + ref_texts + select_texts, args.gen_model, device)
    # hidden-state and conditional caches index this list; texts added later go after it, never into it
    unique = list(dict.fromkeys(texts + ref_texts))
    position = {text: i for i, text in enumerate(unique)}
    row_index = np.array([position[text] for text in texts])
    neutral_texts = [neutral[text] for text in unique]

    methods: dict[str, np.ndarray] = {}
    for path in args.enc_models:
        tag = model_tag(path)
        npz = cache / f"hidden_{tag}.npz"
        tokenizer, model = load_lm(path, device)
        depth = model.config.num_hidden_layers
        layers = sorted({max(1, round(f * depth)) for f in args.layer_fractions})
        if npz.exists():
            stored = np.load(npz)
            assert len(stored[f"t{layers[0]}"]) == len(unique), f"{npz} was built for other texts"
        else:
            plain = hidden_means(unique, tokenizer, model, layers)
            flat = hidden_means(neutral_texts, tokenizer, model, layers)
            stored = {f"{kind}{layer}": data[layer] for kind, data in (("t", plain), ("n", flat)) for layer in layers}
            np.savez(npz, **stored)
        del model
        torch.cuda.empty_cache()
        for layer in layers:
            plain, flat = stored[f"t{layer}"], stored[f"n{layer}"]
            diff = plain - flat
            reference = {name: diff[[position[line] for line in lines]] for name, lines in refs.items()}
            methods[f"raw_{tag}_l{layer}"] = plain[row_index]
            methods[f"diff_{tag}_l{layer}"] = diff[row_index]
            methods[f"persona_{tag}_l{layer}"] = persona_coordinates(diff, reference)[row_index]

    for path in args.cond_models:
        tag = model_tag(path)
        npy = cache / f"cond_{tag}_{len(refs)}x{args.ref_lines}.npy"
        if not npy.exists():
            tokenizer, model = load_lm(path, device)
            np.save(npy, conditional_scores(unique, neutral_texts, refs, tokenizer, model))
            del model
            torch.cuda.empty_cache()
        methods[f"cond_{tag}"] = np.load(npy)[row_index]

    full = list(dict.fromkeys(unique + select_texts))
    full_position = {text: i for i, text in enumerate(full)}
    for path in args.neuron_models:
        tag = model_tag(path)
        npz = cache / f"neurons_{tag}_{len(select)}x{args.select_lines}.npz"
        tokenizer, model = load_lm(path, device)
        depth = model.config.num_hidden_layers
        layers = sorted({max(1, round(f * depth)) for f in args.layer_fractions})
        if npz.exists():
            stored = np.load(npz)
        else:
            plain = neuron_means(full, tokenizer, model, layers)
            flat = neuron_means([neutral[text] for text in full], tokenizer, model, layers)
            stored = {f"d{layer}": (plain[layer].astype(np.float32) - flat[layer]).astype(np.float16)
                      for layer in layers}
            np.savez(npz, **stored)
        del model
        torch.cuda.empty_cache()
        rows_at = np.array([full_position[text] for text in texts])
        for layer in layers:
            diff = stored[f"d{layer}"].astype(np.float32)
            score, scale = fisher_neurons({name: diff[[full_position[line] for line in lines]]
                                           for name, lines in select.items()})
            ranked = np.argsort(-score)
            np.save(cache / f"neuron_rank_{tag}_l{layer}.npy", np.stack([ranked, score[ranked]]))
            methods[f"neuronall_{tag}_l{layer}"] = diff[rows_at] / scale
            for k in args.neuron_k:
                methods[f"neuron_{tag}_l{layer}_k{k}"] = diff[rows_at][:, ranked[:k]] / scale[ranked[:k]]

    table = []
    for name, embeddings in methods.items():
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
