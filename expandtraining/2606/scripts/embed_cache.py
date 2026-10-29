"""Embed a fit sample of the training split with a frozen checkpoint, for heads fitted after training.

Writes outputs/runs/<run>/fitcache/{rows.parquet, embeddings.npy (float16)}: the same row metadata as
validation, so a head fitted here is judged on the run's validation embeddings (speakers in the training
split are never the held-out ones). Rules and counts are fixed so every encoder gets the same sample.

uv run python expandtraining/2606/scripts/embed_cache.py --run-name <run> --dataset d-402f1c32b449
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import numpy as np
from torch.utils.data import DataLoader

from config import load_config, workspace_dir
from dataset import TripletTextDataset, make_triplet_collate
from model import build_qwen_style_encoder
from prebuilt import load_or_build
from validate import collect_validation_embeddings, load_checkpoint_for_validation, select_device

# origin_vs_para gives line/paraphrase pairs (content), same_speaker_real speaker-labelled lines, topic_matched
# same-topic lines of different speakers, the trait rules the named axes.
FIT_RULES = {"origin_vs_para": 6000, "same_speaker_real": 6000, "same_speaker_topic_matched": 4000,
             "trait_positive_vs_negative": 1024, "trait_cross_feature_hard": 1024}


def pick(triplets: list, counts: dict[str, int]) -> list:
    chosen = []
    for rule, count in counts.items():
        rows = sorted((t for t in triplets if t.split == "train" and t.rule == rule), key=lambda t: t.triplet_id)
        if len(rows) > count:
            rows = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, count).astype(int))]
        chosen.extend(rows)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default="latest", help="checkpoints/<name> under the run")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    run_dir = workspace_dir() / "outputs" / "runs" / args.run_name
    config = load_config(str(run_dir / "config.yaml"))
    triplets, _manifest = load_or_build(config, args.dataset, True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(TripletTextDataset(pick(triplets, FIT_RULES), "train"),
                        batch_size=int(config["validation"]["batch_size"]), shuffle=False, num_workers=0,
                        collate_fn=make_triplet_collate(tokenizer, int(config["model"]["max_length"])))
    device = select_device(config)
    model = build_qwen_style_encoder(config).to(device)
    load_checkpoint_for_validation(model, run_dir / "checkpoints" / args.checkpoint)
    rows, embeddings, _summary = collect_validation_embeddings(model, loader, device)
    out_dir = run_dir / "fitcache"
    out_dir.mkdir(exist_ok=True)
    rows.to_parquet(out_dir / "rows.parquet", index=False)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float16))
    print(f"fitcache: {len(rows)} rows -> {out_dir}")


if __name__ == "__main__":
    main()
