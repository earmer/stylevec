# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Exact replication of the **StyleDistance** paper (Section 4.2, Appendix D), extended to **multilingual** (5 languages: en/zh/ja/fr/ru): xlm-roberta-base + LoRA (all-linear, r=8, α=8) + triplet margin loss. The model maps sentences to L2-normalized 768-dim style embeddings that separate style from content across languages.

This is a sub-project inside the broader `stylevec/` research repo. The shared utilities in `../shared/` (silhouette, consistency metrics) are used during evaluation.

## Setup

From the repo root:
```bash
uv sync
```

The local xlm-roberta-base model is expected at `../artifacts/base-models/xlm-roberta-base`. Download it first if missing.

## Running

```bash
# Training (from paper_replication/ directory)
uv run python train.py --use-local-data --batch 64 --epochs 5
uv run python train.py --resume                           # resume from latest checkpoint
uv run python train.py --dryrun --use-local-data          # single-batch test

# Evaluation — all four metrics (silhouette, consistency, STEL, S-o-C)
uv run python test_metrics.py                              # interactive checkpoint picker
uv run python test_metrics.py --checkpoint ../artifacts/paper_replication/checkpoints/latest
uv run python test_metrics.py --reference both             # compare with official models

# Evaluation — paper Section 5.1 tasks only
uv run python evaluate.py --checkpoint ../artifacts/paper_replication/checkpoints/latest --use-local-data
uv run python evaluate.py --compare-reference --use-local-data

# Remote training on GPU cloud
bash remote_train.sh                 # rsync + launch training
bash remote_train.sh --synconly      # sync only, don't train

# TensorBoard
tensorboard --logdir ../artifacts/paper_replication/checkpoints/tf-logs
```

Training outputs go to `../artifacts/paper_replication/checkpoints/` (PEFT adapters + optimizer state). Checkpoints are managed with top-5 rotation by validation loss.

## Architecture

```
config.py          — paper hyperparameters (Section 4.2 + Appendix D)
model.py           — StyleDistance: frozen xlm-roberta-base + LoRA + mean pooling + L2 norm
triplets.py        — MultilingualTripletDataset: 3 balanced triplet types (traditional, cross-lingual, language-as-feature)
train.py           — Training loop with gradient accumulation, AMP, checkpoint rotation, TensorBoard
evaluate.py        — STEL and STEL-or-Content tasks (Section 5.1)
test_metrics.py    — Combined eval: silhouette + consistency + STEL + S-o-C in one run
remote_train.sh    — rsync code to GPU cloud, start training via SSH
```

**Data**: Uses the translated SynthSTEL dataset (positive/negative pairs per style feature, 5 languages). Local parquet files are in `../data/datasets/msynthstel/data/translated/{en,zh,ja,fr,ru}/`. Pass `--use-local-data` to use them; omit to download from HuggingFace (English-only).

**Triplet types** (balanced): Type A (traditional: same-language negative), Type B (cross-lingual: different-language negative), Type C (language-as-feature: language identity as a style dimension).

**Model flow**: `StyleDistance.forward()` takes (anchor, positive, negative) triplets as tokenized IDs → returns three L2-normalized embeddings. `TripletMarginLoss(p=2, margin=0.1)` enforces `||a - p||² < ||a - n||²`.

**test_metrics.py** depends on both `evaluate.py` (same dir) and `../shared/evaluate.py` (silhouette/consistency).

## Key Metrics

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Cosine Silhouette Score | > 0 | Cluster separation by style feature |
| Intra-class Consistency | > 0 | Mean pairwise cosine within each feature |
| STEL Accuracy | near 1.0 | Pairing test sentences to correct anchors |
| STEL-or-Content | near 1.0 | Style matching when content is a distractor |
| Triplet Val Loss | decreasing | Margin loss on validation set |
