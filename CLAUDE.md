# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Never mention Claude, Sonnet, Opus, Haiku, Anthropic, "Co-authored-by", "assisted by", or similar AI attribution in any output including: commit messages, README, code files (except actual API calls), PR descriptions, comments, or documentation.

## Project Overview

`stylevec` is a research repository for style embeddings and style separation experiments on short text. Python drives most data preparation, modeling, and evaluation. `tools/simlar/` is a small experimental Rust/PyO3 aid, not a separate major surface. This repo is mainly Python.

The tree includes active research code, reproducibility artifacts, generated outputs, and large local assets. Treat it like a lab notebook plus toolchain, not a polished product repository.

## Core Rules

- High priority: keep experiment folders self-contained. Preserve each folder's local assumptions, entrypoints, paths, and outputs instead of forcing a unified repo-wide architecture.
- High priority: change scripts function-as-used. Keep single-purpose runners and research utilities shaped around their actual workflow; do not generalize them into full frameworks or shared orchestration unless the User explicitly asks.
- High priority: hardcoded paths, constants, and one-off choices are acceptable in experiment code when they match the active workflow. When cleanup or relocation changes one of them, update every affected reference or rewrite the local script so it remains coherent.
- Prefer the existing repository style and local patterns over introducing a new framework.
- Keep changes narrow and aligned with the directory that owns the behavior.
- Prefer `uv run` for project commands and `uv add` for project dependencies when practical.
- Preserve data pipeline resumability, checkpointing, and idempotency whenever touching scripts that process corpora or model outputs.
- Be careful with large binary artifacts, generated tables, model weights, and dataset files.
- Avoid destructive cleanup. Never remove user work, checkpoints, or data unless the User explicitly asks.
- When in doubt, inspect the nearest README, script header, or existing output directory before editing.

## Work Steps

当在项目根目录下工作时，不应当在根目录下创建临时文件、分析文件、文档文件，除非有意要求这么做。

应当的工作方式是，创建一个根据工作主题相关的目录，或者使用已有的符合工作主题的目录，然后把分析文件放在其中。

减少创建文档文件、说明文件、实施文件等 Markdown 文件。

## Repository Map

- `genshin/`: dialogue import, cleaning, paraphrasing, masking, and analysis for the Genshin corpus.
- `naive/`: baseline experiments such as residual vectors, LDA, prompt residuals, and MLP/ArcFace-style methods.
- `hidden/`: hidden-state probing experiments and data preparation for causal LM analysis.
- `lora/`: LoRA fine-tuning experiments, plots, evaluation code, and training utilities.
- `paper_replication/`: StyleDistance replication code, evaluation scripts, configuration, logs, and paper sources.
- `shared/`: reusable configuration, device, dataset, classifier, and evaluation utilities.
- `tools/simlar/`: Rust/PyO3 batch similarity library.
- `docs/verifier/`: algorithm notes and verification material.
- `data/`: local datasets and corpora that are not meant to be treated like source code.
- `artifacts/`, `artifacts/base-models/`, `artifacts/paper_replication/checkpoints/`, and similar folders: large local assets and derived artifacts.

## Setup

```bash
uv sync
uv run python download_base_models.py
uv run python download_base_models.py --modelscope  # use ModelScope mirror (China)
```

Build the Rust helper only when needed:

```bash
pip install maturin
cd tools/simlar
maturin develop
```

## Running Experiments

```bash
# LoRA fine-tuning (active research)
cd lora/
uv run python train.py --rank 8 --alpha 16 --batch 32 --workers 4

# Hidden layer probes
cd hidden/
uv run python run.py

# Data pipeline (run in order)
cd genshin/
uv run python import_genshin_data.py
uv run python clean_genshin_data.py
uv run python paraphrase_dialogues.py
```

There are no automated tests or linting configured.

## Architecture

### Experiment Modules

- `genshin/` - dialogue import, cleaning, paraphrasing, masking, and similarity scoring.
- `naive/` - baseline approaches such as residual style vectors, prompt residuals, LDA, and MLP/ArcFace variants.
- `hidden/` - causal LM hidden-state probes on Qwen3-0.6B.
- `lora/` - current focus. `model.py` defines the LoRA-backed style model; `train.py` runs training and evaluation; `data.py` handles cached loading.
- `paper_replication/` - StyleDistance replication and evaluation pipeline.
- `shared/` - shared config, data loading, classifiers, and evaluation helpers.
- `tools/simlar/` - small Rust/PyO3 helper for batched similarity scoring.

### Models

- **Qwen3-0.6B** (CausalLM, local) - 28 transformer layers, hidden_size=1024
- **Qwen3-embedding-0.6b / 8b** - via Ollama (naive experiments)
- Local weights stored in `artifacts/base-models/` (not git-tracked)
- Pre-computed embeddings cached in `artifacts/cache/*.npz` (not git-tracked)

### Key Metrics

- **Cosine Silhouette Score** - primary metric for cluster separation by speaker
- **Intra-class Consistency** - average cosine similarity within a speaker
- **ArcFace Accuracy** - speaker classification accuracy

## Validation

- If the change is local, run the relevant script or a focused smoke test.
- If the change affects shared utilities, run the smallest realistic end-to-end path that exercises them.
- For Rust changes in `tools/simlar/`, run `cargo test` when practical and verify `maturin develop` still works if the Python boundary changed.

### lora/train.py CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rank` | `LORA_R` | LoRA rank |
| `--alpha` | `LORA_ALPHA` | LoRA alpha |
| `--batch` | auto | Batch size |
| `--grad` | - | Gradient accumulation steps |
| `--workers` | 4 | DataLoader workers |
| `--no-cache` | - | Disable preprocessed data cache |
