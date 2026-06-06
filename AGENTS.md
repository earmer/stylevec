# AGENTS.md

This file is the working guide for future Codex sessions in `stylevec`.

## Project Snapshot

`stylevec` is a research repository for style embeddings and style separation experiments on short text. The codebase is intentionally mixed and experimental: Python drives most data preparation, modeling, and evaluation, while `tools/simlar/` provides a small Rust/PyO3 helper for similarity scoring.

The repository includes active research code, reproducibility artifacts, generated outputs, and large local assets. Treat the tree as a lab notebook plus toolchain, not as a polished product repository.

## Core Principles

- High priority: keep experiment folders self-contained. Preserve each folder's local assumptions, entrypoints, paths, and outputs instead of forcing a unified repo-wide architecture.
- High priority: change scripts function-as-used. Keep single-purpose runners and research utilities shaped around their actual workflow; do not generalize them into full frameworks or shared orchestration unless the User explicitly asks.
- High priority: hardcoded paths, constants, and one-off choices are acceptable in experiment code when they match the active workflow. When cleanup or relocation changes one of them, update every affected reference or rewrite the local script so it remains coherent.
- Prefer the existing repository style and local patterns over introducing a new framework.
- Keep changes narrow and aligned with the directory that owns the behavior.
- Preserve data pipeline resumability, checkpointing, and idempotency whenever touching scripts that process corpora or model outputs.
- Be careful with large binary artifacts, generated tables, model weights, and dataset files.
- Avoid destructive cleanup. Never remove user work, checkpoints, or data unless the User explicitly asks.
- When in doubt, inspect the nearest README, script header, or existing output directory before editing.

## Repository Map

- `genshin/`: dialogue import, cleaning, paraphrasing, masking, and analysis for the Genshin corpus.
- `naive/`: baseline experiments such as residual vectors, LDA, prompt residuals, and MLP/ArcFace-style methods.
- `hidden/`: hidden-state probing experiments and data preparation for causal LM analysis.
- `lora/`: LoRA fine-tuning experiments, plots, evaluation code, and training utilities.
- `paper_replication/`: StyleDistance replication code, evaluation scripts, configuration, logs, and paper sources.
- `shared/`: reusable configuration, device, dataset, classifier, and evaluation utilities.
- `tools/simlar/`: Rust/PyO3 batch similarity library.
- `verifier/`: algorithm notes and verification material.
- `data/`: local datasets and corpora that are not meant to be treated like source code.
- `artifacts/`, `artifacts/base-models/`, `artifacts/paper_replication/checkpoints/`, and similar folders: large local assets and derived artifacts.

## Typical Working Areas

### Python

Most Python code lives under `genshin/`, `hidden/`, `lora/`, `naive/`, `paper_replication/`, and `shared/`.

Common expectations:

- Python version: `>= 3.11`
- Dependency management: `uv sync`
- Prefer small, script-level validation over broad refactors when changing experiment code
- Keep function and file names descriptive, literal, and research-friendly

NOTE: Should use `uv run...` `uv add ...` instead of raw `pip ...` `python ...`.

### Rust extension

`tools/simlar/` is a small experimental Rust/PyO3 aid. This repo is mainly Python.

Typical workflow:

```bash
cd tools/simlar
maturin develop
```

Keep Rust changes small, keep the Python-facing API stable, and run `cargo test` when practical.

## Setup Expectations

Use the project README as the first source of truth for setup, but the usual flow is:

```bash
uv sync
uv run python download_base_models.py
```

If the Rust helper is needed:

```bash
pip install maturin
cd tools/simlar
maturin develop
```

Avoid adding new setup steps unless the change genuinely requires them.

## Data And Artifact Rules

This repository handles research data that may be copyrighted, sensitive, or simply too large to casually duplicate.

- Do not paste long raw corpus excerpts into logs, commits, or responses.
- Do not assume generated outputs are disposable if they live in tracked research directories.
- Keep large model weights and caches out of source-control changes unless the User explicitly wants them tracked.
- When adding new artifacts, follow the existing naming style, especially timestamped result folders.
- If a script writes outputs, make the output location explicit and easy to find.

## Editing Guidelines

- Prefer ASCII unless the surrounding file already uses another character set.
- Preserve the repository's plain, practical style.
- Keep docstrings and comments short and useful.
- Add comments only where the code is not obvious from context.
- Avoid over-abstracting one-off research code.
- Reuse existing helpers from `shared/` before introducing duplicates.

## Validation Guide

Choose validation that matches the size of the change.

### For Python changes

- If the change is local, run the relevant script or a focused smoke test.
- If the change affects shared utilities, run the smallest realistic end-to-end path that exercises them.
- If a script has a `--help` mode or a tiny sample path, use that before launching a large job.

### For Rust changes

- Run `cargo test` in `tools/simlar/` when the build or logic changes.
- If the Python extension boundary changed, verify that `maturin develop` still succeeds.

### For documentation-only changes

- Verify links, filenames, and command examples against the current tree.

## Research Workflow Notes

This project includes several experiment families:

- Genshin dialogue paraphrase and masking pipelines
- hidden-layer probing and style-vector experiments
- LoRA fine-tuning experiments
- StyleDistance replication work
- paper figures and LaTeX sources

When editing any of these, keep the surrounding experimental narrative intact. If results are being updated, update the nearby documentation or table source at the same time so the repository does not drift into mismatched claims.

## Large-Scale Safety Notes

- Be cautious with scripts that download models, crawl data, or call external APIs.
- Check for hard-coded credentials before adding new API integrations.
- Avoid changing evaluation code in a way that silently invalidates past metrics.
- If a change would affect reproducibility, mention the consequence clearly in the handoff.

## Suggested Working Habit

1. Read the nearest README or module docstring.
2. Inspect the current files that own the behavior.
3. Make the smallest coherent change.
4. Validate the exact path touched.
5. Summarize any remaining risk or incomplete coverage.

## Notes For Future Agents

- This repository is research-heavy, so imperfect intermediate states are normal.
- It is fine to leave experiments in place if they document the path taken.
- It is better to preserve a useful artifact than to polish it away prematurely.
- If a directory already contains generated results, treat them as part of the project history unless the User says otherwise.
