# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two parallel dataset-building pipelines for the multilingual StyleDistance project:

1. **SynthSTEL translation** — translate English contrast pairs (positive/negative per style feature) into zh/ja/fr/ru via LiteLLM, producing a HuggingFace Datasets subset layout consumed by `../../../paper_replication/` for training.

2. **Multilingual raw text pipeline** — build a quality-filtered, deduplicated sentence corpus across ~65 languages from CulturaY (+ SkyPile-150B for zh), for use as content-distractor material in STEL-or-Content evaluation and future dataset expansion.

Both pipelines live in this directory, but they are split by responsibility: `data/translated/` for SynthSTEL outputs, `corpora/` for raw corpora, `pipeline/datadelta/` for the multilingual pipeline, and `analysis/` for derived analysis. The parent project (`../../../paper_replication/`) reads translated parquets from `data/translated/`.

## Setup

```bash
# From the stylevec repo root (not this directory):
uv sync

# LiteLLM proxy must be running for translation (port 3000)
# Node.js required for segmentation and MinHash dedup steps
```

Python scripts use `uv run python` (never bare `python3`). Node scripts use `node`.

## SynthSTEL Translation

Translates a contrast dataset into multiple languages, following the HuggingFace Datasets subset layout (one subdirectory per language, same parquet filename).

```bash
# Single file
uv run python translate.py --input data/train-00000-of-00001.parquet --output-dir data/translated/

# Whole splits directory
uv run python translate.py --splits-dir data/ --output-dir data/translated/
```

**Input schema**: parquet with columns `positive`, `negative`, `feature`, `feature_clean`.
**Output layout**: `data/translated/{en,zh,ja,fr,ru}/train-00000-of-00001.parquet`

**Resume**: A `.ckpt.jsonl` sidecar per language file tracks completed feature groups. Safe to interrupt and restart — completed features are skipped. Per-feature `action: "skip"` config in `feature_lang.json` (e.g., case-based features for zh/ja).

**LiteLLM config**: API base at `http://127.0.0.1:3000/v1`, model from `LITELLM_MODEL` env (default `openai/gemini-3-flash`), batch size from `BATCH_SIZE` env (default 20), parallelism from `PARALLEL` env (default 4).

**`feature_lang.json`**: Per-feature, per-language overrides — `action: "skip"` for features not applicable in a language (e.g., case-based contrasts in CJK), or `action: "note"` with a `text` field for translation guidance (e.g., number substitution works differently in zh/ja).

## Multilingual Raw Text Pipeline

Defined in `implement.md`. Core pipeline scripts and intermediate data live in `pipeline/datadelta/`; raw corpora live in `corpora/`, translated outputs live in `data/translated/`, and analysis artifacts live in `analysis/`. Run in numbered order:

| Step | Script | Input → Output |
|------|--------|----------------|
| 0 | `00_fetch.py` | CulturaY / SkyPile-150B → `raw/<lang>.jsonl` |
| 1 | `01_segment_filter.mjs` | `raw/` → `segmented/<lang>.jsonl` |
| 2 | (inline in step 1) | Token-length filter (15–512 tokens via Qwen3 tokenizer) |
| 3 | `02_heuristic_filter.py` | `segmented/` → `filtered/<lang>.jsonl` |
| 4 | `03_minhash_dedup.mjs` | `filtered/` → `deduped/<lang>.jsonl` |
| 5 | `04_score_ppl.py` | `deduped/` → `scored/<lang>.jsonl` |
| 6–8 | _(not yet implemented)_ | Semantic dedup (granite-embedding), bucket sampling, final parquet output |

**Data sources**: CulturaY (all langs except zh), SkyPile-150B (zh only). Quotas: 10k docs for major langs (en/ru/es/de/fr/ja), 500 for minor langs (all others).

**Step 0** (`00_fetch.py`): Uses HuggingFace `datasets` streaming mode. Random skip-sampling (5–30 row intervals). Resumes by checking if output file exists. `00_fetch_test.py` is a smaller variant (200 docs, 4 langs) for pipeline validation.

**Step 1** (`01_segment_filter.mjs`): Uses `Intl.Segmenter` (Node built-in) for sentence segmentation. Language code mapping: `hbs→sr`, `nb/nn→no`. Then pipes sentences through a Python helper (`_tok_helper.py`) that loads the local Qwen3-0.6B tokenizer to count tokens. Filters to 15–512 token range. Requires `../../../artifacts/base-models/qwen-3-0.6b` to exist.

**Step 2** (`02_heuristic_filter.py`): Gopher/C4-style rules with per-language terminal punctuation sets. Checks: URL/HTML/template residue, terminal punctuation (with CJK/Thai/Mongolian exemptions), digit/special-char ratio > 30%, repeated-word ratio > 30%.

**Step 3** (`03_minhash_dedup.mjs`): 5-gram character shingle MinHash (128 hashes, 16 bands × 8 rows), Jaccard threshold 0.75. Uses LSH bucketing for candidate lookup. Char-level shingles work across scripts without tokenization.

**Step 4** (`04_score_ppl.py`): Loads Qwen3-0.6B locally, computes per-sentence perplexity (teacher-forcing loss), appends `perplexity` field. Uses bf16, batch=16. Supports MPS/CUDA/CPU.

**Intermediate data**: Each step writes to a subdirectory of `pipeline/datadelta/` (`raw/`, `segmented/`, `filtered/`, `deduped/`, `scored/`). Per-language JSONL files with one JSON object per line. Schema evolves: `{text, lang}` → `{text, lang, tokens}` → `{text, lang, tokens}` → `{text, lang, tokens}` → `{text, lang, tokens, perplexity}`.

### Running individual pipeline steps

```bash
# All languages
uv run python pipeline/datadelta/00_fetch.py
node pipeline/datadelta/01_segment_filter.mjs
uv run python pipeline/datadelta/02_heuristic_filter.py
node pipeline/datadelta/03_minhash_dedup.mjs
uv run python pipeline/datadelta/04_score_ppl.py

# Single language
node pipeline/datadelta/01_segment_filter.mjs zh
uv run python pipeline/datadelta/02_heuristic_filter.py zh

# Quick pipeline test (4 langs, small sample)
uv run python pipeline/datadelta/00_fetch_test.py
```

### Final output (planned, not yet implemented)

Schema for `pipeline/datadelta/multilang.parquet`:
```
{text, lang, length_chars, perplexity, quality_score, bucket}
```

Bucket sampling targets: A (15–40 tok) 25%, B (40–120 tok) 40%, C (120–256 tok) 25%, D (256–512 tok) 10%.
Major languages target 20k sentences, minor languages 1k.

## Data Directory

- `data/train-00000-of-00001.parquet` / `test-00000-of-00001.parquet` — original English SynthSTEL splits
- `data/translated/{en,zh,ja,fr,ru}/` — translated output consumed by `../../../paper_replication/`
- `corpora/bookraw/` — raw epub/mobi literary corpus and extraction scripts
- `corpora/ao3_zh_kudos_crawl/` — AO3 Chinese fanfiction crawl (separate sub-project)
- `corpora/ao3_random_subset/` — AO3 author-split subset for style analysis
- `analysis/embedding_analysis/` — Genshin embedding projections and summary metrics

## Parent Project

This dataset directory feeds into `../../../paper_replication/` (StyleDistance multilingual replication using xlm-roberta-base + LoRA + triplet loss). The translated parquets are read by `paper_replication/triplets.py` via `--use-local-data`.

## Environment

- Package manager: `uv` (never bare pip/python3)
- Python scripts: `uv run python <script>`
- Node scripts: `node <script>.mjs` (ESM modules, `.mjs` extension)
- LiteLLM proxy must be running at `127.0.0.1:3000` for translation
- Local models at `../../../artifacts/base-models/qwen-3-0.6b` (for tokenization and perplexity scoring)
