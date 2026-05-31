# stylevec

> **Note 1:** This README reflects the canonical state of the project. READMEs on other branches are not maintained and should be disregarded — refer to this one regardless of which branch you are on.
> **Note 2:** This is the development repository. The files here are unorganized and not currently in a usable state.

Research project exploring **style** (not content) embeddings for short texts — can a speaker's stylistic fingerprint be extracted as a separable vector, independent of topic?

Disclaimer: Datasets used in this repository may be subject to copyrights held by other individuals or organizations. This project is intended solely for academic research purposes.


## Setup

```bash
uv sync
python download_base_models.py          # downloads Qwen3-0.6B locally
python download_base_models.py --modelscope  # use ModelScope mirror (China)
```

Build the Rust similarity utility when needed:
```bash
pip install maturin
cd simlar/ && maturin develop
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `genshin/` | Data import, cleaning, paraphrase generation, content masking |
| `naive/` | Baseline approaches: residual vectors, prompt-residual, LDA, MLP+ArcFace |
| `hidden/` | Causal LM hidden layer probe experiments (Qwen3-0.6B) |
| `lora/` | LoRA fine-tuning approach (active research) |
| `paper_replication/` | StyleDistance (Patel et al., 2024) replication — roberta-base + LoRA + Triplet Loss |
| `shared/` | Shared training infrastructure: config, data loader, classifiers, evaluation |
| `simlar/` | Rust/PyO3 utility for batched char n-gram Jaccard + normalized Levenshtein similarity |
| `verifier/` | Algorithm notes and verification |
| `datasets/` | Datasets (not included in public available version) |
| `cache/` | Pre-computed `.npz` embeddings (not tracked) |
| `base-models/` | Local model weights (not tracked) |

## Data Pipeline

### Genshin Parallel Corpus

283,972 raw dialogue records from 4,332 speakers, cleaned to ~258,868 valid entries. Each utterance is rewritten by an LLM (via OpenRouter API) into semantically equivalent, stylistically neutral standard Chinese, producing 262,196 pairs. The pipeline supports checkpoint resumption, concurrent requests, exponential-backoff rate limiting, and automatic quality validation (speaker consistency, line count parity, n-gram + Levenshtein similarity scoring).

### Content Masking System

To isolate style from content independently of the paraphrase approach, 5,021 Chinese words from the Genshin game lexicon (6,050 terms, 82.9% coverage) are mapped to 12 semantic category masks (person, location, quest, enemy, item, food, loot, weapon, artifact, animal, domain, group). This substitutes domain-specific content words while preserving syntactic structure and stylistic markers.

### Multilingual Text Pipeline (datadelta)

A 75-language text quality pipeline, separate from this repo: document collection (CulturaY/SkyPile, 20k docs for major languages, 1k for others) → sentence segmentation (Intl.Segmenter) → length filtering (15–512 tokens) → heuristic filtering (terminal punctuation, special-char ratio, repeated-word ratio, URL residuals) → MinHash near-dedup (5-gram character shingles, Jaccard threshold 0.75) → quality scoring (Qwen3-0.6B perplexity / LiteLLM API, pending). Currently ~2.3M deduplicated sentences across the top 6 languages (zh/en/ru/ja/fr/de); quality scoring not yet run.

### Literary Book Collection

Several Chinese and English literary works collected from Anna's Archive and Z-Library. Raw epub/mobi formats acquired; text extraction and character-dialogue annotation pending to done.

Copyright status varies depending on the author and your jurisdiction.

### AO3 Fanfiction Corpus

Archive of Our Own Chinese-language fanfiction crawled by kudos ranking across 34 fandoms (Genshin Impact, Harry Potter, Marvel, Jujutsu Kaisen, etc.), ~1,000 works per fandom. An additional HuggingFace AO3 random subset (64,000 works with rich metadata) provides a baseline corpus for general style/metadata experiments.

### SynthSTEL Multilingual Translation

The SynthSTEL benchmark (40 style features × 100 pairs, 3,600 train + 400 test) translated from English into Chinese, Japanese, French, and Russian via LiteLLM. Style features were adapted per language: features meaningless in the target language (e.g., capitalization variants for CJK) were skipped; language-specific features (Chinese homophone-digit substitution, Japanese contracted forms, Russian colloquial reductions) received localized rewrite instructions.

## Method Exploration

### Phase 1: Static Embedding Extraction (Dec 2025 – Jan 2026)

Attempted to extract style vectors from frozen embedding models (Qwen3-embedding-0.6b/8b, embeddinggemma).

**Residual method:** `style = embed(original) − embed(paraphrase)`

| Model | Mode | Avg Consistency | Silhouette |
|-------|------|----------------|------------|
| qwen3-embedding:0.6b | raw residual | 0.0183 | −0.0085 |
| qwen3-embedding:0.6b | PCA projected | 0.0043 | −0.0055 |
| embeddinggemma:latest | raw residual | 0.0089 | −0.0085 |
| embeddinggemma:latest | PCA projected | 0.0024 | −0.0067 |

**Instruction-aware prompting:** Tested 7 prompt variants by prepending instructions to input text (the Ollama/OpenRouter `prompt` field is silently ignored; instructions must be concatenated into the input string).

| Prompt | Model | Avg Consistency | Silhouette |
|--------|-------|----------------|------------|
| baseline | 0.6b | 0.0171 | −0.0087 |
| style_v2 ("analyze linguistic style") | 0.6b | 0.0341 | −0.0147 |
| style_v2 | 8b | 0.7675 | −0.0372 |
| style_v4 ("identify the speaker") | 8b | 0.6946 | −0.0427 |

8b + style_v2 achieved 0.77 intra-speaker consistency, but silhouette remained negative — different speakers' style directions are highly overlapping. The residual method destroyed signal (0.77 → 0.08).

**LDA supervised dimensionality reduction:**

| Scale | Dim | Train Sil | Test Sil |
|-------|-----|-----------|----------|
| 6→5 speakers | — | 0.35 | −0.03 |
| 80→20 speakers | 64 | 0.019 | −0.015 |
| 80→20 speakers | 32 | 0.003 | −0.032 |
| 80→20 speakers | 16 | −0.075 | −0.068 |

LDA learns speaker-specific discriminants, not a generalizable style space.

**MLP + ArcFace (141→36 speakers, stratified sampling):**

| Arch | Dim | Train Sil | Val Sil | Test Sil | Train Cons |
|------|-----|-----------|---------|----------|------------|
| linear | 32 | −0.198 | −0.277 | −0.146 | 0.981 |
| linear | 128 | −0.075 | −0.160 | −0.071 | 0.984 |
| 1h-512 | 128 | −0.110 | −0.246 | −0.131 | 0.995 |
| 2h-1k | 128 | −0.314 | −0.593 | −0.685 | 0.976 |
| 1h-2k | 64 | −0.225 | −0.332 | −0.190 | 0.994 |

Universal mode collapse: consistency ~0.98–0.99 across train/val/test, silhouette uniformly negative. ArcFace's margin is too aggressive at this data scale (141-class loss floor ~0, actual minimum ~8).

**Phase 1 conclusion:** Frozen embedding models have been thoroughly stripped of style information during contrastive pretraining. No post-hoc projection, dimensionality reduction, or supervised classification can recover a separable style signal. The model must be fine-tuned to actively learn style separability.

### Phase 2: Causal LM Hidden Layer Probes (Jan – Feb 2026)

Tested whether Qwen3-0.6B CausalLM intermediate hidden states (not contrastive-trained) retain extractable style information.

**Setup:** 29 layers (embedding + 28 transformer) × 4 pooling strategies (last token, attention-weighted, reverse-attention complement, reverse-attention inverse) × 5 classifiers (LDA + 4 MLP+ArcFace variants) = 580 independent evaluations. 8 training speakers, 4 held-out speakers, 1,200 sentences.

**Key findings:**

- **All 580 val_sil scores are negative.** Best result: reverse-attention inverse pooling at layer 1 + MLP-linear-d64, val_sil = −0.0705.
- **Extreme overfitting:** MLP+ArcFace reaches train_sil +0.87 while val_sil crashes to −0.10. The model memorizes sentence identity, not speaker style.
- **Attention pooling collapse:** Forward attention-weighted pooling locks consistency at 1.0000 from layer 3 onward — all vectors collapse to a single direction.
- **Layer differences are noise:** Layers 2–28 val_sil fluctuate within a 0.011 band with no systematic trend (shallow vs. deep vs. middle).

**Theoretical interpretation:** Style information exists in causal LMs (they perform stylistic continuation and style transfer), but it is encoded in generation dynamics — each token step draws on style cues through the full 28-layer computation graph to modulate the conditional distribution. Style is a function, not a point. Static vector extraction from individual layers is a paradigm mismatch.

**Phase 2 conclusion:** Single-layer hidden state snapshots cannot capture style. Style is encoded in the computation, not in the representation.

### Phase 3: LoRA Fine-tuning (Feb 2026 – present, active)

Forces style separability into the representation space by fine-tuning Qwen3-0.6B with LoRA adapters and ArcFace supervision.

**Architecture** (`lora/model.py`):
- Frozen Qwen3-0.6B encoder + LoRA adapters (rank=8, alpha=16, targeting q_proj/k_proj/v_proj/o_proj)
- Optional LayerFusion: learned weighted combination of selected intermediate hidden states
- Optional AttentionPooling: learned query vector replaces mean pooling
- style_head (linear projection, no bias) + L2 normalization → 128-dim style vector
- ArcFaceHead (s=30, m=0.3) for speaker classification
- PKSampler: P speakers × K utterances per batch

**5-epoch results** (batch=32, rank=8, alpha=16):

| Epoch | Train Loss | Train Sil | Val Sil | Train Acc | Val Acc |
|-------|-----------|-----------|---------|-----------|---------|
| 1 | 14.79 | −0.266 | −0.225 | 0.004 | 0.004 |
| 2 | 12.61 | −0.192 | −0.185 | 0.082 | 0.061 |
| 3 | 10.34 | −0.103 | −0.137 | 0.177 | 0.118 |
| 4 | 8.61 | −0.026 | −0.088 | 0.272 | 0.154 |
| 5 | 7.11 | **+0.037** | −0.053 | 0.370 | 0.188 |

Training silhouette turned positive for the first time across all experiments (previous phases: near-zero or train/val gap of 0.9+). Loss is decreasing steadily, accuracy climbing. Validation silhouette remains negative (−0.053), indicating generalization to unseen speakers is the current bottleneck. Post-training analysis of 354 speakers shows median separability of 0.027 with only 6 speakers above 0.1 — training is far from converged at 5 epochs.

**Phase 3 status:** The problem has shifted from "completely inseparable" to "learnable but not yet generalizing." Next steps: more epochs, higher rank, hyperparameter grid search (alpha, margin, lr), synthetic data augmentation, and multilingual joint training.

## StyleDistance Replication (May 2026)

Reproduced StyleDistance (Patel et al., 2024) to establish a quantitative baseline:

- **Model:** roberta-base + LoRA rank=8 (1.34M / 126M trainable params, 1.06%)
- **Data:** SynthSTEL (40 features × 100 pairs each)
- **Training:** batch=512, triplet margin=0.1, lr=1e-4, cosine schedule, remote AutoDL GPU
- **Result:** Training loss 0.017 → 0.0004 over 10 epochs; validation loss converged at epoch 2 (0.057), consistent with the paper's early-stopping design (patience=1). Pending STEL/STEL-or-Content evaluation on the remote instance.

## Engineering Infrastructure

Shared training infrastructure in `shared/`: unified Config system (Device/Model/Data/Train/Eval), standardized DataLoader (raw/cached/full/core modes), PKSampler for balanced mini-batches, ArcFaceHead/LDA/MLP classifier interfaces, and silhouette + consistency evaluation functions. LoRA training pipeline supports bf16 mixed precision, `torch.compile` acceleration, fused AdamW, gradient clipping, checkpoint persistence, and TensorBoard logging. `simlar/` provides a Rust/PyO3 library for high-performance batched character n-gram Jaccard and normalized Levenshtein similarity.

## License

GPL-3.0. See [LICENSE](LICENSE).
