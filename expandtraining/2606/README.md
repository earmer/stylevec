# StyleVec 2606 Training Plan

This folder is the 2606 training expansion workspace. The plan is deliberately
small: first make one clean training path work, then add more broad corpora only
after the Genshin + paratarget + msynthstel loop proves useful.

The high-level thesis is:

> Do not build a general research platform first. Build a compact, reproducible
> training lane for Genshin top100 + paratarget + msynthstel, with strict
> imports and a real dry-run. AO3/bookraw/datadelta can wait.

## Current Facts

Checked on 2026-06-10:

- `data/genshin/genshin.db`
  - `264221` rows
  - `4384` speakers
  - `262196` rows with both `origin_text` and `para_text`
- `data/paratarget/generations_pairwise_validation_2p5cny.jsonl`
  - `789301` lines
  - fields include `source_speaker`, `target_speaker`, `source_lines`,
    `generated_lines`, `reference_ids`, `validation`
- `data/paratarget/speakers_top100.json` exists and should drive the first
  speaker set
- `data/datasets/msynthstel/data/*.parquet` exists and should stay as a
  low-weight trait calibration source
- `artifacts/base-models/qwen-3-0.6b` exists
- AO3/bookraw/datadelta exist, but they are not first-lane data

## Scope

First implementation:

1. [x] Genshin top100 speakers as the real-style anchor.
2. [x] Paratarget valid generations as same-content different-style expansion.
3. [x] msynthstel as low-weight trait-axis calibration.
4. [x] Qwen 0.6B + LoRA + mean pooling + triplet loss.
5. [x] TensorBoard, checkpoint, dry-run, validation, and UMAP.
6. [x] Remote `5090 * 2` entry through one script: `scripts/remote_train.sh`.

Delayed:

- [ ] AO3 and bookraw author style. Useful later, but easy to overfit topic/work.
- [ ] datadelta. Very delayed. If used, it needs embedding matching to build rough
  same-meaning different-style pairs; same-style different-meaning pairs are
  much harder to identify, so it should not drive the first supervised setup.

## Self-Contained Rule

Only, but all code files must be self-contained inside `expandtraining/2606/`.

Data, base models, outputs, and dependency management stay at repository level:

- data paths remain under `/Users/earmercarey/stylevec/data/...`
- model paths remain under `/Users/earmercarey/stylevec/artifacts/...`
- dependencies stay in `/Users/earmercarey/stylevec/pyproject.toml`
- no local `expandtraining/2606/pyproject.toml`
- no local `expandtraining/2606/uv.lock`

The code inside this folder MUST NOT import/softlink/etc. parent repo Python modules. If a
helper is needed, copy the small amount of code into `src/` and own it here.

## Directory

```text
expandtraining/2606/
  README.md
  configs/
    dryrun.yaml
    remote_5090x2.yaml
    colab_t4.yaml
    colab_smoke.yaml
  colab/
    colab_train.ipynb
    prepare_drive.sh
    requirements.txt
  src/
    config.py
    records.py
    import_genshin.py
    import_paratarget.py
    import_msynthstel.py
    build_triplets.py
    dataset.py
    model.py
    losses.py
    train.py
    validate.py
  scripts/
    build_dataset.py
    train.py
    validate.py
    dryrun.sh
    remote_train.sh
    validate.sh
    sync_to_remote.sh
    sync_from_remote.sh
    tensorboard.sh
  tests/
    test_imports.py
    test_triplets.py
    test_model_shapes.py
```

This is intentionally flat. `src/` is the package root.

## Entry Contract

Keep user-facing commands narrow:

```bash
bash expandtraining/2606/scripts/dryrun.sh
bash expandtraining/2606/scripts/remote_train.sh <run_name>
bash expandtraining/2606/scripts/validate.sh <run_name>
```

Python entrypoints stay narrow:

```text
build_dataset.py: --config --run-name
train.py:         --config --run-name [--resume]
validate.py:      --config --run-name [--resume]
```

The messy work belongs inside importer files and config files, not in training
entrypoints.

Curriculum strategies and ablations (one config per arm and seed, one shared
evaluation set, paired comparisons) are described in `CURRICULUM.md`:

```bash
uv run python expandtraining/2606/scripts/ablation.py plan <spec> [--manifest <manifest.json>]
uv run python expandtraining/2606/scripts/ablation.py compare <spec>
uv run python expandtraining/2606/scripts/ablation.py tiers <run>/validation
``` The core training code should receive already-normalized triplets.

## Data Contract

Normalize all imported rows to:

```text
sample_id
text
style_id
style_type      # speaker / generated_target / trait
domain          # genshin / paratarget / msynthstel
content_id
source_id
language
split
quality
meta
```

Normalize triplets to:

```text
anchor_text
positive_text
negative_text
rule
weight
margin
split
```

Keep metadata compact. Do not write long raw corpus excerpts to logs or
TensorBoard.

## First-Lane Data Rules

### Genshin top100

Use `speakers_top100.json`. It is an explicit configured input, so a missing
file is an error, not a signal to compute another speaker list.

Filters:

- speaker is in top100
- `origin_text` is non-empty
- `para_text` is non-empty when a rule needs it
- text length is inside config bounds
- exact normalized duplicates are removed

Triplets:

- `same_speaker_real`
  - anchor: speaker A real line
  - positive: another speaker A real line
  - negative: speaker B real line
- `origin_vs_para`
  - anchor: real line
  - positive: another real line from the same speaker
  - negative: the neutral paraphrase
### Paratarget

Use only validated jobs:

- `validation.valid == true`
- source and target speakers are both in top100
- source speaker differs from target speaker
- source/generated line counts match
- generated line is non-empty and not copied from source

Triplets:

- `generated_to_target`
  - anchor: generated target-style line
  - positive: real target-speaker line
  - negative: real source-speaker line
- `same_content_diff_style`
  - anchor: source line
  - positive: real source-speaker line
  - negative: generated target-style line
- `target_hard_negative`
  - anchor/positive: target speaker A
  - negative: generated or real line for target speaker B

Generated data must be capped per target speaker so it cannot outweigh real
Genshin lines.

### msynthstel

Keep msynthstel in the first lane, but keep it small and low-weight.

Filters:

- require non-empty `positive`, `negative`, `feature`, `feature_clean`
- use the existing train/test split
- cap rows per feature in dry-run and first remote runs

Triplets:

- `trait_positive_vs_negative`
  - anchor: positive example for feature F
  - positive: another positive example for feature F
  - negative: negative example for feature F
- `trait_cross_feature_hard`
  - anchor/positive: same feature F
  - negative: positive example from another feature

The purpose is trait-axis calibration, not replacing speaker-style training.

## Later Data

AO3/bookraw:

- add after Genshin + paratarget + msynthstel training has a usable validation
  signal
- split by work, not random segment
- same-author positives should prefer different works
- hard negatives should match fandom/topic/language where possible

datadelta:

- not supervised style data by default
- if used later, first build rough semantic neighborhoods with embeddings
- use only as low-weight background or calibration until proven useful

## Model

Base model:

- Qwen 0.6B from `artifacts/base-models/qwen-3-0.6b`
- `transformers.AutoModel`
- LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`
- mean pooling over the attention mask
- projection head to the configured style dimension
- L2 normalization

Configurable:

- selected hidden layers
- layer combine mode: `last`, `weighted_sum`, `concat`
- projection dimension
- LoRA rank/alpha/dropout
- margin and source weights

Trainable parameter count must be computed from actual `requires_grad`
parameters and logged. Do not report hand-estimated counts as actual counts.

## Loss

Start with weighted triplet margin loss:

```text
loss = relu(d(anchor, positive) - d(anchor, negative) + margin) * weight
```

Current experiment configs keep that full-dimension triplet loss as the main
objective, then add narrow auxiliary pressure:

- spread regularization for collapse checks
- Matryoshka prefix triplet loss for smaller usable vector prefixes
- low-weight length/paratarget course rules for length-invariance probes

## Logging

TensorBoard must log:

- train loss
- validation loss
- positive cosine
- negative cosine
- violation ratio
- learning rate
- gradient norm
- trainable parameter count
- sample counts and triplet counts
- config snapshot
- per-rule loss/violation/mix for diagnosing in-epoch rule shifts
- Matryoshka prefix loss
- validation vector probes and length leakage metrics

TensorBoard should log to:
- ./expandtraining/2606/tf-logs when MPS available
- /root/tf-logs when CUDA available

## Dry-Run

Dry-run should be as small as possible while still touching the real path.

Suggested limits:

```yaml
data:
  speakers: 4
  genshin_lines_per_speaker: 4
  paratarget_jobs: 4
  msynthstel_features: 2
  max_triplets: 16
train:
  batch_size: 2
  max_steps: 1
  num_workers: 0
validation:
  max_samples: 16
```

It must still run:

- import
- triplet build
- tokenization
- model forward
- loss backward
- one optimizer step
- validation loss
- checkpoint write
- TensorBoard write

Dry-run command:

```bash
bash expandtraining/2606/scripts/dryrun.sh
```

## Remote Training

A new box needs the environment once: `bash expandtraining/2606/scripts/remote_setup.sh` installs
`.venv` from `uv.lock` through the region's pip mirror, with the uv cache on autodl-fs so the next
box in the region installs from it. For ablation specs, `scripts/run_queue.sh <spec>` (inside tmux)
runs Stage 0b, then trains and validates each run with resume; see `CURRICULUM.md`, Run protocol.
`DATASET=<name>` makes `remote_train.sh` and `validate.sh` load a prebuilt dataset
(`scripts/build_dataset.py`) instead of rebuilding triplets from raw. `NUM_PROCESSES` sets the GPU count.

Primary remote command:

```bash
bash expandtraining/2606/scripts/remote_train.sh qwen06b_top100_paratarget_msynthstel_r64
```

The script should wrap:

```bash
uv run accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  expandtraining/2606/scripts/train.py \
  --config expandtraining/2606/configs/remote_5090x2.yaml \
  --run-name "$RUN_NAME"
```

Remote defaults:

```yaml
train:
  batch_size: 16
  grad_accum: 8
  epochs: 3
  lr: 2.0e-4
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  dataloader_workers: 8
```

The first remote lane writes a final validation pass and one `latest`
checkpoint. Resume uses the same script with a `--resume` pass-through if
needed. Do not add top-k checkpoint ranking until there is a real model-selection
metric.

## Google Colab (Free T4)

A single-GPU lane for Colab's free T4. It is the same code path as the remote
lane, with a config that adapts to one Turing GPU:

- `configs/colab_t4.yaml` — `precision: fp16` (T4 has no native bf16) and
  `distributed: accelerate` so Accelerator installs the fp16 GradScaler;
  batch 16 x accum 8 = 128 effective; 2 dataloader workers; 3000-step / 3-hour cap.
- `configs/colab_smoke.yaml` — the `dryrun.yaml` shape for CUDA, no bookraw.

Colab has no persistent disk, so code, data and outputs live in Drive:

```text
<Drive>/stylevec-colab/
  code/expandtraining/2606/...   staged from the repo by prepare_drive.sh
  data/...                       repo-relative data files
  outputs/                       symlinked from the workspace (checkpoints)
  tf-logs/                       symlinked from the workspace (TensorBoard)
  base-models/                   Qwen3-0.6B downloaded once from HuggingFace
```

One-time staging from the machine that holds the data:

```bash
bash expandtraining/2606/colab/prepare_drive.sh \
  "/path/to/Google Drive/My Drive/stylevec-colab"
```

Then open `expandtraining/2606/colab/colab_train.ipynb` in Colab with a T4
runtime and run the cells. The notebook copies the code and data into
`/content/stylevec`, symlinks `outputs/` and `tf-logs/` back to Drive, pulls
`Qwen/Qwen3-0.6B` from HuggingFace, runs the smoke test, then trains. Resume
after a disconnect uses the same `--resume` checkpoint in Drive.

Do not install the repo's `pyproject.toml` on Colab: it would replace the
preinstalled CUDA torch build. `colab/requirements.txt` lists only the extra
packages the lane imports.

## Validation

Validation should produce:

```text
outputs/runs/<run_name>/validation/
  rows.parquet
  embeddings.npy
  cosine_summary.json
  umap.csv
  umap.png
  report.md
```

Checks:

- same speaker lines closer than different speaker lines
- generated target-style lines closer to target speaker than source speaker
- hard negatives still too close
- train/validation cosine distributions
- embedding collapse via mean vector norm and PCA top component ratio

Validation command:

```bash
bash expandtraining/2606/scripts/validate.sh qwen06b_top100_paratarget_msynthstel_r64
```

## Sync

Keep sync simple and explicit.

To remote:

```bash
SSH_PORT=12345 bash expandtraining/2606/scripts/sync_to_remote.sh root@host:/root/stylevec \
  counts_remote_5090x2_20261013
```

Pushes code (with `--delete`, run outputs untouched) to the remote root, and the training inputs,
base model and any named prebuilt datasets from `outputs/datasets/` to `DATA_ROOT` (default
`/root/autodl-fs/stylevec`, which survives switching instances in a region). The remote root's
`data/`, `artifacts/` and `outputs/datasets/` become links into it. sha256 is compared on the remote.
`DATA_ROOT=<remote root>` keeps everything on the instance disk. Re-runs skip
unchanged files and resume interrupted ones. `BOOKRAW=1` adds the volume's `books_cleaned.parquet`
as `data/bookraw/books_cleaned.parquet`; `CHECKSUM=1` repairs a copy that fails verification;
`VERIFY=0` skips the hash pass (~15 s for 3 GB).

From remote:

```bash
bash expandtraining/2606/scripts/sync_from_remote.sh --instance pro-xxxx curriculum_a
bash expandtraining/2606/scripts/sync_from_remote.sh root@host:/root/stylevec <run_name>
```

Names are run names or spec names (every `<spec>__*` run). It pulls results only: validation, configs,
schedule and logs. It never pulls `training_state.pt`; `WITH_ADAPTER=1` adds the adapter and `WITH_TB=1`
the TensorBoard logs. `--instance` starts a stopped Pro box for the pull and stops it afterwards.

## Experiments

No checked-in experiment grid yet. First get `dryrun.yaml` and
`remote_5090x2.yaml` working. After the baseline has a usable validation report,
copy `remote_5090x2.yaml` to a named config for exactly one question at a time:
layer fusion, margin, negative mix, or msynthstel weight.

## Implementation Order

1. Root dependency check in `/Users/earmercarey/stylevec/pyproject.toml`.
2. `configs/dryrun.yaml` and `configs/remote_5090x2.yaml`.
3. Genshin importer.
4. Paratarget importer.
5. msynthstel importer.
6. Triplet builder.
7. Dataset/collate.
8. Model and loss.
9. Training loop and TensorBoard log.
10. `dryrun.sh`.
11. `remote_train.sh`.
12. Validation and UMAP.
13. Sync scripts.
14. Only then add delayed broad corpora.

## Kill List

Delete these ideas from the first implementation:

- local `pyproject.toml` under this folder
- local `uv.lock` under this folder
- nested package name `src/stylevec2606`
- many tiny subpackages for train/model/validate/sync
- checked-in duplicate experiment configs before the baseline exists
- AO3/bookraw/datadelta in the first training lane
- batch-hard or contrastive auxiliary loss before a simple triplet baseline
- checkpoint top-k ranking before a real selection metric exists
- CLI flags for every small sampling choice
