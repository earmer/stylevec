# 2606 Curriculum: Strategies, Comparisons, Ablations

Date: 2026-10-13
Scope: extends the curriculum outline in `automating.md` §3 for the `expandtraining/2606` lane.
Code: `src/curriculum.py` (strategies), `src/ablation.py` + `scripts/ablation.py` (plan / compare / tiers),
specs in `configs/ablations/`. No GPU run has used any of this yet; every number below is from a
dataset build or a saved artifact.

## 1. What a curriculum result has to survive

A staged run can beat the old training order for two separate reasons:

- **exposure**: it spends a different share of the budget on each rule;
- **order**: it spends that share at a different time.

Only the second is a curriculum effect. Re-weighting alone can produce the first. Published controls
(Hacohen & Weinshall 2019; Wu, Dyer & Neyshabur 2021) find that ordering often adds little over
random or reversed order under a standard budget, and helps mainly with short budgets or noisy labels.
Both conditions may apply here, but the experiment has to be able to report "no order effect".

So every comparison in this lane is built to change one thing, at equal budget, scored on one
evaluation set. Headline metrics stay those of `automating.md` §3 and `FINDINGS.md` §6. Held-out
speaker retrieval has to rise and the provenance probe has to fall. Loss curves are not evidence.

## 2. Strategies

All modes share one budget accounting: `max_steps × batch × grad_accum × world` rows, drawn in a
single pass, with the stage set by position (`train.py:curriculum_row_budget`).

| knob | values | what it does |
|---|---|---|
| `curriculum.mode` | `staged` | stage mixes, easy-heavy → hard-heavy (default stages: hard 5% → 25% → 50%) |
| | `anti` | the staged schedule reversed. Same exposure as staged, opposite order |
| | `mixed` | staged's time-averaged mix at every position. Same exposure, no order |
| | `uniform` | every rule equally likely, tiers ignored. Roughly the legacy rule-balanced order |
| `curriculum.pacing` | `step` / `linear` | switch mixes at boundaries / interpolate between stage midpoints |
| `curriculum.stages` | list | `until` progress boundaries and tier weights per stage |
| `curriculum.tiers` | `{rule: easy\|mid\|hard\|drop}` | move a rule to another tier, or out of training |
| `curriculum.rule_shares` | `{rule: weight}` | weight rules within their tier (default 1 each) |

Implementation guarantees, each covered by a test in `tests/test_curriculum_strategies.py`:

- staged, anti and mixed have **identical expected draws per rule**. Schedules are tabulated over
  1000 progress buckets, and mixed averages the same buckets after renormalising onto tiers that
  have data. Before this change, mixed averaged the raw stage weights. So when a tier was empty,
  staged and mixed spent different exposures (easy 0.54 vs 0.58 with no hard rules).
- A curriculum run no longer gets shuffled by the DataLoader when `rule_balanced_batches: false`
  (`train_loader_shuffles`). Before, that combination silently turned staged into mixed.
- `train.max_seconds > 0` under a curriculum prints a warning. The stages follow `max_steps`, so a time
  stop cuts off the hard stage. The ablation planner rejects it outright.
- Every run now writes `config.yaml` (exact config) and, under a curriculum, `curriculum_schedule.json`.
  That file records realised tier share per tenth of training, plus draws and repeats per rule: evidence
  that the run did what its arm claims.

## 3. Comparison design

### 3.1 The decomposition (round A)

| comparison | differing key | isolates |
|---|---|---|
| mixed vs uniform | `curriculum.mode` | exposure: tier weights vs one share per rule |
| staged vs mixed | `curriculum.mode` | **order**: the curriculum proper |
| staged vs anti | `curriculum.mode` | direction: is easy-first the useful order? |
| staged_linear vs staged | `curriculum.pacing` | pacing shape (tier shares move by <0.01) |

Read the result as a chain. If mixed beats uniform but staged ≈ mixed, the gain is re-weighting, and
the later rounds can drop ordering. If staged ≈ anti ≈ mixed, ordering does nothing at this budget.

### 3.2 Rules the planner enforces

`ablation.py plan` refuses a spec, before any GPU time, when:

1. a comparison differs in more keys than it declares (exactly one key if it declares none). This is
   the 1h → 3h failure in `FINDINGS.md` §4.4, where four variables changed at once;
2. arms spend different budgets (`max_steps` or rows), unless `allow_budget_change: true`;
3. any arm sets `train.max_seconds > 0`;
4. any arm runs without `curriculum.enabled` (so every arm uses the same accounting; `uniform`
   stands in for the legacy order);
5. an arm changes `validation.*`, `data.speakers` or `data.heldout_speakers`. These would change
   what is measured, or leak held-out identities;
6. an override names a key absent from the base config. Unknown keys are treated as typos, except
   under `curriculum.` or when listed in `new_keys`.

All arms are scored on **one evaluation config** (`eval.yaml`, generated from base + `common`). An arm
that changes `data.*` therefore trains on its own triplets but is measured on the shared validation
set. An arm that changes `model.*` gets `eval__<arm>.yaml`, which keeps the shared data but uses
the arm's model shape.

### 3.3 Seeds and verdicts

`compare` pairs arms by seed and reports the mean and sd of the paired delta. It calls "better" or
"worse" only when |Δ| > 2·sd/√n. With 3 seeds that corresponds to roughly p ≈ 0.18, two-sided. It is
a screening rule, not a significance test: confirm a winner with more seeds before building on it.
With one seed it reports `n<2` and makes no call.

### 3.4 Combinations: twin arms and interactions (round D)

Changing stage weights changes exposure too. So "late hard share 0.65 vs 0.50" between two staged
arms confounds order with exposure. The fix is a 2×2 factorial with a mixed twin for each schedule:

```
              default (hard 0.50)   hard 0.65
order on      staged                staged_hard65
order off     mixed                 mixed_hard65
```

The order effect is read within each column. `interactions` reports the difference of the two paired
order effects: does ordering matter more when the late stage is harder? The same pattern works for
any schedule knob, including stage boundaries, easy floor and pacing.

### 3.5 Leave-one-rule-out is not "minus one rule"

Dropping a rule hands its draws to the other rules of its tier. Dropping a tier's only rule hands the
whole tier's share to the other tiers. Round B's exposure audit shows the size: dropping
`same_speaker_topic_matched` empties the hard tier and raises every other rule's exposure by
25–45%. That arm measures "hard tier present vs absent", not one rule.

## 4. Exposure audit (measured 2026-10-13, `remote_5090x2.yaml`)

Dataset build: `outputs/datasets/counts_remote_5090x2_20261013/manifest.json`. Budget per run:
3000 × 32 × 4 × 2 = 768,000 rows over 50,000 train triplets.

| rule | tier | train triplets | repeats, uniform | repeats, staged/mixed/anti |
|---|---|--:|--:|--:|
| same_speaker_topic_matched | hard | 3,535 | 21.7 | **55.4** |
| same_speaker_real | easy | 4,105 | 18.7 | 16.0 |
| origin_vs_para | mid | 4,106 | 18.7 | 14.8 |
| generated_to_target | easy | 11,402 | 6.7 | 5.8 |
| target_hard_negative | mid | 11,402 | 6.7 | 5.3 |
| paratarget_original_push | mid | 11,402 | 6.7 | 5.3 |
| trait_positive_vs_negative | easy | 1,024 | 75.0 | 64.1 |
| trait_cross_feature_hard | mid | 1,024 | 75.0 | 59.5 |
| length_course_genshin | easy | 1,000 | 76.8 | 65.7 |
| length_course_msynthstel | easy | 1,000 | 76.8 | 65.7 |

Findings:

1. **The hard tier is one rule with 3,535 triplets, reused ~55×.** Staged spends about 34 of those
   passes in the last third. A staged arm can lose to mixed by memorising that pool late. That would be
   an overfitting effect, not an ordering one. A probe build with `genshin_lines_per_speaker: 256` and
   `max_triplets: 150000` (`outputs/datasets/probe_lines256_cap150000_20261013`) gives 13,590
   topic-matched triplets, about 14× repeats. This is a base-config decision, and the same for every
   arm. It should be made **before** round A, because it changes how much "hard" can be memorised.
   `max_triplets` matters as much as lines per speaker: `cap_by_rule` equalises rules at about
   50,000 / 11 per rule.
   **Decided 2026-10-18:** after `curriculum_a_1gpu` (staged best on held-out top-1 but weakest on
   both same-speaker gaps), the base moved to 256 lines and a 150k cap (dataset `d-350af1498f6a`:
   13,590 topic-matched, 15,639 `same_speaker_real`, 15,640 `origin_vs_para`). Its validation set
   moved from 64 to up to 512 scored triplets per (rule, seen/held-out) bucket, about ±2 points of
   standard error on a violation rate instead of ±6.
2. **Rules are uniform within a tier, so splitting a rule buys budget.** The length course is split per
   domain and holds 2 of the 5 easy-tier slots. As a result `same_speaker_real`, the core identity rule,
   gets the same 8.6% of the budget as `length_course_msynthstel`. Round B tests
   `rule_shares: 0.5` for each half.
3. The trait and length-course pools (about 1,000 each) are reused 60–77× in every arm. Their loss
   weights (0.08–0.14) limit the gradient they contribute, but not the budget they consume.

## 5. Ablation catalogue

| spec | runs | question | gate before running |
|---|--:|---|---|
| `curriculum_a_order.yaml` | 15 | exposure vs order vs direction vs pacing | `curriculum_a2_1gpu` read |
| `curriculum_b_rules.yaml` | 15 | which rules earn their place; length-course share | round A done; set `common.curriculum.mode` to its winner |
| `curriculum_b_bookraw.yaml` | 9 | keep / drop / demote `bookraw_same_work` | `BOOKRAW=1` sync; build dataset `d-60e5adb1cb07` |
| `methods_c.yaml` | 12 | spread reg, matryoshka, max_length 384 | round A control chosen; `no_spread` mostly tests the variance floor (see `spread_a_1gpu`) |
| `curriculum_d_factorial.yaml` | 12 | does ordering scale with late hard share | round A shows an order effect |
| `curriculum_a_1gpu.yaml` | 3 | round A's exposure and order steps, 1 seed, 1×5090 (`grad_accum` 8, same row budget); pinned to the old 64-line base | done 2026-10-18; trained heads lost to `--prune` |
| `curriculum_a2_1gpu.yaml` | 6 | the same on the refined base, 2 seeds, stage-boundary weight snapshots | none; ~10 h |
| `spread_a_1gpu.yaml` | 4 | do decorrelation (standardised), uniformity, or a 5× head lr widen the embedding and help retrieval | `curriculum_a2_1gpu` read; set `common.curriculum.mode` to its winner; ~6.5 h |
| `smoke_1gpu.yaml` | 2 | pipeline check: 40 steps, both arms, train→validate→pull→compare | none; ~4 min |

Stop rule: if any round-A arm meets the kill criterion of `FINDINGS.md` §6, B–D do not run.
That criterion is held-out retrieval at chance while provenance stays above ~0.85. `compare` flags it
per arm, as top-1 lift over chance < 0.05 (adjust under `kill:` in a spec). A schedule cannot fix that; the representation or objective is next.

Headline metric per spec is set in its `headline:` list. All metrics are in `compare.json`:
`heldout_top1(_lift)`, `heldout_top5`, `seen_top1`, `memory_gap` (seen − held-out),
`provenance_acc/lift`, `macro_violation`, `gap:<rule>` / `violation:<rule>` for focus rules, and the
validation geometry: `effective_rank`, `top_component_ratio` and `length_r2` (from `cosine_summary.json`).
Per-rule values use held-out-speaker rows where a rule has them.

### Run protocol

On the Mac (Stage 0a), then push:

```bash
uv run python expandtraining/2606/scripts/ablation.py plan expandtraining/2606/configs/ablations/curriculum_a_order.yaml
uv run python expandtraining/2606/scripts/build_dataset.py --config <configs plan.md lists as unbuilt>
uv run python expandtraining/2606/scripts/preflight.py local expandtraining/2606/configs/ablations/curriculum_a_order.yaml
SSH_PORT=<port> bash expandtraining/2606/scripts/sync_to_remote.sh root@<host>:/root/stylevec <datasets plan.md names>
```

`plan` writes per-(arm, seed) configs to `configs/ablations/generated/<name>/` (gitignored, synced) and
`outputs/ablations/<name>/plan.md`. The plan names each run's prebuilt dataset `d-<data key>`, a hash of
the config sections the builder reads plus the builder's source. Arms that share data share one
directory: rounds A, B, C and D all train on `d-350af1498f6a`. Once every dataset is built, the
exposure audit uses each arm's own counts, so data-changing arms no longer need a rebuild to be exact.
`preflight.py local` checks the datasets and runs the tests, then records a code fingerprint in
`generated/<name>/preflight.json`.

On the instance, once per box (`remote_setup.sh` builds `.venv` from `uv.lock` via the region's
pip mirror, using a uv cache on autodl-fs), then the queue inside tmux:

```bash
bash expandtraining/2606/scripts/remote_setup.sh
tmux new -s queue 'bash expandtraining/2606/scripts/run_queue.sh expandtraining/2606/configs/ablations/curriculum_a_order.yaml --prune'
```

Or all of it from the Mac in one command, once the box has its `.venv`:

```bash
bash expandtraining/2606/scripts/launch_queue.sh <pro-id> expandtraining/2606/configs/ablations/curriculum_a_order.yaml --prune
```

`launch_queue.sh` runs Stage 0a, starts the box (retrying every 15 s for as long as no GPU is free on
its host), pushes the code and the plan's datasets, runs the queue's dry run (Stage 0b) and starts the
queue in tmux, logging to `/root/queue_<spec>.log`. If anything fails after it started the box, it
stops the box again.

The queue starts with Stage 0b: the code matches the 0a fingerprint, the GPU count equals
`world_size`, bf16 kernels exist for the card, and the base model, datasets and disk headroom are in
place. Before each run it does 0c (configs present, datasets match their configs, no
`allow_manifest_mismatch_on_resume`). Then it trains (resuming from `checkpoints/latest` if a run was
cut off), validates on the eval config, and with `--prune` deletes `training_state.pt` once
`weights.pt` (LoRA plus the projection head) sits beside it. Finished runs
are skipped, so re-running the queue continues where it stopped. The box powers off at the end, so
billing stops (`--keep-up` leaves it running, and so does stopping the queue by hand). SwanLab mirrors
each run's TensorBoard scalars live and logs the final `compare` metrics under `final/`, grouped by
spec and tagged by arm and seed.

Back on the Mac:

```bash
bash expandtraining/2606/scripts/sync_from_remote.sh --instance <pro-id> curriculum_a
uv run python expandtraining/2606/scripts/ablation.py compare expandtraining/2606/configs/ablations/curriculum_a_order.yaml
```

`--instance` starts a stopped box for the pull and stops it again. Only results come back
(`validation/`, configs, schedule, logs) plus `checkpoints/latest/weights.pt`, enough to re-validate.
Adapters and the `step_NNNNN/` snapshots come back only with `WITH_ADAPTER=1`. `compare`
writes `outputs/ablations/<name>/compare.md` and lists missing runs. Partial results are fine: pairs
are formed only on seeds where both arms finished.

## 6. Measured tiers

In `automating.md` §3, difficulty means how far a triplet blocks the shortcuts, not a label someone
assigns. `ablation.py tiers <validation dir>` therefore tiers each rule by a prior checkpoint's violation rate
(held-out rows preferred; easy < 0.15 ≤ mid ≤ 0.35 < hard) and prints a `curriculum.tiers` block
for a spec arm. On the 3h run (`qwen06b_3hour_continue_round3_bookraw_20260620`):

| rule | manual | measured (3h violation) |
|---|---|---|
| trait_positive_vs_negative | easy | **hard** (48.5%) |
| bookraw_same_author_cross_work | mid | **hard** (41.2%) |
| trait_cross_feature_hard | mid | **hard** (36.4%) |
| generated_to_target | easy | **mid** (17.6%) |
| same_content_diff_style | hard | **mid** (26.5%) |
| same_speaker_real, bookraw_same_work, bookraw_same_language_author_negative | easy / easy / mid | easy (≤9.1%) |

Illustrative only. That validation used seen speakers, covered 336 triplets, and predates
`same_speaker_topic_matched`. Use the round-A control's validation instead. Caveat: label noise also
measures as hard. If the trait rules come out hard, it is unclear whether they need style or are just
noisy at a 0.10 loss weight. Audit before promoting them.

## 7. Not implemented, and why

- **Gating stage changes on measurement** (`automating.md` §3: advance on held-out retrieval up and
  provenance down). Needs held-out evaluation during training. The loop evaluates only at the end,
  and checkpoints overwrite `latest`. First prerequisite: step-numbered checkpoints, which also allow
  learning-curve comparisons (area under the held-out curve, not just the endpoint).
- **Self-paced / loss-driven rule weights.** Needs an online sampler fed by per-rule violation EMAs.
  Under `accelerate`, each rank would adapt on its own shard and drift apart unless synced.
  Not worth building until round A shows ordering matters at all.
- **Hard-negative mining with a checkpoint.** An offline step between rounds. Mined negatives need the
  label and duplicate audit of `automating.md` §3 before they enter training.
- **`origin_vs_para` weight ×3** (`FINDINGS.md` §6). Not expressible yet: that rule shares
  `weights.genshin_real` with `same_speaker_real`. It needs its own weight key first.
- **Objective changes** (ArcFace / SupCon, `automating.md` §5). These belong to a separate track:
  they change what "hard" means, so mixing them into these rounds would confound both.

## 8. Open items found on the way

- The bookraw spec names `data/bookraw/books_cleaned.parquet`. Locally that is a link to the volume's
  file; `sync_to_remote.sh BOOKRAW=1` puts a real copy there on the box. Its dataset is
  `d-60e5adb1cb07` on the current base (not built yet; the yields below are from the old 64-line base,
  `d-0b8e3cafc83a`).
- All specs assume `world_size: 2`. On a 1×5090 box the queue refuses them in 0b. Run them on 2×5090,
  or write a 1-GPU twin with `train.grad_accum` doubled so the row budget stays the same.
- `bookraw_max_chars: 1800` in the bookraw spec is copied from `dryrun.yaml`. The handoff measured that
  the 160–1800 window discards over half of the repaired `primary_clean` (median 134 chars). The
  build gives 4,377 `bookraw_same_work`, 938 `bookraw_same_author_cross_work` and 4,378
  `bookraw_same_language_author_negative` train triplets. The cross-work rule, the one the round is
  judged on, is the thinnest of the three.
- Use `remote_5090x2.yaml` as the base. The `remote_1hour*` and colab configs have time caps and
  fail the planner.

- **The embedding uses ~4 of 256 directions** (2026-10-18). `curriculum_a_1gpu` uniform s42: validation
  effective rank 3.97, top component 49%, and length predictable from the embedding (linear R² 0.59).
  The covariance penalty works on raw covariances; unit vectors have per-dimension std ~0.05, so it is
  ~1e-6 and, at weight 0.005, in effect off. `regularization.decorrelation: correlation` standardises
  dimensions first (mean squared correlation: 0.31 on those embeddings, 0.01 isotropic);
  `regularization.uniformity_weight` adds a uniformity loss within anchors, positives and negatives;
  `train.head_lr` gives the projection head (and fusion) its own learning rate. All default off, so
  existing specs are unchanged. `spread_a_1gpu.yaml` screens them.
- Margins are per rule already: `same_speaker_topic_matched` and `origin_vs_para` use the largest (0.18).
  A violation rate depends on its margin, so compare rules by `gap`, not by violation rate.

## References

- Bengio, Louradour, Collobert, Weston. *Curriculum Learning.* ICML 2009.
- Kumar, Packer, Koller. *Self-Paced Learning for Latent Variable Models.* NeurIPS 2010.
- Hacohen, Weinshall. *On the Power of Curriculum Learning in Training Deep Networks.* ICML 2019.
- Platanios et al. *Competence-based Curriculum Learning for Neural Machine Translation.* NAACL 2019.
- Wu, Dyer, Neyshabur. *When Do Curricula Work?* ICLR 2021.
