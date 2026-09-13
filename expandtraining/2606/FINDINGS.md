# 2606 Findings: Model Failure and Training Performance

Date: 2026-09-13
Scope: `expandtraining/2606` lane only. StyleGPT is excluded.

Method note: per-rule numbers were reconstructed from the saved validation
artifacts (`validation/rows.parquet` + `validation/embeddings.npy`) by joining
on `(triplet_id, role)` and computing cosines per triplet. The provenance
probe is a logistic regression (StratifiedKFold, up to 4 folds) on anchor
embeddings grouped by rule into domain labels. All numbers below are from
existing artifacts; no new training was run.

## 1. Summary

- The 1h model learns content/topic and corpus provenance, not style.
- The one content-controlled rule, `origin_vs_para`, fails in 1h
  (violation 63%) and only partly recovers in 3h (violation 26%) while that
  run is confounded and validated on seen speakers only.
- A linear probe recovers generated-vs-real text at ~88-89% and domain at
  ~91% (4-way) from embeddings, so provenance is a dominant axis.
- Validation is blind: it aggregates mixed rules, covers ~6% of val triplets,
  stops mid-batch, and never tests unseen speakers.
- The recent runs also used the slower training path (3 forwards/step,
  non-fused AdamW, per-step logging) and checkpoint only at the end.

## 2. Per-rule validation evidence

Complete val triplets available in the saved row caps: 1h = 128 of 2048,
3h = 336 of 4096. Cosine values are raw embeddings.

### 1h run (`qwen06b_1hour_20260619_r3`)

| rule | n | pos | neg | gap | violation |
|---|---|---|---|---|---|
| `generated_to_target` | 19 | 0.8462 | 0.4522 | +0.3940 | 15.8% |
| `origin_vs_para` | 19 | 0.7644 | 0.7889 | **-0.0245** | **63.2%** |
| `same_content_diff_style` | 18 | 0.8589 | 0.4682 | +0.3908 | 16.7% |
| `same_speaker_real` | 18 | 0.7768 | 0.8069 | **-0.0301** | **50.0%** |
| `target_hard_negative` | 18 | 0.8575 | 0.3628 | +0.4947 | 22.2% |
| `trait_cross_feature_hard` | 18 | 0.9105 | 0.8735 | +0.0369 | 33.3% |
| `trait_positive_vs_negative` | 18 | 0.9105 | 0.7940 | +0.1165 | 27.8% |

Centered cosine for `origin_vs_para`: cpos 0.3160, cneg 0.4289,
cgap **-0.1129**. The content-matched neutral paraphrase is closer than
another line by the same speaker.

### 3h run (`qwen06b_3hour_continue_round3_bookraw_20260620`)

| rule | n | pos | neg | gap | violation |
|---|---|---|---|---|---|
| `bookraw_same_author_cross_work` | 34 | 0.4193 | 0.3682 | +0.0512 | **41.2%** |
| `bookraw_same_language_author_negative` | 34 | 0.5999 | 0.0593 | +0.5406 | 0.0% |
| `bookraw_same_work` | 34 | 0.5999 | 0.0593 | +0.5406 | 0.0% |
| `generated_to_target` | 34 | 0.8610 | 0.7066 | +0.1544 | 17.6% |
| `origin_vs_para` | 34 | 0.9835 | 0.6603 | +0.3232 | 26.5% |
| `same_content_diff_style` | 34 | 0.8559 | 0.6824 | +0.1735 | 26.5% |
| `same_speaker_real` | 33 | 0.9831 | 0.8077 | +0.1754 | 9.1% |
| `target_hard_negative` | 33 | 0.8678 | 0.5477 | +0.3201 | 18.2% |
| `trait_cross_feature_hard` | 33 | 0.9415 | 0.9214 | +0.0202 | 36.4% |
| `trait_positive_vs_negative` | 33 | 0.9415 | 0.9203 | +0.0212 | 48.5% |

3h improves the pure genshin rules but shows the same pattern elsewhere:
`same_work` separates well while `same_author_cross_work` barely moves
(gap 0.05, violation 41%), i.e. bookraw geometry is work/topic, not author.
All genshin validation is on speakers seen in training.

## 3. Provenance probe evidence

Anchor embeddings grouped by rule-derived domain
(`genshin_real` = same_speaker_real/origin_vs_para/same_content_diff_style,
`generated` = generated_to_target/target_hard_negative, `trait` = msynthstel,
`bookraw` = bookraw rules).

### Centroid cosine matrices

1h: `[generated, genshin_real, trait]`

```
[[1.000, 0.782, 0.186],
 [0.782, 1.000, 0.576],
 [0.186, 0.576, 1.000]]
```

3h: `[bookraw, generated, genshin_real, trait]`

```
[[1.000, 0.171, 0.266, 0.443],
 [0.171, 1.000, 0.911, 0.630],
 [0.266, 0.911, 1.000, 0.750],
 [0.443, 0.630, 0.750, 1.000]]
```

### Linear probe (CV accuracy vs majority chance)

| run | task | folds | mean | chance | n |
|---|---|---|---|---|---|
| 1h | 3-way domain | 0.906 / 0.875 / 0.781 / 0.938 | 0.875 | 0.43 | 128 |
| 1h | generated vs rest | 0.907 / 0.837 / 0.905 | 0.883 | - | 128 |
| 3h | 4-way domain | 0.852 / 0.920 / 0.920 / 0.943 | 0.909 | 0.30 | 352 |
| 3h | generated vs rest | 0.864 / 0.889 / 0.923 | 0.892 | - | 352 |

Caveat: small n, 256-dim input, rule-derived labels. The margin over chance
is large enough that the conclusion (provenance is a dominant axis) is safe.

## 4. Root causes

### 4.1 Topic/scene shortcut in `same_speaker_real`

- `import_genshin.py` selects rows ordered by `id`; `build_triplets.py:133`
  uses `rows[(index + 1) % len(rows)]` as positive. Positives are adjacent
  DB rows, typically the same scene/dialogue, so the rule is largely a
  topic-discrimination task.
- Negatives are the next speaker in the speaker list
  (`build_triplets.py:130`), arbitrary topic.
- Result: 50% violation in 1h and it can be satisfied without style.

### 4.2 Generator/provenance shortcut

- `same_content_diff_style` (`build_triplets.py:214-225`) compares a human
  source line against generated text of the same content. A model that
  detects "LLM vs human" satisfies it without style.
- `paratarget_original_push` has the same structure.
- The provenance probe confirms the model has exactly this axis available
  (88-91% linearly separable).
- `generated_to_target` compares generated against two human lines, so it is
  less exposed, but its two references are content-uncontrolled.

### 4.3 Validation blind spots

- `report.md` aggregates pos/neg over mixed rules. The standalone
  `validate.py` collects `max_samples` role-rows and stops mid-batch
  (`validate.py:136-145`): 1h covered 128 complete triplets, 3h 336, of the
  2048 cap. Rules later in the round-robin can be missed entirely.
- All genshin validation uses the same 100 speakers as training
  (`import_genshin.py:73-76` takes the last 20% of each speaker's lines).
- Paratarget split is per-line hash `stable_id(job_id, index, size=2) < "20"`
  (`import_paratarget.py:64`), so one generation job straddles train and val
  and targets are seen in training.
- 1h `data_manifest.json` lists 7 val rules and no `paratarget_original_push`,
  although the checked-in config has `paratarget_original_push_enabled: true`.
  The config was edited after the run; the checked-in files no longer
  reproduce it.

### 4.4 Config semantics and run drift

- `0` means unlimited, not disabled:
  - `genshin_lines_per_speaker: 0` (`import_genshin.py:71`)
  - `paratarget_jobs: 0` and `paratarget_per_target_cap: 0`
    (`import_paratarget.py:38,94`)
- Therefore the 3h config is "all data + bookraw + 4x steps + new seed/lr",
  not a bookraw continuation. The 1h to 3h comparison is confounded.
- `allow_manifest_mismatch_on_resume: true` permits resuming across data
  changes, so the manifest hash guard can be bypassed.
- `remote_train.sh` hardcodes `remote_5090x2.yaml`, while the recent runs used
  the `remote_1hour*.yaml` variants via extra args.

### 4.5 Mixture equalization

`cap_by_rule` (`build_triplets.py:628-655`) round-robins rules to equal counts
(~293 val each in 1h) and `balanced_triplet_order` equalizes batches.
Configured `weights` only scale within-batch loss; they do not reduce the
number of gradient steps spent on cheat-able rules.

Manifest evidence (1h): genshin 5200 train / 1200 val samples, paratarget
29758 / 4242, msynthstel 2048 / 680; 50000 train / 2048 val triplets.
Paratarget filters: `invalid_validation` 756264, `target_cap` 25492,
`length_ratio` 3344, `copied_line` 247. The 3h `data_manifest.json` is not
in the local sync of that run.

Bookraw label quality (from `sample50k/sample_umap/`): 7888 rows, 53
"authors", including genre/source buckets such as `classic_*`, `poetry`,
`佚名`. Top-author counts are ~200 per label, so labels are largely
genre-shaped.

## 5. Training performance findings

### 5.1 Throughput

- 3 forward passes per step in the runs that produced the checkpoints:
  `combine_triplet_forward` exists (`model.py:171`) and is set in
  `remote_5090x2.yaml:75`, but not in `remote_1hour.yaml` /
  `remote_3hour_continue_round3_bookraw.yaml`.
- Non-fused AdamW in the same runs: `fused_adamw: true` only in
  `remote_5090x2.yaml:76`; `build_optimizer` supports it (`train.py:250`).
- Per-step logging: no `logging.train_every_steps` in the used configs ->
  default 1 (`train.py:356`). `log_train_step` / `log_rule_batch_stats` issue
  dozens of `.item()`/`.cpu()` syncs and a per-rule mask loop per step.
  `remote_5090x2.yaml:95` sets 5.
- Spread diagnostics run every step and duplicate work:
  `embedding_spread_loss` computes the covariance, then
  `embedding_spread_diagnostics` recomputes centering + covariance and runs
  `eigvalsh(256x256)` (`losses.py:116-122`, `losses.py:140-146`).
- Worker respawn every epoch: `rule_balance_rotate_each_epoch: true` forces
  `persistent_workers=False` (`train.py:235`), re-pickling the full train
  list (50k, or 220k for 3h) to each worker each epoch.
- Triplet cache unused: `scripts/build_dataset.py` writes `triplets.jsonl`,
  nothing reads it; `train.py:269` rebuilds triplets on every run, including
  the full 2.2 GB paratarget scan for unlimited configs.
- Gradient checkpointing is on in the 3h config (`model.py:136`) although
  0.6B fits a 5090; it is pure recompute overhead at that size.

These are the differences between the checked-in primary config and the runs
actually executed. Expected speedup from combine + fused + log-every-5 is
roughly 1.5-2x per step, but it must be measured with a timed short run, not
assumed.

### 5.2 Reliability

- Checkpoint only after the training loop (`train.py:487`). A crash or
  timeout on a 3h/12k-step run loses all progress, and
  `sync_from_remote.sh` fails its `training_state.pt` check.
- Final partial gradient-accumulation window is underweighted
  (`scaled_loss = loss / grad_accum`, `train.py:442`; accelerate path has the
  same partial-window caveat). The same bug was fixed in stylegpt, not here.
- The 1h log shows `libgomp: Invalid value for environment variable
  OMP_NUM_THREADS` plus repeated weight-loading bars.

### 5.3 Optimization efficacy

- The variance floor `min_dim_std=0.03` is below the typical per-dim std of
  unit-normalized 256-d vectors (~0.06), so the relu term is nearly inactive;
  the covariance decorrelation term still acts and can flatten style
  directions.
- `matryoshka_weight=0.2` applies full margins to 32/64/128-d prefixes from
  step 0, adding early gradient noise to the main objective.
- `max_length=192` truncates length-course positives built from
  4000-6000 chars (`length_course_max_chars`) and bookraw segments up to
  3000 chars, so much of that objective trains on cut text.
- Validation cannot track progress (see 4.3), so experiment iteration costs
  are paid without readable feedback.

## 6. Implications for fixes

Local, high-confidence (measurement):
1. Per-rule validation with complete triplets, all rules included, fixed N
   per rule.
2. Held-out speaker probe (speakers 101-200): nearest-centroid top-1/top-5
   vs chance, and the origin vs own-para vs other-line ordering.
3. Provenance probe as a first-class metric; progress requires it to fall
   while unseen-speaker retrieval rises.

Local, high-probability (signal):
4. Topic-matched genshin triplets using char n-gram Jaccard
   (`tools/simlar`): positive = same speaker, low Jaccard;
   negative = different speaker, high Jaccard, length-matched.
5. Downweight or drop rules whose negative is generated text from
   early-stopping metrics; keep `generated_to_target` as the paratarget
   signal.

Config toggles to verify (fixed data mix, fixed seed, fixed steps, judged by
the new probe):
- Jaccard-matched triplets on/off
- spread regularization on/off
- `origin_vs_para` weight x3 with generated-negative rules removed
- Matryoshka 0.2 vs 0
- Jaccard matching on positives only vs both sides

Kill criterion: if held-out retrieval stays at chance while provenance
accuracy stays above ~85%, the limitation is the representation/objective,
not the data mixture.

## 7. Artifacts used

```
expandtraining/2606/outputs/runs/qwen06b_1hour_20260619_r3/validation/{rows.parquet,embeddings.npy,cosine_summary.json,report.md}
expandtraining/2606/outputs/runs/qwen06b_3hour_continue_round3_bookraw_20260620/validation/{rows.parquet,embeddings.npy,cosine_summary.json,report.md}
expandtraining/2606/outputs/runs/qwen06b_1hour_20260619_r3/data_manifest.json
expandtraining/2606/outputs/runs/qwen06b_3hour_continue_round3_bookraw_20260620_sample50k/sample_umap/{sample_counts.json,bookraw_author_plot_summary.json}
```
