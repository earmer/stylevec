# spread_a_1gpu results (2026-10-19)

Spec: `configs/ablations/spread_a_1gpu.yaml`. One seed (42), 1x5090, 3000 steps, `curriculum.mode: staged`,
dataset d-cfa298ce2f92 (120,811 train triplets; the audited data, not the a2 data). Each run took about 70 min.

## Data

- Full compare: `outputs/ablations/spread_a_1gpu/compare.md` and `compare.json` (gitignored; rebuild with
  `uv run python expandtraining/2606/scripts/ablation.py compare expandtraining/2606/configs/ablations/spread_a_1gpu.yaml`).
- Per-run validation: `outputs/runs/spread_a_1gpu__<arm>__s42/validation*/` (final, `validation_step_01050`, `_02100`).
- Train curves: `tf-logs/spread_a_1gpu__<arm>__s42`, and SwanLab project `earmercarey/stylevec-2606`
  (`final/*` and `validation/*` scalars, readable without the box).

## Arms

| arm | change | status |
|---|---|---|
| control | none | done |
| decorr | correlation decorrelation, weight 0.02 | stopped at about step 1800 after blowing up; not validated |
| center | `model.center: running`: subtract an EMA mean of pooled hidden states before the projection | done |
| center_decorr | center + decorr | done |
| uniformity, decorr_headlr | — | not run |

## What happened in training

On this data `control` never left the narrow cone. Train cosines stayed at 0.85–0.96, violations at 0.6–0.95,
and grad_norm rose to 5.4. a2 staged escaped only because its data had 68 memorisable target_hard_negative
negatives; this data has 23k. `decorr` escaped until step ~950, then blew up (train effective rank 37,
violations 0.98). `center` escaped and stayed out: at step 1300 its violations were 0.48 (control 0.84) and its
negative cosine 0.64 (control 0.90), with grad_norm about 0.35. The shared offset it removes has norm 135
against a pre-norm output of about 45.

## Final validation

| metric | control | center | center_decorr |
|---|---|---|---|
| heldout_top1 | 0.095 | **0.108** | 0.100 |
| heldout_top1_macro | 0.091 | **0.108** | 0.097 |
| heldout_top5 | 0.393 | 0.410 | 0.411 |
| whitened_top1 | 0.194 | **0.250** | 0.210 |
| provenance_acc (lower is better) | **0.669** | 0.696 | 0.693 |
| gap: same_speaker_topic_matched | 0.059 | **0.133** | 0.103 |
| gap: origin_vs_para | 0.012 | **0.189** | 0.071 |
| gap: same_speaker_real | 0.025 | 0.118 | **0.172** |
| validation pos / neg cosine | 0.84 / 0.79 | 0.63 / 0.43 | 0.52 / 0.37 |
| effective_rank | 5.6 | 3.7 | 3.7 |
| length_r2_speaker (lower is better) | **0.29** | 0.77 | 0.79 |

Speaker-paired held-out top-1 (20 held-out speakers, one seed):

- center − control: +0.016, SE 0.039, within noise. center wins on 11 speakers, loses on 7. Two speakers
  swing by about 0.5 in opposite directions (托萝莎 +0.53, 爱诺 −0.48). Without them the mean is still about
  +0.015, at about 1.2 SE.
- center_decorr − center: −0.011, SE 0.010, within noise.

## Trajectory (heldout_top1_lift)

| arm | step 1050 | step 2100 | final |
|---|---|---|---|
| control | 0.060 | 0.044 | 0.045 |
| center | 0.065 | 0.053 | 0.058 |
| center_decorr | 0.073 | 0.056 | 0.050 |

Every arm peaks at step 1050 (35% of training, when the staged hard share rises) and then slips, as in a2. The
same-speaker gaps keep growing after that point (center topic_matched 0.025 → 0.133) even though retrieval falls.

## Reading

1. Keep `center`. It fixes the cone and every gap grows. Its retrieval gain is likely, but not significant
   on one seed.
2. Drop decorrelation. It does not raise effective rank once centered and costs most of the origin_vs_para
   gap.
3. Watch the length signal. Centering makes length explain much more of the embedding (0.29 → 0.77 within
   speaker). Check it before treating the larger gaps as style.
4. Open problems, in order of how much they would move the result:
   - the late-training drop after step 1050, which centering does not fix (an earlier checkpoint, or a slower
     hard-share ramp);
   - removing the length signal from `center`;
   - a second seed of center vs control.
