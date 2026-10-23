# InfoNCE rounds A and B: results and open questions

Written 2026-10-23 for an outside review. It is self-contained: it covers the project, setup, every run in
these two rounds, per-rule numbers, what we think is happening, and the next experiments under consideration.
Everything is one seed (42) on one RTX 5090. Treat single-arm differences under ~0.02 as noise.

## 1. What the project is trying to do

StyleVec trains a **style embedding**: lines written or spoken by the same character or author should sit close
together whatever they are about, and lines with the same content in a different voice should sit apart.
The failure we keep fighting is **content leakage**: the embedding takes the shortcut of matching topic or
wording instead of voice.

- **Encoder:** Qwen3-0.6B with LoRA, mean pooling, `max_length` 192, then a linear projection to 256 dims,
  L2-normalised. Validation also scores Matryoshka prefixes (32 / 64 / 128 dims).
- **Data:**
  - Genshin dialogue for the top 100 speakers, about 256 lines each, with an LLM paraphrase (`para_text`) of
    almost every line.
  - "paratarget": LLM rewrites of a source speaker's line into a target speaker's style.
  - msynthstel: synthetic trait-controlled text, at low weight.
- **Held-out speakers:** speakers 101–200, never trained on. Retrieval on them is the main generalisation metric.

## 2. Training rules (triplet types)

Each training row is (anchor, positive, negative). The overlap columns are the mean char-bigram Jaccard
overlap, measured on the validation triplets. They are a rough surface-content proxy, since paraphrases share
meaning beyond their bigrams.

| rule | tier | anchor / positive / negative | overlap anchor–pos | overlap anchor–neg | train repeats |
|---|---|---|---|---|---|
| same_speaker_real | easy | real line / same speaker, other line / other speaker | 0.010 | 0.003 | 4.2× |
| origin_vs_para | mid | real line / same speaker, other line / **paraphrase of the anchor** | 0.009 | **0.479** | 4.4× |
| same_speaker_topic_matched | hard | real line / same speaker, low overlap / other speaker, high overlap, length-matched | 0.000 | 0.011 | **14×** |
| generated_to_target | easy | rewrite into target T / real T line / real line of the source speaker | 0.005 | 0.006 | 2.7× |
| paratarget_original_push | mid | rewrite into T / real T line / **the source text it was rewritten from** | 0.004 | 0.268 | 2.5× |
| target_hard_negative | mid | rewrite into T / real T line / rewrite into another target | 0.005 | 0.005 | 2.5× |
| length_course_genshin | easy | short line / **concatenation of same-speaker lines that includes the anchor** / short other-speaker line | **0.480** | 0.004 | 66× |
| length_course_msynthstel | easy | same, on msynthstel | **0.795** | 0.197 | 65× |
| trait_positive_vs_negative | easy | msynthstel trait pairs | 0.203 | 0.680 | 64× |
| trait_cross_feature_hard | mid | msynthstel cross-trait | 0.203 | 0.195 | 59× |

The two rules where the content shortcut points the **wrong** way, origin_vs_para and
paratarget_original_push, are the ones that break under InfoNCE. The two where it points the **right** way,
the length_course rules, are trivially satisfied, and they are repeated about 65 times each.

## 3. Training setup shared by all runs here

- **Steps and batch:** 3,000 steps at batch 32, with grad_accum 8 for these rounds. lr 2e-4, warmup 5%,
  weight decay 0.01, rule-balanced batches.
- **Staged curriculum** (tier shares; the stage ends are exactly where the weight snapshots are taken):

  | stage | steps | easy | mid | hard (topic_matched) |
  |---|---|---|---|---|
  | 1 | 0–1050 | 0.70 | 0.25 | 0.05 |
  | 2 | 1050–2100 | 0.35 | 0.40 | 0.25 |
  | 3 | 2100–3000 | 0.20 | 0.30 | 0.50 |

- **`model.center: running`:** a running-mean centering of the pooled state before the projection. It came
  from spread_a and is now the base.
- **Regularisers:** variance (weight 0.01), covariance (0.005) and Matryoshka (0.2) at [32, 64, 128].
- **Triplet hinge:** `relu(margin − (cos(a,p) − cos(a,n)))`. Per-tier margins: easy 0.08, mid 0.12,
  hard 0.18. origin_vs_para and topic_matched use 0.18.
- **InfoNCE (`train.loss: infonce`):**
  - Cosine over temperature T = 0.05, within each micro-batch of 32 rows.
  - Each anchor's candidates are all 32 positives (its own is the target), the other 31 anchors and all 32
    mined negatives (its own negative is always kept). That is about 95 candidates.
  - Candidates that share the anchor's style label are false negatives, so they are masked out of the
    denominator.
  - `infonce_triplet_weight` adds the hinge on top.

## 4. Metrics

- **heldout_top1_macro:** nearest-centroid speaker retrieval on held-out speakers, averaged over 20 speakers.
  `heldout_top1_lift` is the same number minus chance. `whitened_top1` uses PCA whitening onto 128
  directions fitted on seen speakers.
- **gap:\<rule\>:** mean of cos(a,p) − cos(a,n) on held-out validation triplets of that rule. Above 0 means
  the right ordering. **cgap** is the same computed after removing the validation-set mean embedding. It is
  more honest when the space is anisotropic, and the two can disagree in sign (see §6).
- **violation:** share of triplets with cos(a,p) ≤ cos(a,n).
- **effective_rank:** participation-ratio rank of the embeddings.
- **provenance_acc:** how well a probe recovers the data source (lower is better).
- **length_r2_speaker:** how much of the embedding is explained by text length (lower is better).

## 5. Runs and headline results

| run | what changed | top-1 (macro) | whitened | gap ovp | gap topic | gap real | rank |
|---|---|---|---|---|---|---|---|
| spread_a / infonce_a `center` | hinge only (reference) | 0.108 | 0.250 | +0.189 | +0.133 | +0.118 | 3.7 |
| infonce_a `center_infonce` | InfoNCE only | **0.232** | 0.327 | **−0.262** | +0.017 | +0.146 | 33.2 |
| infonce_a `center_infonce_m64` | + Matryoshka at [64] | ≈ center_infonce at 1050/2100; stopped at ~2100 | | | | | |
| infonce_b `hinge` | InfoNCE + hinge ×10 | 0.176 | 0.263 | −0.134 | +0.093 | +0.185 | 11.1 |
| infonce_b `hinge_chead` | + 8-dim content head + leak penalty | 0.194 | **0.353** | −0.178 | +0.069 | +0.199 | 18.8 |

(ovp = origin_vs_para, topic = same_speaker_topic_matched, real = same_speaker_real. Final weights.)

- **Hinge weight:** the hinge first ran at weight 1. InfoNCE's gradient is scaled by 1/T = 20, so the hinge
  was about 4% of the loss, and the run tracked center_infonce exactly up to step 400. It was stopped and rerun
  at weight 10.
- **Content head (chead):**
  - A separate Linear(pooled → 8) head, L2-normalised. Validation and export never read it.
  - It is trained by InfoNCE (T 0.1, weight 0.5) on rows whose anchor and negative share content. The target
    is the row's own content twin; same-speaker positives act as content negatives.
  - A leak penalty (weight 0.05) sums squared batch correlations between each content dim and all style
    dims, with the content side detached. Its noise floor is 256 / 96 ≈ 2.7.
- **chead against hinge:** +0.018 ± 0.068 held-out top-1, paired over 20 speakers. That is within noise.

### Trajectory at the stage boundaries (held-out)

| run | step | top-1 | gap ovp | cgap ovp | gap topic | gap real | rank |
|---|---|---|---|---|---|---|---|
| center | 1050 | 0.112 | +0.099 | +0.073 | +0.025 | +0.033 | 3.5 |
| center | 2100 | 0.102 | +0.185 | +0.086 | +0.139 | +0.093 | 3.1 |
| center | 3000 | 0.108 | +0.189 | +0.040 | +0.133 | +0.118 | 3.7 |
| center_infonce | 1050 | 0.236 | −0.056 | −0.234 | +0.008 | +0.036 | 32.9 |
| center_infonce | 2100 | 0.243 | −0.177 | −0.284 | +0.030 | +0.134 | 32.0 |
| center_infonce | 3000 | 0.232 | −0.262 | −0.296 | +0.017 | +0.146 | 33.2 |
| hinge | 1050 | 0.196 | +0.029 | −0.085 | +0.023 | +0.054 | 12.3 |
| hinge | 2100 | 0.187 | +0.003 | −0.111 | +0.072 | +0.123 | 10.6 |
| hinge | 3000 | 0.176 | −0.134 | −0.178 | +0.093 | +0.185 | 11.1 |
| hinge_chead | 1050 | 0.219 | +0.002 | −0.186 | +0.011 | +0.038 | 18.6 |
| hinge_chead | 2100 | 0.203 | −0.038 | −0.188 | +0.070 | +0.139 | 17.1 |
| hinge_chead | 3000 | 0.194 | −0.178 | −0.230 | +0.069 | +0.199 | 18.8 |

### Raw cosines for origin_vs_para (held-out): cos(anchor, same-speaker line) / cos(anchor, its paraphrase)

| run | step 1050 | step 2100 | step 3000 |
|---|---|---|---|
| center | 0.92 / 0.82 | 0.69 / 0.51 | 0.56 / 0.37 |
| center_infonce | 0.81 / 0.87 | 0.43 / 0.61 | 0.26 / 0.52 |
| hinge | 0.86 / 0.83 | 0.64 / 0.64 | 0.32 / 0.45 |
| hinge_chead | 0.89 / 0.89 | 0.57 / 0.61 | 0.29 / 0.47 |

length_course_genshin, whose positive contains the anchor's own text, ends at 0.87–0.89 positive cosine
against 0.14–0.18 negative in every InfoNCE arm, with a gap of about +0.71. It is the most separated rule
by far.

### Content head diagnostics (training logs, averaged per 300 steps)

| steps | 0–300 | 900–1200 | 1800–2100 | 2700–3000 |
|---|---|---|---|---|
| content accuracy (own twin top-1) | 0.83 | 0.98 | 0.97 | 0.94 |
| cos(anchor, content twin) | 0.90 | 0.95 | 0.95 | 0.94 |
| cos(anchor, same-speaker line) | 0.21 | 0.14 | 0.10 | 0.13 |
| style/content leak (noise floor 2.67) | 9.66 | 3.74 | 3.34 | 3.23 |
| content pairs per micro-batch | 4.4 | 5.2 | 5.2 | 4.1 |

The head learns content easily, and the linear leak penalty is driven down to near chance. The style
embedding still prefers the paraphrase.

## 6. What we think is happening

1. **InfoNCE spreads the space, and what survives the spreading is content similarity.**
   - In every arm, cosines shrink as training goes on: the space moves from a narrow cone towards uniformity.
   - Pairs whose only link is speaker identity fall fastest. For origin_vs_para, cos(anchor, same-speaker
     line) drops 0.86 → 0.32.
   - Pairs that share content keep most of their similarity: the paraphrase only drops 0.83 → 0.45.
   - So the ordering flips. The plain hinge model never spreads (rank 3.7), so it never flips.
2. **The leak is not a late event. It shows up early once you remove anisotropy.**
   - The raw gap at step 1050 for the hinge arm (+0.029) looked healthy, but its cgap was already −0.085.
     For center_infonce it was −0.234 from the first snapshot.
   - The "late collapse" in the raw gap is mostly the space spreading further. That removes the shared
     direction that was propping the raw gap up.
   - The hinge slows the slide but doesn't reverse it. It acts on only about 7% (ovp) + 7% (original_push)
     of rows, with margin 0.18, against an InfoNCE term that acts on every row.
3. **Only about 14% of rows carry a "same content, different style" negative at all.** In an InfoNCE
   denominator of about 95 candidates, a row's own paraphrase is 1 term, and most rows have no content twin.
   Nothing in the objective asks most anchors to ignore content.
4. **Some positives reward content matching.**
   - length_course positives contain the anchor text: overlap 0.48 and 0.80, repeated about 65×.
   - Under a hinge they saturate quickly and stop contributing. Under InfoNCE at T 0.05 they keep
     contributing gradient, teaching "overlapping text → closest candidate".
   - This is a hypothesis we have not isolated yet.
5. **The stage-3 curriculum may add to it.** topic_matched rises to 50% of rows, and its 14k triplets are
   repeated about 14×. Its own gap rises in stage 3 (hinge: 0.072 → 0.093) while origin_vs_para falls.
   But plain `center` saw the same curriculum and improved on both, so stage 3 is at most an amplifier.
6. **Why the content head didn't help:**
   - The leak penalty only removes *linear* correlation between the style dims and an 8-dim head that already
     solves its task.
   - Content can be carried in style-space directions that the head never needs. CURRICULUM.md had already
     found that low-variance directions carry content: whitening flips the origin_vs_para gap.
   - Getting the head to near-floor correlation says little about whether the style vector ranks a
     paraphrase above a same-speaker line.
   - The head did raise whitened top-1 (0.263 → 0.353) and rank (11 → 19). That suggests it soaks up some
     content variance in the shared trunk, but not enough to fix the ordering.

## 7. Caveats

- One seed per arm.
- Held-out retrieval is over 20 speakers. A paired difference needs about 0.07 to be called.
- Raw gaps and cgaps disagree in sign for the InfoNCE arms. We should decide which one is the success criterion:
  cgap is the stricter of the two.
- The origin_vs_para negative is an LLM paraphrase, so part of what we call "style" may be "human text vs LLM
  text". The paraphrase also keeps names and rare words.
- Validation texts are truncated at 120 chars for the overlap table only. Training uses `max_length` 192.

## 8. Candidate next experiments (our ranking)

1. **Give every InfoNCE anchor a content twin.**
   - Genshin has a paraphrase for about 262k of 264k lines. Add each anchor's paraphrase (and, for
     paratarget, its source text) as an extra hard-negative candidate in the denominator for every row, not
     just origin_vs_para rows.
   - Optionally upweight those terms (the hard-negative weighting of Robinson et al. 2021).
   - This puts the content-invariance pressure on the same footing as the spreading pressure.
2. **Take content-shortcut positives out of the InfoNCE term.** Keep length_course (and trait) rows
   hinge-only, or drop the anchor's own text from the length_course positive. This is cheap and isolates
   hypothesis 4.
3. **Schedule InfoNCE** (the question raised in review):
   - (a) InfoNCE on for stage 1 only, then anneal to hinge-only. Does the spread survive while the ordering
     recovers?
   - (b) Hinge only in stage 1, then turn InfoNCE on.
   - (c) Anneal T from 0.05 up to 0.1–0.2.
   - Because the cgap is already negative at step 1050, we expect scheduling to move the trade-off point
     rather than remove the trade-off. It is still worth one cheap arm, (a), since it costs nothing to code.
4. **Use a stronger, nonlinear adversary instead of linear decorrelation.** For example, a gradient-reversal
   discriminator that must tell "paraphrase of x" from "same-speaker line" using only the style embedding.
5. **Choose checkpoints on the gaps as well as retrieval.** Pick the checkpoint by the held-out cgap and
   top-1 together, not by the final step.
6. **Lower the stage-3 topic_matched share or its repetition.**

## 9. Questions for the reviewer

- Is there a better standard recipe than ours for "spread the space but stay invariant to a known nuisance
  pair"? For example, explicit invariance positives, VICReg-style invariance terms, or a debiased contrastive
  loss.
- Should the paraphrase be treated as a negative (different style) at all? Or should we generate
  same-content rewrites *in the anchor's own style* and use those as positives?
- Is cgap the right primary criterion, or should we judge content leakage with retrieval-style probes, such
  as "rank of own paraphrase among same-speaker lines"?
- Given InfoNCE alone already doubles held-out retrieval, would the reviewer accept some content leakage
  and filter it out downstream? Or is it disqualifying for a style embedding?

Source files: `outputs/ablations/infonce_{a,b}_1gpu/compare.md` and `compare.json`,
`outputs/runs/*/validation*/per_rule.csv`, `configs/ablations/infonce_{a,b}_1gpu.yaml`,
`configs/remote_5090x2.yaml`, `src/losses.py` and `src/train.py`.
