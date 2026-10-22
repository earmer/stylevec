# Hongloumeng author-switch probe (2026-10-21)

Question: can the `center` model (spread_a_1gpu, seed 42), used with no further training, detect the change of
hand after chapter 80?

- **Script:** `scripts/hongloumeng_probe.py`.
- **Text:** Project Gutenberg #24264, traditional characters, Cheng-edition lineage.
- **Passages:** 6,851, each 60–297 characters.
- **Outputs:** `outputs/hongloumeng/` (gitignored).

## First run

A logistic regression is trained on chapters 1–60 against 91–120 and scores 61–90, which it never saw.

| method | where 61–90 switch to "late" |
|---|---|
| function-character frequencies (about 50 characters) | 80 → 81, sharp |
| base Qwen3-0.6B, length regressed out | 80 → 81 |
| trained `center`, length regressed out (256 and 64 dims) | 80 → 81 |

- **The trained model does not beat the function-character baseline** on this test.
- **After 81 the curve fluctuates,** with chapters 82, 87 and 89 back to "early", and settles only from 90.
- **Chapter 67 scores as "late"** (trained 0.70, base 0.75).
- **The best single split** lands at chapter 2 or 5 for every method, driven by the frame-story opening.
- **The last 40 chapters are more alike among themselves:** within 81–120 0.63, within 1–80 0.34, across 0.30.

## Agreement with existing Hongloumeng scholarship (the owner's note)

Several of these patterns agree with existing research, so the probe recovers known structure rather than noise:

- **The switch near 80/81** matches the traditional division: Cao Xueqin for 1–80, and the 程高 continuation for
  81–120.
- **The fluctuation in 81–90** fits the view that the early continuation chapters are not uniformly in the new
  hand. Some scholars argue they draw on drafts that survived, which the Cheng preface claims were collected.
- **Chapter 67:** the 庚辰本 manuscript lacks chapters 64 and 67, and the text of 67 in the Cheng editions is
  sometimes considered supplied or reworked.
- **81–120 is more uniform** than 1–80, consistent with one continuator's more even register.

Exact citations are still to be added and checked.

## Second run

Embeddings cached locally (float16 npy, `outputs/hongloumeng/results/cache/`) after a box run. All
analysis below runs from cache without the GPU.

### Placebo boundaries (b = 41–80)

For each candidate boundary b, logistic regression is trained on chapters [b−40, b−11] against
[b+11, b+40] and scored on the held-out window [b−10, b+10]. Rank 1 = strongest boundary among
the 39 tested (b = 41–79 provide the null; b = 80 is the test).

| method | AUC at b=80 | rank | z | step AUC | step z |
|---|---|---|---|---|---|
| trained center 256d (raw) | 0.945 | **1/39** | 2.05 | 0.683 | 2.79 |
| trained center 256d (len-corr) | 0.909 | **1/39** | 1.85 | 0.632 | 2.30 |
| trained center 64d (len-corr) | 0.900 | 3/39 | 1.68 | 0.619 | 2.68 |
| base Qwen3-0.6B (len-corr) | 0.918 | 2/39 | 2.01 | 0.755 | 2.60 |
| function chars (raw) | 0.836 | 3/39 | 1.39 | 0.341 | 0.92 |
| function chars (len-corr) | 0.700 | 22/39 | 0.12 | 0.250 | 0.45 |

**Correction (reviewed against `results.json`):**
- **Rank among all 39 overstates the result.** Neighbouring boundaries share most of their windows, and for
  b ≥ 70 the scored window [b−10, b+10] contains the real chapter-80 change, so those are not nulls. The AUC
  ramps up smoothly from 0.83 at b=77 to 0.91 at b=78–80; there is no sharp peak at 80.
- **Compared with the clean nulls only (b ≤ 69):**

| method | AUC at b=80 | max of clean nulls | mean of clean nulls |
|---|---|---|---|
| trained 256d (raw) | **0.95** | 0.83 | 0.68 |
| trained 256d (len-corr) | **0.91** | 0.83 | 0.66 |
| trained 64d (len-corr) | 0.90 | 0.92 | 0.68 |
| base Qwen3 (len-corr) | 0.92 | 0.92 | 0.68 |
| function chars (raw) | 0.84 | 0.87 | 0.62 |
| function chars (len-corr) | 0.70 | 0.87 | 0.66 |

**Finding:** only the trained 256-d model puts b = 80 above every clean null. Base Qwen3 ties its largest null,
and function characters do not separate 80 from the nulls. So on this test our training does add something over
the base model and over classic stylometry. Caveats: one book, and the ~28 nulls are strongly autocorrelated.
**Strengthens** the scholarly attribution at chapter 80.

### Change-point without introduction (skip ch 1–5)

All methods, length-corrected, find chapter 11 as the best single split (p = 0.003–0.030). The frame
narrative covers chapters 1–5; removing it shifts the dominant change-point away from chapter 2/5 but
still not to chapter 80. The next-strongest structural break in chapters 6–120 is around chapter 11–12,
coinciding with the start of 賈家 domestic life proper (Jia Rui episodes, etc.). Chapter 80 does not
emerge as the dominant unsupervised split even with the intro removed; it only surfaces when trained
explicitly as a binary classification boundary (the logistic probe and the placebo test).

**Correction: the "chapter 11" result is a bug.**
- `change_point_test(skip_first=5)` keeps chapters 1–5 in the first segment, only forbids splitting before
  chapter 6, and then adds 5 to the reported chapter.
- Re-run properly in `scripts/hongloumeng_style_vs_content.py`: chapters 1–5 are dropped and the reported chapter is the last one
  in the first segment. Length is corrected for all rows.

| method | chapters 1–120 | 6–120 | 12–120 |
|---|---|---|---|
| trained 256d | 22 (p 0.06) | 22 (p 0.14) | 18 (p 0.13) |
| trained 256d, names masked | 5 (p 0.14) | 22 (p 0.26) | 18 (p 0.06) |
| base Qwen3 | 5 (p 0.05) | 18 (p 0.23) | 18 (p 0.03) |
| function chars | 5 (p 0.00) | **80** (p 0.03) | 19 (p 0.02) |

### What the breaks at 5 and 18–22 are

Mean rates per segment, from rule features that use no content words:

| feature | 1–5 | 6–11 | 12–80 | 81–120 |
|---|---|---|---|---|
| classical 之乎者也矣焉哉 | **0.018** | 0.009 | 0.010 | 0.009 |
| vernacular 了的着呢麼罷 | **0.030** | 0.066 | 0.056 | 0.064 |
| dialogue markers | **0.013** | 0.023 | 0.025 | 0.025 |
| 5/7-character clauses | **0.019** | 0.017 | 0.016 | 0.014 |
| mean passage length (chars) | **171** | 119 | 127 | 120 |

- **The break after 5 is register.** The frame chapters have twice the classical function words, half the
  vernacular and dialogue, longer passages and more verse (the 判词 and 红楼梦曲 of chapter 5). Function
  characters, the method lightest on content, find it most strongly (p = 0.000). This is style in the
  genre/register sense, by the *same* author, so it is not writer identity.
- **The break after 18–22 is most likely plot phase, i.e. content.** It falls where 大观园 is built
  (chapters 17–18) and the move into it (chapter 23). It survives name masking but is not significant
  (p 0.06–0.26), and only the neural models rank it first.

### Is the chapter-80 signal style or content?

Same placebo test as above, at chapter level. "− content" removes 10 principal components of content-character
frequencies (the 600 most frequent characters, excluding function characters); "− register" removes the five
features above. z is against the clean nulls (b ≤ 69).

| variant | trained 256d | trained, masked | base Qwen3 | function chars |
|---|---|---|---|---|
| as is | 0.92 (z +1.9) | 0.97 (+2.5) | 0.92 (+2.3) | 0.95 (+2.6) |
| − register | 0.86 (+1.6) | 0.87 (+2.7) | 0.85 (+1.4) | 0.75 (+0.7) |
| − content | 0.64 (+0.1) | 0.35 (−1.8) | 0.13 (−3.2) | 0.13 (−3.1) |
| − register − content | 0.47 (−0.8) | 0.21 (−2.7) | 0.15 (−2.3) | 0.16 (−2.2) |

Content characters on their own: AUC 1.00 at b = 80 (z +2.9), the strongest signal of all.

- **The 80 signal survives removing register,** and the masked trained model keeps the clearest margin (z +2.7).
- **It does not survive removing content,** for any method. AUCs below 0.5 mean the correction overshoots; it is
  not a reversed signal.
- **So this design cannot separate writer style from content at chapter 80.** The plot and vocabulary change there
  (decline of the house, 黛玉's death), and every method, ours included, is at least partly reading that change.
  "Content characters" also carry lexical preference, which is itself authorial, so removing them also removes
  some real style. The test is inconclusive, not negative.
- **A cleaner test:** embed a function-word skeleton (content words and names masked) or content-matched passages,
  e.g. one character's speech before and after 80. That needs the weights on the box once more.

### Name masking

Approximately 60 main character names (traditional + simplified) replaced with 某.

**Correction:** "flip chapter" means the first chapter in 61–90 with P ≥ 0.5, so "ch 66" only marks the
chapter-67 spike. It does not mean the boundary moved. Mean P(late), all length-corrected:

| method | 61–80 unmasked → masked | 81–90 unmasked → masked | ch 67 unmasked → masked |
|---|---|---|---|
| trained 256d | 0.04 → 0.05 | 0.64 → 0.70 | 0.69 → 0.82 |
| base Qwen3 | 0.05 → 0.06 | 0.72 → 0.78 | 0.44 → 0.88 |
| function chars | 0.08 → 0.07 | 0.32 → 0.33 | 0.01 → 0.01 |

**Finding:** masking the names does not weaken the break at 80 for any method; the neural methods separate
slightly better with names masked. Chapter 67's "late" score *rises* when names are masked, so it isn't
driven by which characters appear. Function characters separate the halves only weakly (0.08 against 0.32).
**Strengthens** both the chapter-80 and the chapter-67 agreement.

### Dialogue share

Fraction of passages containing a dialogue marker (道：/說：/"/「) is nearly identical in both halves:
ch 1–80: **0.896**, ch 81–120: **0.902**. Dialogue density is not a confound. **Neutral** for the
scholarly comparison (does not distinguish the two periods).

### Verse detector

34 of 6,851 passages (0.5%) tagged as verse (5/7-char CJK lines or balanced couplets). Removing them
changes no result. The detector misses embedded poems that span multiple lines with mixed prose; a more
complete detector is needed for a proper verse-vs-prose decomposition. **Neutral.**

### Passage-level bootstrap

Bootstrapping over individual passages rather than chapters produces near-uninformative confidence
intervals (trained 256d, len-corr: within 1–80 = −0.009 [−0.94, 0.96]; within 81–120 = 0.101
[−0.94, 0.97]; cross = −0.037 [−0.94, 0.97]). Each 60–150-char passage carries too little style
signal for passage-to-passage cosines to be meaningful. The chapter-level aggregation used in the
first run is necessary. **Neutral.**

### Summary of second-run impact on scholarship agreement

| finding | impact |
|---|---|
| Placebo: only trained 256d puts b=80 above every clean null (0.91–0.95 against ≤0.83) | **strengthens**; the only result where training beats base Qwen3 and stylometry |
| Function chars do not separate b=80 from the nulls | the raw Phase 1 flip was partly a length effect |
| Name masking: the break at 80 holds for every method; ch 67 rises | **strengthens** (80 and 67) |
| Break after 5 = register (classical frame, verse); after 18–22 = plot phase; "11" was a bug | **neutral**: style or content, not identity |
| The 80 signal disappears for every method once content is removed; content alone gives AUC 1.00 | **weakens** the claim that it is *style*: the break is real but content-entangled |
| Dialogue share identical across halves | **neutral** (not a confound) |
| Verse <0.5%; removal changes nothing | **neutral** |
| Passage-level bootstrap uninformative | **neutral** (confirms chapter aggregation is right) |
