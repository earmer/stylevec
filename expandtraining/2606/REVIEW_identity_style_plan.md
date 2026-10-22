# stylevec: results so far and the identity/style plan, for external review

This is a self-contained summary of one working conversation, written to be sent to another model for review.
Sections 1–3 are facts and measured results. Sections 4–8 are plans that have not been run. Section 9 lists the
questions we most want a second opinion on. Section 10 is a chronological log of the conversation.

---

## 0. How to read this

- **Goal of the project:** a text embedding that captures **who** wrote a text, whether a human author, a
  character or speaker, or an LLM, across topics and genres, and that is **explainable**: some dimensions should
  have assigned, named meanings.
- **Key distinction the owner insists on:** *identity ≠ style*. An author can change style (verbose → concise,
  formal → casual, poem → chat) but can hardly change identity. Style should be nameable and controllable;
  identity is what stays invariant.
- **Hard constraint for every new idea:** it must not hurt the model's ability to identify **human** writers.

---

## 1. Background

### Model

- `QwenStyleEncoder` is built in this order:
  1. Qwen3-0.6B with LoRA;
  2. mean pooling;
  3. optional running-mean centering;
  4. a 1024→256 linear projection with bias;
  5. L2 normalisation.
- **Loss:** a triplet hinge on cosine distance (1 − cos) with per-rule margins, and Matryoshka prefixes.
- **Curriculum:** "staged". The share of hard negatives rises at 35% and 70% of training (steps 1050 and 2100 of
  3000).
- **Hardware:** one RTX 5090, 3000 steps, about 70 minutes per run.

### Data

- Triplets come from several *rules*, each pairing an anchor with a positive and a mined negative:
  - same game-character speaker across scenes;
  - origin against paraphrase;
  - MSynthSTEL controlled-style pairs;
  - trait-positive against trait-negative;
  - length-course negatives;
  - topic-matched negatives.
- The current dataset is d-cfa298ce2f92, with 120,811 training triplets. Most of it is Simplified Chinese: game
  dialogue (Genshin-style characters), plus the MSynthSTEL zh-hans controlled-style rows.
- Validation:
  - retrieval of 20 held-out speakers;
  - "gaps": same-speaker similarity minus different-speaker similarity, under topic-matched, origin-vs-paraphrase
    and real-speaker conditions;
  - a provenance probe (lower is better);
  - a within-speaker length R² probe (lower is better);
  - effective rank.

### Glossary

| term | meaning |
|---|---|
| heldout_top1 | top-1 speaker retrieval on held-out speakers (chance about 0.05) |
| whitened_top1 | the same, after PCA-whitening fitted on seen rows; shows information present in low-variance directions |
| gap: X | mean same-speaker cosine minus different-speaker cosine under condition X; should be positive |
| origin_vs_para gap | whether the model keeps the speaker after paraphrasing; can be inflated by "which generator wrote this" |
| provenance_acc | a linear probe predicting the generator or source of a row; high means the embedding encodes provenance |
| length_r2_speaker | R² of predicting log length from the embedding *within* a speaker; high means length leaks in |
| effective rank | exp(entropy of normalised covariance eigenvalues) |

---

## 2. Experiment history

### The cone problem

Without centering, every embedding sat in a narrow cone. Training cosines stayed at 0.85–0.96, triplet
violations at 0.6–0.95, and gradient norm rose to 5.4. A shared offset of norm about 135, against a pre-norm output
of about 45, dominated.

### The fix: `model.center: running`

An EMA mean of the pooled hidden states (momentum 0.01) is subtracted before the projection. The buffers are saved
with the weights. Committed in 2a595cc.

### spread_a_1gpu ablation (seed 42, same data)

| arm | change | outcome |
|---|---|---|
| control | none | stayed in the cone |
| decorr | correlation decorrelation, weight 0.02 | escaped until step ~950, then blew up (violations 0.98); stopped |
| center | running-mean centering | escaped and stayed out |
| center_decorr | center + decorr | done; no better than center |

Final validation:

| metric | control | center | center_decorr |
|---|---|---|---|
| heldout_top1 | 0.095 | **0.108** | 0.100 |
| heldout_top5 | 0.393 | 0.410 | 0.411 |
| whitened_top1 | 0.194 | **0.250** | 0.210 |
| provenance_acc ↓ | **0.669** | 0.696 | 0.693 |
| gap: topic_matched | 0.059 | **0.133** | 0.103 |
| gap: origin_vs_para | 0.012 | **0.189** | 0.071 |
| gap: same_speaker_real | 0.025 | 0.118 | **0.172** |
| effective rank | 5.6 | 3.7 | 3.7 |
| length_r2_speaker ↓ | **0.29** | 0.77 | 0.79 |

- **Speaker-paired comparison** over the 20 held-out speakers:
  - center − control: +0.016, SE 0.039. Within noise; center wins on 11 speakers and loses on 7.
  - center_decorr − center: −0.011, SE 0.010.
- **Trajectory:** every arm peaks at step 1050, when the hard share rises, and then slips. The gaps keep growing
  after that point even though retrieval falls.
- **Reading:**
  - keep centering, and drop decorrelation;
  - centering makes length leak much more (within-speaker R² 0.29 → 0.77);
  - the late-training retrieval drop is still unexplained.

---

## 3. Spectrum analysis of the `center` embeddings (no training)

The analysis used the 20,688 × 256 validation embeddings.

### Spectrum

| PC | variance share | what it tracks |
|---|---|---|
| PC1 | 51% | partly speaker (speaker η² 0.35) |
| PC2 | 36% | MSynthSTEL/trait rows (≈ −0.98) against dialogue rows (≈ +0.2); rule η² 0.47 |
| PC3 | 4% | — |

The effective rank is 3.65 over all rows and 2.70 on dialogue rows only.

### Removing length and provenance by regression

| variant | effective rank (all / dialogue) | heldout_top1 | origin_vs_para gap |
|---|---|---|---|
| raw | 3.65 / 2.70 | 0.108 | +0.154 |
| − log length | 4.22 / 3.51 | 0.109 | +0.042 |
| − provenance | 4.23 / 3.29 | 0.106 | −0.017 |
| − both | 4.21 / 3.27 | 0.093 | −0.026 |

After removing both, length R² falls from 0.79 to 0.19 and provenance accuracy from 0.696 to 0.570. The
topic_matched gap barely moves (+0.124 → +0.122).

### Soft whitening

The transform is (Σ + λ·mean(eig)·I)^(−1/2), fitted on seen rows.

| λ | effective rank | heldout_top1 | cross-scene top1 |
|---|---|---|---|
| raw | 3.65 | 0.108 | 0.098 |
| 1.0 | 63 | 0.177 | — |
| 0.1 | 152 | 0.226 | 0.201 |
| 0.01 | 226 | **0.279** | **0.242** |

At λ = 0.01, top-5 is 0.640, but both content gaps turn negative (origin_vs_para −0.337, topic −0.010).

### Conclusions

- Speaker information is present but buried in low-variance directions, mixed with content. The ceiling is about
  0.28 top-1 against 0.108 raw.
- Most of the origin_vs_para gap is **provenance** (which generator wrote the paraphrase), not style.
- The hinge loss stops pushing once its margin is met. That suggests a loss with a full in-batch softmax
  (InfoNCE).

---

## 4. Issue #1: center + in-batch InfoNCE; Matryoshka trained at [64] only (implemented, not yet run)

Tracked at <https://github.com/earmer/stylevec/issues/1>. The change is in the working tree, uncommitted, with 115
tests passing.

### Loss

- `train.loss: infonce` turns it on. The temperature is 0.05, and the logits are computed in float32.
- For anchor i, the candidates are every positive in the batch (its own is the target), every other anchor and
  every mined negative.
- **False-negative masking:** another row's candidate with the anchor's own speaker label is *removed* from the
  denominator, not pushed away. The anchor's own mined negative is always kept.
- Rows without a speaker (neutral paraphrases, negative-trait texts) are never masked.
- An optional `infonce_triplet_weight` adds the hinge on top.

### Matryoshka

- `regularization.matryoshka_dims` now controls training only: `[64]` trains the 64 prefix plus the full 256.
- `validation.matryoshka_dims` keeps reporting 32/64/128/256.
- Each prefix is re-normalised before the loss.

### Spec `infonce_a_1gpu`

- Arms: `center` (triplet), `center_infonce` and `center_infonce_m64`.
- Guard: the origin_vs_para and topic_matched gaps must stay positive, so the gain isn't a content leak.
- Target: beat 0.108 held-out top-1 and move toward the whitened ceiling of 0.279.

---

## 5. Issue #2: expanding the data with outputs from many LLMs

Tracked at <https://github.com/earmer/stylevec/issues/2>.

### Why

Today a speaker's style is confounded with the generator that produced its paraphrases (provenance accuracy
about 0.70). If every speaker or author gets rows from **every** generator, the generator no longer predicts the
identity.

### Methods: what we send and what we get back

| # | method | input sent | output | what it gives training |
|---|---|---|---|---|
| 1 | paraphrase in character | a real line + 3–5 example lines from the speaker | new wording, same voice | positives from every generator |
| 2 | voice swap | a line from speaker A + example lines from speaker B | A's content in B's voice | content-matched hard negatives |
| 3 | neutralise | a real line | a plain rewrite, meaning kept | a content-only anchor |
| 4 | new lines in character | persona card + examples + a situation | new lines | more positives, more speakers |
| 5 | length-controlled rewrite | a line + a target length (×0.5, ×2) | same voice, different length | attacks the length leak (R² 0.77) |
| 6 | single-factor edit | a line + "change only punctuation, register, language, emoji or sentence count" | a pair differing in one known factor | supervision for named factor dimensions |
| 7 | back-translation | zh→en→zh, zh→ja→zh, across models | content-preserving noise | drift from the generator, not the speaker |
| 8 | same prompt, many models | one situation, no speaker | N "default voices" | a pure provenance set, to measure and penalise |

### Models, mixed on purpose

- **Chinese-native APIs:** DeepSeek, Qwen-Max, Kimi, GLM, Doubao and MiniMax.
- **Western APIs:** Claude, GPT and Gemini. Their punctuation and phrasing habits differ.
- **Open models run locally with vLLM:** Qwen3 at several sizes, GLM-4, Yi, InternLM, Gemma and Llama.
- **Caution:** the backbone is Qwen3, so Qwen-generated text may be "familiar" to it. Keep Qwen to a minority, and
  hold it out once to test this.

### Content types

- **Beyond dialogue** (the owner's addition): casual chat and documentary or expository writing, plus the
  sources in `automating.md` (see 6.2).
- **Spoken-style text:**
  - dialogue, deliberately short (<15 characters) and long (>80);
  - monologue and narration;
  - multi-turn replies;
  - persona and trait descriptions;
  - in-character answers to everyday prompts.
- **Controlled variants:** emoji and kaomoji, full-width against half-width punctuation, classical against
  colloquial register, and code-mixing.

### Pipeline

1. **Prompt bank:** example lines and a persona card for each speaker, plus situation prompts.
2. **Fan-out:** method × model × temperature. Every row keeps `generator, method, factor_changed, temperature,
   source_row_id, prompt_id, language`.
3. **Filters:**
   - an LLM judge that is not the generator scores in-character fidelity and meaning preservation;
   - dedupe on n-grams and embeddings;
   - drop refusals and meta-text.
4. **Balance:** each identity gets roughly equal rows per generator family.
5. **Splits:**
   - held-out speakers (as now);
   - one entire held-out generator;
   - held-out prompt templates.

### Pilot

20 speakers × 30 source lines × methods {1, 2, 3, 5, 6} × 4 models, about 12k rows. Then one arm: center against
center plus the pilot data.

---

## 6. AI-generated-text detection and model identity

### 6.1 The owner's idea

Add dimensions that say whether a text is AI-generated, plus dimensions for the *specific* model. The owner has
seen a detector that recognises model-written classical Chinese poems and essays precisely. That suggests a
model's identity survives genre: **genre is part of style, not of the underlying identity**, and cross-genre pairs
are among the hardest training examples.

### 6.2 Data sources, by role (many already listed in `automating.md`)

| role | sources |
|---|---|
| clean human text (pre-2022, so no LLM contamination) | chinese-poetry (全唐诗/宋词), Wikisource, Project Gutenberg, Blog Authorship, Reddit and Weibo before 2022, CrossNews |
| human casual chat, in the same setting as the AI answers | **user turns** in WildChat, LMSYS-Chat-1M and Arena; filter out text the user pasted from an AI by n-gram matching against the assistant turns |
| AI, same prompt across models | Arena Human Preference 140k, Arena-Hard v2 answer files |
| AI, one model at a time | TeichAI sets (Claude Sonnet/Opus 4.6, Gemini 3 Flash, DeepSeek V4 Flash), EQ-Bench creative writing, RAID, M4GT-Bench, MAGE |
| nonfiction and documentary | our own generation from a shared factual seed (narration script, news write-up, encyclopedia entry), with human counterparts from Wikisource and CrossNews |
| controlled style | MSynthSTEL, SynthSTEL, and our own rewrite sets (section 5) |

Known label caveats:
- the TeichAI identity labels are third-party and unverified;
- CNNovel125K has about one book per author (2,769 authors in 2,799 rows), so it cannot supply identity positives;
- Reddit Cross-Topic has exactly 5 texts per author.

### 6.3 Hard examples for the sampler

- **Positives:** same source, different genre. For example, one model writing a regulated poem, a casual chat reply
  and an expository essay.
- **Hardest negatives:** different source, same genre and same prompt (from Arena).
- **Human against AI, matched:** a real classical poem against one written by each model to the same title, tune
  pattern and rhyme. The genre fixes the format, so the easy tells vanish.
- **AI imitating a human author** against that author's real text: a negative for identity even when the surface
  style matches.
- **"Humanised" AI text,** prompted to avoid AI tells, for robustness.
- **AI-edited human text** as a third class, not as AI. It is the common real-world case.

### 6.4 Shortcuts to block

Ordered from the most to the least dangerous:
1. **Length:** AI answers are long and human chat is short, so sample length-matched.
2. **Format:** markdown, lists, stock openers ("当然！", "Certainly!") and closers ("总之", "In summary"),
   em-dashes. Strip them for training, and evaluate on both the stripped and the raw versions.
3. **The first and last sentences:** trim them on part of the data.
4. **Topic:** coding and maths are mostly AI-side and chit-chat mostly human-side, so balance by topic cluster.

### 6.5 Evaluation

- held-out model, open set: AI-ness should still fire on a model never seen in training;
- held-out genre;
- calibration of the AI-ness score at a fixed false-positive rate on pre-2022 human text.

---

## 7. Identity against style: layout and dimension plan

### 7.1 Principle

- **Style** is what an author can change at will. It is measurable, so dimensions can be *named in advance* and
  given labels.
- **Identity** is what stays invariant while style changes. It is defined by invariance, so it cannot be
  pre-named; it is explained *after* training (7.6).

### 7.2 Layout: separate heads on the shared pooled feature

Ranges inside one vector won't work: Matryoshka truncation would cut through named ranges.

```
pooled ─┬─ I: identity     256-d, L2-normalised, Matryoshka (as now)  → retrieval
        ├─ L: language/script  categorical + mixing ratios           → see section 8
        ├─ S: style        ~16 named scalars, not normalised          → explainable factors
        ├─ T: typography   ~10 named scalars (punctuation/orthography habits; section 8.3)
        ├─ G: genre        8 classes: poem, essay/prose, narration, dialogue, chat, social post,
        │                  Q&A answer, documentary/expository
        └─ P: provenance   human / AI / AI-edited human
```

- **Each LLM is one more author in I.** DeepSeek is an author in the same way a game character is, so no separate
  model-identity block is needed.
- **P only holds the coarse "is this AI".**

### 7.3 Two kinds of supervision for S and T

- **Absolute, computed by rules.** Regression from each dimension to a computed feature, z-scored per language
  (the full table is in 8.2). Exact, cheap and reproducible.
- **Relative, from controlled rewrites.** For factors that only make sense relative to the content, such as
  verbosity, a concise and a verbose version of the same content give a ranking loss,
  dim(verbose) > dim(concise), plus an invariance loss on the other dimensions. **LLM rewrites train only S, T
  and L, never I.** A rewrite carries the generator's identity, not the author's.

### 7.4 Protecting human identity

1. **Gate.** Every new head is an ablation arm and must stay within −0.01 of the baseline on held-out human
   speaker top-1, cross-scene top-1 and author retrieval. An arm that fails doesn't go in.
2. **Cross-style positives for I** come only from real same-author pairs across registers:
   - game speakers across scenes;
   - CrossNews, formal article against social post;
   - Weibo and Reddit, short against long posts.
3. **Decorrelation within an author, not overall.**
   - I *may* correlate with S across authors: an author's typical verbosity is a fair part of who they are.
   - I must *not* move when the same author changes style.
   - So the penalty applies to I's dependence on within-author deviations of S, T and L. This generalises the
     existing `length_r2_speaker` probe to every factor.
4. **Gradual introduction.**
   - Start the heads at low weight, or with stop-gradient into the backbone.
   - Remove the stop-gradient only if the gate holds.

### 7.5 Arms, in order

Each arm must pass the gate before the next is built on it.

1. `center_infonce` (issue #1)
2. + S/T/L/G heads with stop-gradient
3. + the same heads trained jointly, plus the within-author I⊥S penalty
4. + P head and the multi-LLM data

### 7.6 Explaining identity after training

- **Habit card for each author.** List the S/T factors with much lower within-author variance than the
  population's: the author's stable signatures, e.g. "always uses ……, never ！, always a space between CJK and
  Latin".
- **Sparse autoencoder on I.** Its atoms are named afterwards from their top-activating texts, e.g. "sentence-final
  呐", "opens replies with 嗯", "lowercase sentence starts".
- **Leak probes:**
  - I → S/T/L within an author should have R² near 0;
  - P → speaker should be at chance on human text.
- **Caveat:** a named dimension is guaranteed to *encode* its factor, not to be the *only* place that factor
  appears. The leak probes are the check.

---

## 8. Multilingual expansion (the owner's addition)

The plan in section 7 was written for Simplified Chinese. The owner wants it multilingual: Chinese (Simplified and
Traditional), English and other Latin-script languages, German, Russian, Japanese and so on. Punctuation should
cover every one of their conventions, not only Chinese ones.

### 8.1 Language is a factor, not an identity

- Language would otherwise dominate the embedding, the way rule type dominates PC2 today. So it gets its own head
  (L), and I must be **invariant to language within an author**.
- **L head:**
  - language ID: zh-Hans, zh-Hant, ja, en, de, ru, fr, es and others;
  - script shares: Han, kana, Latin, Cyrillic and others;
  - code-mixing ratio;
  - a translated-text flag.
- **Data for cross-language identity:**
  - **LLMs are the natural multilingual authors.** The same model answering in zh, en, de, ru and ja gives
    same-identity positives across languages. Arena and WildChat are already multilingual (WildChat is
    especially rich in Russian).
  - **Human authors who write in two languages are rare.** Candidates:
    - multilingual Reddit or blog users;
    - authors who translated their own work.

    Human identity will mostly be trained within a language at first.
- **Translations carry the translator's identity**, not the author's. Label translator and version, as
  `automating.md` already says for Gutenberg.

### 8.2 Style dimensions (S): definitions shared across languages, extractors per language

Each factor has one meaning everywhere. The extractor and lexicon differ by language, and labels are z-scored
within a language, so "high" means high *for that language*.

| dim | factor | how it is measured, per language |
|---|---|---|
| S0 | length | log of Qwen tokens (one measure across scripts) |
| S1 | sentence length | tokens per sentence; sentence split per language (。！？ / . ! ? / 。！？ for ja) |
| S2 | punctuation density | Unicode P* category per token |
| S3 | clause chaining | clause separators per sentence (，、 / , ; / 、) |
| S4 | questions | ？/? rate, plus question particles (吗/呢, か, ли) |
| S5 | exclamation | ！/! rate, plus Spanish ¡ |
| S6 | trailing off | …… / … / ... / 。。。 / —— |
| S7 | emoji and emoticons | emoji, kaomoji (顔文字), ASCII emoticons such as :) and xD |
| S8 | discourse particles and hedges | zh 呢/吧/啊/嘛; ja ね/よ/かな/っす; en like / you know / well; de halt / eben / ja / doch; ru ну / вот / же |
| S9 | first person | pronoun lexicons; for ja, *which* pronoun (私/僕/俺/あたし) is itself a strong character signal |
| S10 | addressing the reader | 你/您, あなた/君, you, du/Sie, ты/вы; the T–V choice is part of register |
| S11 | archaic or literary register | 文言 function words; Japanese 古文/文語; English thee/thou and archaic forms; Russian pre-reform or Church Slavonic forms |
| S12 | lexical diversity | MTLD on words; segmentation for zh/ja (jieba / fugashi), whitespace for the others |
| S13 | code mixing | share of tokens outside the text's main language or script |
| S14 | structure | lists, numbering, markdown, headings |
| S15 | verbosity | relative, from rewrite pairs (7.3) |

### 8.3 Typography and orthography habits (T): punctuation across languages

These are rule-measurable, and many are **very stable within an author**. They sit between style and identity, and
are prime material for the habit card.

| dim | habit | variants |
|---|---|---|
| T0 | quote style | zh “” / 「」; ja 「」『』; de „…“ / »…«; fr « … »; ru «…» / „…“; en "…" against curly “…” |
| T1 | space before punctuation | French-style `mot !` against `mot!` |
| T2 | spacing between CJK and Latin | 中文 English against 中文English (盘古之白) |
| T3 | full-width against half-width | ，。！？ against ,.!? inside CJK text |
| T4 | ellipsis form | …… / … / ... / .. / 。。。 |
| T5 | dash form | —— / — / – / -- ; Russian dialogue dashes |
| T6 | diacritics | German ä→ae; Russian ё against е; accent dropping in fr/es/pt |
| T7 | capitalisation | missing capitals at sentence start, ALL-CAPS emphasis, i against I; German noun capitals |
| T8 | sentence spacing | two spaces after a period, missing space after commas |
| T9 | script variant | zh-Hans against zh-Hant, and mixing the two; ja kanji/kana balance and katakana rate |

### 8.4 Multilingual data candidates (to verify before use)

- **English:** Blog Authorship, Reddit (REDDIT_comments, Cross-Topic), Gutenberg, CrossNews, SynthSTEL.
- **German, French, Russian and others:** Gutenberg (multilingual subset), Wikisource in each language, the
  Deutsches Textarchiv (de), Taiga (ru); all candidates to verify.
- **Japanese:** Aozora Bunko (public-domain authors with many works each, good for identity across works).
- **Multilingual LLM text:** Arena 140k, WildChat, LMSYS-Chat-1M, M4GT-Bench (multilingual detection).
- **Controlled style:** mStyleDistance / MSynthSTEL covers several languages, not only zh-hans.
- **Our own generation:** run section 5's methods in each target language, including method 7 (back-translation)
  as a language-only intervention, which gives positives where only L changes.

### 8.5 Multilingual evaluation

- **Gates per language:** human-identity top-1 must not drop in *any* language, not only on average.
- **Cross-language identity:** mainly for LLMs, e.g. answers by the same model in zh and in de should retrieve
  each other.
- **Language leak:** an L → identity probe on human authors writing in a single language should stay near chance.
- **Balance:** report every metric per language, so English volume doesn't hide a regression in Chinese.

---

## 9. Questions for the reviewer

1. Is "separate heads (I, L, S, T, G, P) on one pooled feature, with style supervision that is absolute (rules) or
   relative (rewrite pairs)" the right way to get explainable dimensions? Or would one vector with assigned
   coordinate blocks be better?
2. Is decorrelating identity from style *within an author* (rather than globally) a sound formulation of "identity
   is invariant to style changes"?
3. Should LLM-generated rewrites ever be used as identity positives, or only for style heads (our current rule)?
4. How should one avoid the AI-detection head learning length, format and topic shortcuts, beyond length-matched
   sampling and format stripping?
5. For human identity across languages, where would you find real multilingual same-author data?
6. Which named style or typography factors are missing, especially language-specific ones?
7. With a 0.6B backbone, would you expect the extra heads to cost human-identity accuracy? What else should we
   watch?

---

## 10. Conversation log (chronological, condensed)

The owner's words are quoted; the assistant's turns are summarised.

1. **Owner:** "i'm wondering if the non-centered version somehow is correct, because its effective rank goes up
   to 12 or more." Follow-ups asked how the centered and uncentered versions compare in scale, "the data scale i
   mean!"
   - **Assistant:** the uncentered model's higher rank comes from a broken, cone-shaped geometry (violations
     0.6–0.95). Centering removes an offset of norm about 135, against outputs of about 45.
2. **Owner:** "check it now and i'll go sleep after your verdict."
   - **Assistant:** the verdict favoured `center`. It escaped the cone and stayed out.
3. **Owner:** asked to pull the results, and pointed to the start-with-retry script and to "no card mode" (a
   GPU-less box).
   - **Assistant:** pulled the metrics, ran the speaker-paired compare, and wrote `RESULTS_spread_a.md`
     (section 2). Committed as 2a595cc.
4. **Owner:** "effective_rank. current about 4 is somehow low. any ideas?" Then: "that's not what we are talking
   about. what we are talking about, currently, is explainability. for example, expanding the dimentions that
   lang ~= 3 dim; length~= 2 dim; puncs ~= 3 dims; ... current model may can distinguish, but it cant split them
   well."
5. **Owner** pasted a 3-step plan:
   1. a local spectrum check;
   2. `center` + in-batch InfoNCE with same-speaker masking;
   3. Matryoshka trained at [64].
   - **Assistant:** ran step 1 (section 3).
6. **Owner:** asked for two brief GitHub issues.
   - **Assistant:** opened #1 (InfoNCE and Matryoshka) and #2 (data from different LLMs).
7. **Owner:** "Make a sub-agent to implement the first issue. For the second issue, list the methods, the models,
   and the content type… If we can, we send the data to them and let them paraphrase or so."
   - **Assistant:** a sub-agent implemented #1 (section 4). Brainstormed #2 (section 5).
8. **Owner:** wants to collect more than spoken text: the `automating.md` sources, Arena data, casual chats and
   documentary works. Also AI-detection dimensions and model-identity dimensions, noting that "this kind of
   character identity is robust across various types of literature… the genre is part of the style, but not the
   underlying hidden styles, and that might be a very hard triplet."
   - **Assistant:** section 6.
9. **Owner:** "we should keep in mind that this issue is on top of our current, entire goal… not hurting the
   model's ability to identify the human writers' identities and styles… the identity of the author is different
   from their style. An author may change style, like from verbose to concise, but can hardly change their
   identity. Could we map or plan the identity dimensions?"
   - **Assistant:** section 7. Style is named in advance, identity is defined by invariance, and human identity
     acts as a gate.
10. **Owner:** "the dimensions also contain factors like language… include punctuation: not just Chinese, but
    also English, Latin characters, German, Russian, Japanese… I thought the previous plan… was to focus on
    Simplified Chinese, so could you somehow expand it to multi-language?" Asked for this transcript to send to
    Gemini.
    - **Assistant:** section 8 and this document.
