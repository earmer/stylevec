# Trait predictor: sealed confirmation on fresh human ratings

Written after the 95 were scored, before any pair below was rated or scored. The 95 are now the development benchmark, chosen on
freely; this set decides.

Why a new set: three arms were compared on the 95 and on small-dev, and every design choice came after seeing Jev do well on the 95.

## Arms (frozen, seed 42, step 3000, under expandtraining/2606/outputs/trait_predictor/)

- Primary: lora_r8_delta1.0_s42_attention. It led the 95 (within .590, small .526, overall .702, minus edit +.373).
- Secondary: lora_r8_delta1.0_s42 (.555, .455, .682, +.338) and
  frozen_delta1.0_s42_attention_layers6-12-18-24-28 (.534, .481, .698, +.317).
- Baseline: edit size (1 - difflib ratio on characters). The score of a pair is the mean absolute predicted trait difference.

No arm is retrained, tuned or added after this file is committed.

## Items

Frozen in data/human_ratings/voice_change_confirm_key.jsonl, built by rate_voice_server.py build-confirm, 125 pairs,
SHA-256 1545845489baa505f79946d1eb5e4e261cf904965517c9b647f567d721482ab1.

- fresh, 80: 60 small (edit < .15), 12 mid, 4 large, 4 rewrite. Random, no LLM tag used. Speakers have at least 20 lines, none of
  their lines is in the predictor's train or dev, none is a rated speaker, at most 3 pairs per speaker. Not in small-dev.
- retest_95, 18: 12 small and 6 other pairs from the 95, picked by edit size and never by rating. They measure the rater's self-agreement.
- retest_gemini, 27: every pair first rated with Gemini's score on screen, now rated without it.

A character known under a second name is not checked for, beyond the one alias already found, so a few fresh speakers may be a
trained character under another name. Re-rated speakers are familiar ones, so a re-rate can be guessed from the speaker.

## Rating

Blind to model and LLM scores: the server runs without --show-llm and sends no edit size or block. Change highlighting starts on, as
the rater chose; the views log records it. Ratings go to data/human_ratings/voice_change_confirm_ratings.jsonl, so no first rating
is overwritten; the latest row per id wins within that log. Broken pairs and pairs without a voice rating are excluded and counted.

## Scoring (only the fresh block decides)

pairwise_voice_baseline.report, as on the 95: within is the Spearman inside each edit bin x batch cell of at least 5 items, weighted
by items; 4,000 bootstrap resamples within cells, seed 20261105. Also small-edit Spearman and overall Spearman.

Pass for the primary arm: within minus edit size at least +.25, and its 99% interval excludes 0. 80 pairs cannot put the lower bound
itself above .25; on the 95 this interval spans about +/-.3. Reported either way, with the secondary arms read as descriptive. Passing
confirms that the predictor beats edit size on new speakers; it does not rank the three arms.

## Re-rates (no bearing on the pass)

- retest_95: Spearman, exact agreement and within-one agreement with the first rating, overall and on small edits. This is a rough
  ceiling, with memory of yesterday's rating possibly inflating it.
- retest_gemini: the same, plus the mean absolute distance to Gemini's mean answer in the first rating and in the blind one. A larger
  distance when blind means the first rating leaned on the score. The 27 are not a random sample, and most are label 1.

## Amendment, before any score of these ratings

The rater stopped after 69 pairs: 67 tagged and 2 with no voice rating. Usable: 40 fresh, 5 retest_95, 16 retest_gemini; 6 are broken. The
stop was blind, and the page's speaker order was shuffled, so the rated part is not chosen by any score. Ratings are frozen at
SHA-256 9cea28054f02a371785f437528b25a48d434ab38803d298ddbb14ba630599a51.

The rule and arms are unchanged. Only the small-old (30) and mid-old (6) cells reach 5 items, so within rests on 36 pairs, and small
ratings are only 0 or 1. Low power is expected: if the interval includes 0 the result is inconclusive, not a failure, and the same if
a point estimate clears +.25 with a wide interval. The 5 retest_95 pairs are too few for a ceiling and are reported as counts only.

## Result, on the 67 tagged pairs

No pass. Fresh pairs: 40. Within / small / overall / minus edit, 99% interval: LoRA + attention .142/.220/.536/-.232 (-.715,.307); LoRA +
mean .129/.204/.521/-.245 (-.703,.258); frozen attention + taps .218/.268/.559/-.156 (-.589,.303); edit size .374/.344/.675. By the amended
rule this is inconclusive, but all three point estimates sit below edit size, and the primary arm's +.373 on the 95 is above this
interval, so the 95's gain does not carry over to these speakers at face value. Not separable here: reuse of the 95, a harder speaker
mix or noise on 36 pairs. retest_95, 5 pairs: exact .4, within one 1.0. retest_gemini, 14 pairs: exact .429, within one 1.0, Spearman
.076, blind ratings .286 lower on average; distance to Gemini's mean .476 at first, .381 blind, so no sign of leaning on the score.

## Post-hoc diagnostics, after the result (explanatory only; the result above stands)

Within / small / overall. Fresh: edit .374/.344/.675; Jev mean .291/.356/.641; Jev tone .202/.348/.606; rule untied .453/.553/.696; scan average, 39
pairs, .472/.536/.712; LoRA attention .142/.220/.536. The 95: edit .217/.258/.483; Jev mean .650/.571/.715; Jev tone .288/.488/.495; rule untied
.625/.654/.684; scan average .662/.661/.749; LoRA attention .590/.526/.702. Every teacher falls on the fresh set, and the model falls most. Scan
and rule stay above edit, so the set is not voiceless.

Small edits: rater 0/30 at 2 or more against 10/32 on the 95 (Fisher p .0009), mean .633 against 1.25; Gemini scan mean .563 against .989, at 2 or
more 1/29 against 3/31. Fresh speakers have a median of 42.5 lines against 263. LoRA attention against Jev, small edits: rho .695 fresh, .788 on the 95.
On the 95 with small edits rated 0 or 1 only (22 pairs): LoRA attention .370, edit .193.

The same 19 re-rated pairs, against the first rating then the blind one: LoRA attention .645 then .064; frozen taps .624 then .169; edit .455 then
.335; scan .261 then .014. Ratings at 2 fell from 6 to 1. Gemini's score was on screen for 14 of the first ratings, and 19 pairs, 15 of them rated 1, is
very few.
