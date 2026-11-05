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
