# Trait predictor: sealed test against Argon's scan

Written before any pair below was scanned or scored. The rater reports that their ratings are less consistent than Argon's (re-rates: rater Spearman .21,
Argon .46 on the same 19 pairs), so this asks a separate question from prereg/trait_confirmation.md, whose no-pass stands. Argon is a working reference, not
truth: it is also an LLM, its level moves with its prompt, and the only evidence for this reference is that consistency and its agreement with the rater.

## Arms (frozen, seed 42, step 3000)

Primary: lora_r8_delta1.0_s42_attention, unchanged from the earlier prereg. Secondary: lora_r8_delta1.0_s42, frozen_delta1.0_s42_attention_layers6-12-18-24-28.
Baseline: edit size. A pair's score is the mean absolute predicted trait difference. Nothing is retrained, tuned or added.

## Items

data/human_ratings/argon_test_key.jsonl, built by rate_voice_server.py build-argon, 300 small edits (edit < .15), SHA-256 2f9e3b5926101b90cdb42c533b649b45bf4ecb3d52bfbe8c16cc42e33a4ec672. Random, chosen by no LLM tag. Speakers
have at least 20 lines, none in the predictor's train or dev, none rated; at most 3 pairs per speaker. No pair from small-dev or the confirmation key.

## Reference

scripts/argon_scan.py: plain run 1, plain run 2 and the careful prompt, batches of 40, one line of talk on each side. To save tokens this differs from earlier
scans, which had four lines of talk and high effort on the careful run. On the 40 rated fresh pairs: one line against four, exact .92, Spearman .94, no level shift; a
second full-context plain run scored .97 and .97. Default-effort careful against high-effort careful: exact .69, Spearman .78, level 1.00 against 1.15. So the careful
run here is a lighter instrument than before. Run budget: 150,000 tokens.

Primary target: mean of the three runs. Also reported: mean of the two plain runs. A pair without a digit from all three runs is dropped and counted.

## Pass

Spearman of the primary arm minus Spearman of edit size against the primary target at least +.10, with its 99% paired bootstrap interval excluding 0 (4,000 resamples of pairs, seed 20261105).
The +.10 follows prereg/external_style_test.md. It is set after seeing +.12 on 29 pairs, so it is not independent of that peek; those pairs are not in this set. The secondary
arms are descriptive. A pass says the model ranks small edits more like Argon than edit size does on unseen speakers. It does not show that either reads style as a person does.

## Disagreement audit, after scoring

Pairs of similar edit size, one where the primary arm ranks far above Argon and one the reverse, shown side by side without saying which is which. The rater chooses
which line changes voice more. The tally says which scorer the rater sided with; it has no bearing on the pass.

## Result

Pass. 293 of 300 pairs usable, 7 dropped for a broken answer in some run; 117,795 tokens. Spearman against the three-run mean, gain over edit size (.412), 99% interval: LoRA attention .613, +.201 (.069,.341);
LoRA mean .611, +.199 (.071,.333); frozen taps .589, +.177 (.044,.319). Against plain runs only (edit .384): .588, +.205 (.068,.349); .586, +.202 (.064,.345); .567, +.183 (.041,.330). The arms cannot be ranked from this.
Argon run 1 against run 2: exact .746, Spearman .722. Model against edit size: Spearman .473.
