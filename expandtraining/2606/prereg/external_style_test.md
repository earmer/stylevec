# External style test, frozen 2026-11-04 before any arm was scored on these items

Question: on speakers no arm trained on, with labels this project did not produce, does any head_f arm beat a
text-only baseline?

Primary metric, only one: accuracy on data/episodes/external/cped.jsonl, the share of items where
cos(query, positive) > cos(query, negative), ties 0.5. Query, positive and negative are CPED TV-drama lines (subtitle
text, Chinese). The positive is the same speaker in another dialogue. The negative is another speaker in the query's
dialogue, closest in length to the positive, so scene favours the negative.

Baseline: char 1-3 TF-IDF, sublinear, fitted on the item texts. Also reported: length, the untrained encoder.
Arms: data, episodes, push, both, all seed 42, checkpoint latest. nohead has no local checkpoint and is not tested.

Pass for an arm: accuracy minus TF-IDF > 0.02, and the 99% speaker-bootstrap interval excludes 0.
A pass makes the arm a candidate only; a claim needs 3 seeds. It is one seed.
Null outcome: no arm passes. Then we say head_f shows no gain over text features on new Chinese dialogue speakers,
and no more training is proposed on this recipe until the metric or the data changes.
Direction of a loss: if every arm is below TF-IDF by more than 0.02 with the interval excluding 0, we say so.

Exploratory, no decision rests on them: each set's _shuffled version (speakers permuted, voice removed, topic kept: each
encoder's own null, reported beside it), authbench (new authors, same source as training), aozora (Japanese literature,
never trained on), the Reddit probe against its own shuffled null.

What the test can support: "beats text features on new speakers in the CPED domain". All 23 held-out shows also have
training speakers, so it cannot support "generalizes to the product" or to unseen shows. Aozora is the only never-trained
domain, and it is Japanese literature, not dialogue. Labels come from the datasets' authors, not from human style ratings.
Negatives may be spoken by speakers in the training set; only held-out speakers are queries and positives.

Item files (sha256, first 16):
- efc10274188d13be  data/episodes/external/aozora.jsonl
- 1ec93dc87979cefd  data/episodes/external/aozora_shuffled.jsonl
- 70b45b659da94b62  data/episodes/external/authbench.jsonl
- 7f2110fceda2f937  data/episodes/external/authbench_shuffled.jsonl
- 676071fbb6704cc7  data/episodes/external/cped.jsonl
- 7f829be3bf16f454  data/episodes/external/cped_shuffled.jsonl

## Voice-change ratings, frozen 2026-11-05 before any arm was scored on these items

Items: data/human_ratings/voice_change_{quick,full}.md, genshin_v2.db pairs from held-out speakers, unique A, 12-60 chars,
stratified by edit size and paraphrase batch. One rater, the user; broken items are excluded and counted.
Primary, on the full set only: Spearman between each arm's cosine distance (line, paraphrase) and the 0-3 rating, within
edit-size bins and within batch. Baseline: edit size. Pass: above the baseline by > 0.1 with a 99% item-bootstrap interval
excluding 0. The quick set is only a check of the form; no decision rests on it.
Item key files (sha256, first 16):
- d7fd22b8901b77e5  data/human_ratings/voice_change_full_key.jsonl
- 6404879857358ddd  data/human_ratings/voice_change_quick_key.jsonl
