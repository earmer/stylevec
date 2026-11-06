# 红楼梦: does the chapter-80 trait shift survive content matching?

Fixed before any matched result was computed. The unmatched results (`hongloumeng_trait_test.json`) were already known, and they chose the five traits.
Script: `scripts/hongloumeng_matched_test.py`. No new predictions.

**Matching.** Verse passages are dropped. Content is TF-IDF over the book's 600 most common non-function characters, so names count. Each late passage gets one early
passage, without replacement, by the best total cosine (Hungarian), inside the same length quartile and the same dialogue class (quoted share below .15, .15 to .5, at least .5).
The main set keeps the most similar half of the pairs. Traits have log length regressed out first.

**Primary model.** LoRA attention, unmasked. Names-masked and frozen taps are sensitivity runs.

**Pass, all three:**
1. Balance: a content-character classifier, cross-validated by chapter, has AUC at most .65 on the matched set. Above that the test counts as failed to remove content, whatever else it shows.
2. Traits: at least 4 of swagger, irony, playfulness, hedging, deference keep the unmatched sign in the matched pairs, with the 95% two-way chapter bootstrap interval excluding 0.
3. Separation: for the 30 chapters on each side of chapter 80 (51-80 vs 81-110), matched the same way, the trait classifier's AUC, cross-validated by chapter, is above all 20 placebo
   boundaries b = 31..50 (b-29..b vs b+1..b+30).

A pass means a tone shift that content matching did not remove. It does not establish a change of author: chapter 80 stays a noisy proxy for style, and matching on characters cannot hold plot mood constant.
Descriptive only: banquet, poetry, household and illness scenes by fixed keyword lists; narration and dialogue classes; base Qwen, old 256d, function and content characters on the same sets.

## Result: no pass (2026-11-06)

Balance failed, so by the rule above this test did not remove content. Content classifier: unmatched .855, matched .715, limit .65; the mean cosine of kept pairs is only .32. The other two parts met their
limits: all five traits kept their sign with intervals excluding 0 (playfulness -.48, irony -.46, swagger -.34, deference +.28, hedging +.26 passage SD, nearly the unmatched sizes), and the window AUC of .702
is above all 20 placebos (mean .484, max .551). The same matched window gives content characters .763 (placebos .601, .670) and function characters .701, so the traits do not beat them. 1,130 pairs, length still 161 against 134 characters before regression.
