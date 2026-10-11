# Weak-OCR audit of the strict scanned-books table

**All judgments below are model-assisted image inspection (Claude reading 200-DPI crops; 79 of 649 lines were first read by a Haiku sub-model, 39 of those re-read by Claude). They are not human proofreading.** Nothing in `reports/bookraw/` or `/Volumes/EMC/stylevec_data/processed/` was modified; the per-line `image_reading` values in `weak_ocr_review_20261011.jsonl` are an overlay for human approval, not a repair.

## Headline numbers

- Sample: **551 of 1,395** `weak_ocr` pages, **649 weak lines** (seed `20261011`, stratified by book, 30 pages/book; books with ≤30 weak pages taken in full).
- Weak-line verdicts: `artifact` **572 (88.1%)**, `char_error` **47 (7.2%)**, `unclear` **29 (4.5%)**, `source_damage` **1**.
- **Residual character error rate** (char_error lines ÷ all OCR lines on the sampled pages): 47 / 14,893 = **0.3%** (Wilson 95% 0.2%–0.4%; ignores page clustering, so slightly narrow). Counting every `unclear` line as an error gives an upper bound of 0.5%.
- Among OCR lines that were actually **retained in strict-table rows** on those pages: 16 char_error lines / 10,785 retained lines = **0.1%** (Wilson 95% 0.1%–0.2%).
- Where the 47 char_error lines landed: **16 in strict-table rows**, 29 already excluded (covers, handwriting, short/boundary rows), 2 in neither. Of all 649 weak lines: 116 in strict rows, 517 excluded, 16 in neither.
- **Corrupted strict rows, extrapolated to the weak-page population:** 35 rows (95% stratified bootstrap 18–54) of 40,838 = **0.09%**. Counting `unclear` retained lines too: 36 (19–55). This is **roughly 35, not ~120**. It covers only pages the heuristic flagged `weak_ocr`; errors on high-confidence lines (or on the 110 non-weak flagged pages) were not measured, so treat it as a floor for that error class.
- In the sample, 16 distinct strict rows contain a char_error line; in 13 of them the wrong OCR text is still present verbatim in the exported row (the other 3 were presumably changed by the earlier cleanup/verified-transcription passes).

## Separate finding that matters more: stray `O`/`0` at row ends

Most `artifact` lines are `。` or `”` misread as `O`, `,`, `9`… For lines that were **not** retained this is harmless, but **60 retained weak lines differ from the image while counted `artifact`, and in 55 the wrong text is still in the exported row** (36 of them are a literal `O` for `。`; for short strings like `,` the substring test can match coincidentally). A full deterministic scan of the table (no sampling) finds **77 of 40,838 strict rows ending in a stray Latin `O`/`0` after a Chinese character or punctuation** (55 `。O`, 22 other, e.g. `了O`) — a sentence-final-punctuation corruption that a style model would learn. Rows by book: bc87fbaddf81 15, ba1ea6a1eb11 14, 36900f1061f3 11, 0d144c794cf6 9, 1c0e28a46b63 8, 1cc7f875ba10 8, d5f0f5c8c05e 6, b9b40c19e637 4, b551c3764d50 1, f81d523111a4 1. List: `weak_ocr_trailing_O_rows_20261011.csv`. A regex strip/replace is a **proposal for human approval**, not applied. The `。O` form may be a duplicate ring detection rather than a wrong glyph; I did not determine which.

## Per-book results

`n`/`N` = sampled / weak pages. Page rate = share of sampled weak pages with ≥1 char_error line (Wilson 95%). Books marked census were read in full (no sampling error). Books with ≤6 pages cannot support a rate, so only counts are given.

| book | N | n | weak lines | artifact | char_error | unclear | page rate (≥1 char_error) | char_error / OCR lines | char_error in strict rows |
|---|--:|--:|--:|--:|--:|--:|---|---|--:|
| 0d144c794cf6 柳青文集 | 59 | 30 | 37 | 35 | 2 | 0 | 6.7% (1.8%–21.3%) | 0.2% (0.1%–0.9%) | 0 |
| 1c0e28a46b63 梁斌文集 | 30 | 30 | 32 | 29 | 3 | 0 | 10.0% (3.5%–25.6%) census | 0.4% (0.1%–1.1%) | 1 |
| 1cc7f875ba10 梁斌文集 | 64 | 30 | 35 | 33 | 2 | 0 | 6.7% (1.8%–21.3%) | 0.3% (0.1%–0.9%) | 0 |
| 24520e2f06ee 何其芳全集全八卷 | 5 | 5 | 5 | 4 | 1 | 0 | counts only (n too small) | — | 0 |
| 34139a2ac15b 陆文夫代表作 | 84 | 30 | 37 | 36 | 0 | 1 | 0.0% (0.0%–11.4%) | 0.0% (0.0%–0.5%) | 0 |
| 36900f1061f3 梁斌文集 | 77 | 30 | 34 | 32 | 2 | 0 | 6.7% (1.8%–21.3%) | 0.2% (0.1%–0.9%) | 0 |
| 4484675aeb7e 周立波文集 | 41 | 30 | 38 | 30 | 3 | 5 | 10.0% (3.5%–25.6%) | 0.4% (0.1%–1.2%) | 1 |
| 55e91c672ed4 周立波文集 | 109 | 30 | 34 | 27 | 3 | 4 | 10.0% (3.5%–25.6%) | 0.4% (0.1%–1.2%) | 0 |
| 91558afdf6a5 周立波文集 | 96 | 30 | 34 | 28 | 4 | 2 | 10.0% (3.5%–25.6%) | 0.6% (0.2%–1.5%) | 2 |
| 94a5fb7e75e1 英华之歌 (杨沫) (z-library.s | 39 | 30 | 31 | 25 | 5 | 1 | 13.3% (5.3%–29.7%) | 0.7% (0.3%–1.6%) | 3 |
| 95f2c9116a47 周立波文集 | 138 | 30 | 34 | 33 | 1 | 0 | 3.3% (0.6%–16.7%) | 0.1% (0.0%–0.7%) | 0 |
| b551c3764d50 张承志精选集 | 38 | 30 | 33 | 30 | 3 | 0 | 10.0% (3.5%–25.6%) | 0.2% (0.1%–0.7%) | 0 |
| b9b40c19e637 杨沫文集 | 76 | 30 | 34 | 29 | 5 | 0 | 16.7% (7.3%–33.6%) | 0.6% (0.3%–1.4%) | 3 |
| ba1ea6a1eb11 柳青文集 | 103 | 30 | 33 | 33 | 0 | 0 | 0.0% (0.0%–11.4%) | 0.0% (0.0%–0.4%) | 0 |
| bc43c71cc312 何其芳全集全八卷 | 6 | 6 | 7 | 7 | 0 | 0 | counts only (n too small) | — | 0 |
| bc87fbaddf81 柳青文集 | 130 | 30 | 36 | 36 | 0 | 0 | 0.0% (0.0%–11.4%) | 0.0% (0.0%–0.4%) | 0 |
| ce71b57bb15e 周立波文集 | 115 | 30 | 33 | 30 | 3 | 0 | 6.7% (1.8%–21.3%) | 0.4% (0.1%–1.2%) | 1 |
| d5f0f5c8c05e 柳青文集 | 55 | 30 | 35 | 35 | 0 | 0 | 0.0% (0.0%–11.4%) | 0.0% (0.0%–0.5%) | 0 |
| f81d523111a4 杨沫文集 | 71 | 30 | 37 | 31 | 6 | 0 | 20.0% (9.5%–37.3%) | 0.8% (0.3%–1.6%) | 2 |
| f8cd13a4e2f9 臧克家全集 | 59 | 30 | 50 | 29 | 4 | 16 | 13.3% (5.3%–29.7%) | 0.5% (0.2%–1.4%) | 3 |

With 30 pages per book the intervals are wide (e.g. 0/30 still allows up to ~11%); differences between most books are not statistically distinguishable. Clearly different: `f8cd13a4e2f9` (handwritten facsimile pages → 16 `unclear`), `4484675aeb7e` (handwriting, 5 unclear) and `55e91c672ed4` (1-bit scan, 4 unclear).

## Method

1. Queue: `quality_audit/queue.json` → 1,395 pages with a `weak_ocr` issue (1,707 lines scored <0.75; 649 lines on the sampled pages).
2. Sample: per book `random.Random(20261011).sample(pages sorted by page, 30)`; books with ≤30 weak pages taken whole. Page numbers in the jsonl.
3. Render: PyMuPDF at 200 DPI (dimensions matched the OCR frame for every page) into `/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011/pages/`; each weak line cropped by its `box` (+70×28 px context, red outline) into 10-crop contact sheets (`sheets/`), shown with the OCR text alongside.
4. Classify each line (`artifact`/`char_error`/`source_damage`/`unclear`), recording `image_reading` and `ocr_text_differs_from_image`. Retained status via `source_fragment_ids` in the lineage; exclusions from `books_scanned_exclusions.jsonl` + `books_scanned_strict_exclusions.jsonl`. Wrong text persisting in a row checked by substring against `books_scanned.parquet`.
5. Rates: Wilson intervals; row extrapolation by per-book scaling (N/n × sample rows) with 10,000 stratified page bootstraps (numpy seed 20261011, no finite-population correction → slightly conservative).

Reviewer provenance (field `reviewer`): lines 0–569 `claude`; 570–648 first read by Haiku sub-models, of which 39 lines (580–599, 620–629, 640–648) were re-read by Claude and 7 verdicts changed (580, 583, 588, 590, 594, 598, 628); **40 Haiku lines (570–579, 600–619, 630–639) were not independently verified**, including ~20 handwriting `unclear` lines.

## Step 6: `ce71b57bb15e` p.19 truncation

- Truncated paragraph `19:1:0` is still excluded (`boundary_reidentification_requires_review`); the page image shows the print cut mid-line after “我们应当从广大的民”.
- Fragment `19:0:0` (top of p.19) is in the **same retained row as `18:2:0`** (pages [18,19]); the image shows it is the genuine, complete continuation of the p.18 paragraph (ends with 。), so that merge is correct.
- Page 20 begins mid-sentence: `20:0:0` is excluded, **but `20:0:1` is retained** as row `ce71b57bb15e:ba364402edfec025467b` and starts “一新的现实主义一的作家们十分概念的产物…” — a headless continuation of the truncated text (see incidental findings).

## Incidental findings (not acted on)

1. `ce71b57bb15e` p.20: retained row `ce71b57bb15e:ba364402edfec025467b` (fragment `20:0:1`) begins mid-clause because its first line `20:0:0` and the preceding page body were lost to the confirmed p.19 truncation. Candidate for exclusion; needs human decision.
2. Trailing `O`/`0` rows (above): 77 rows, deterministic scan.
3. Many weak lines in `55e91c672ed4`, `4484675aeb7e`, `f8cd13a4e2f9` sit on handwriting/halftone facsimile pages; their OCR text is meaningless (marked `unclear`). None of their `unclear` lines were in strict rows.
4. Cover/spine/contents char errors (`文集`→`文繁`, `徐訏`→`徐訂`, `人民文学出版社`→`…出服社` etc.) are all outside the strict table.

## Strict-table rows with a confirmed/likely char_error (overlay for approval)

| book | page | fragment id | OCR text | image reading | note |
|---|--:|---|---|---|---|
| 1c0e28a46b63 | 181 | `1c0e28a46b63:181:1:0` | 同四中” | 向西冲…… | OCR 同四中” (three wrong chars + spurious quote) |
| 4484675aeb7e | 244 | `4484675aeb7e:244:0:0` | 的； | 的： | colon (two stacked equal dots) read as semicolon ；; moderate confidence |
| 91558afdf6a5 | 472 | `91558afdf6a5:472:0:0` | 街 | 声。 | 声 read as 街 (3 chars? box covers one glyph) |
| 91558afdf6a5 | 716 | `91558afdf6a5:716:0:0` | L- |  | box contains only specks; OCR hallucinated L- |
| 94a5fb7e75e1 | 123 | `94a5fb7e75e1:123:4:0` | 在我找展用安靜极了 | 有人来找，屋里寂静极了。 | whole line misread (在我找展用安靜极了) |
| 94a5fb7e75e1 | 192 | `94a5fb7e75e1:192:4:0` | b√,i) |  | box blank/specks; OCR hallucinated b√,i) |
| 94a5fb7e75e1 | 579 | `94a5fb7e75e1:579:0:0` | 产： | 声： | 声 read as 产 |
| b9b40c19e637 | 40 | `b9b40c19e637:40:0:0` | 穷： | 穷。 | bold 。 read as ： |
| b9b40c19e637 | 343 | `b9b40c19e637:343:5:0` | 做， | 做。 | bold 。 read as ， |
| b9b40c19e637 | 357 | `b9b40c19e637:357:4:0` | 下， | 下： | colon read as ， |
| ce71b57bb15e | 405 | `ce71b57bb15e:405:2:0` | 是恐俱。”】 | 是恐惧。”] | 惧 read as 俱 (1-bit scan; moderate confidence) |
| f81d523111a4 | 293 | `f81d523111a4:293:2:0` | 的， | 的。 | bold 。(solid dot) read as ，; verified by Claude |
| f81d523111a4 | 550 | `f81d523111a4:550:4:0` | 醫。 | 冒。 | 冒 read as 醫 (moderate confidence); Claude override of reviewer unclear |
| f8cd13a4e2f9 | 53 | `f8cd13a4e2f9:53:5:0` | 我及心起： | 我爱想起： | 爱/想 read as 及/心; verified by Claude |
| f8cd13a4e2f9 | 639 | `f8cd13a4e2f9:639:2:0` | 他昕来的比化卖出的更本宜 | 他听来的比他卖出的更丰富， | 昕 read as 听, 化 read as 他, 本宜 read as 丰富 |
| f8cd13a4e2f9 | 706 | `f8cd13a4e2f9:706:2:0` | 坐在抗头上像企大址娘 | 坐在炕头上像个大姑娘， | 抗 read as 炕, 企 read as 个, 址 read as 姑 |

Several of these readings are moderate confidence (1-bit low-resolution scans, e.g. 俱/惧, ：/；, bold 。 vs ，). They need human confirmation before use.

## Limitations

- Model judgment on images, not human proofreading. Blurred, damaged or handwritten print remains unrecoverable however often it is re-read; such lines are `unclear`/`source_damage`, not resolved.
- Only `weak_ocr` pages were sampled. The 110 pages flagged for other reasons and ~8,800 unflagged pages were not examined, and errors on lines OCR was confident about are invisible to this design.
- Many residual errors are judgment calls on 1-bit scans (`unclear` = 29 lines; they could shift totals by ~±1 row).
- Intervals ignore line clustering within pages and use no finite-population correction for the row extrapolation.
- Reviewer = Claude for 570 lines and Haiku (partly spot-checked) for 79; Haiku-only verdicts are unverified.
