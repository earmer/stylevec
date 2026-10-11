import json, math, collections
A='/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011/'
RB='/Users/earmercarey/stylevec/reports/bookraw/'
S=json.load(open(A+'stats.json')); st=S['S']
R=[json.loads(x) for x in open(RB+'weak_ocr_review_20261011.jsonl')]; lines=[r for r in R if r['kind']=='line']
names={}
import json as j
for e in j.load(open('/Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/quality_audit/queue.json')): names[e['book']]=e['name']
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return max(0,c-h),min(1,c+h)
pc=lambda x:f"{100*x:.1f}%"
short=lambda b:names[b].split('_')[0][:22]
out=[]
w=out.append
w("# Weak-OCR audit of the strict scanned-books table\n")
w("**All judgments below are model-assisted image inspection (Claude reading 200-DPI crops; 79 of 649 lines were first read by a Haiku sub-model, 39 of those re-read by Claude). They are not human proofreading.** Nothing in `reports/bookraw/` or `/Volumes/EMC/stylevec_data/processed/` was modified; the per-line `image_reading` values in `weak_ocr_review_20261011.jsonl` are an overlay for human approval, not a repair.\n")
w("## Headline numbers\n")
tl,ta,tc,tu,tsd=S['tl'],S['ta'],S['tc'],S['tu'],S['tsd']
lo,hi=wilson(tc,S['tocr']); lo2,hi2=wilson(S['ce_strict'],S['tret'])
w(f"- Sample: **{sum(s['n'] for s in st.values())} of 1,395** `weak_ocr` pages, **{tl} weak lines** (seed `20261011`, stratified by book, 30 pages/book; books with ≤30 weak pages taken in full).")
w(f"- Weak-line verdicts: `artifact` **{ta} ({pc(ta/tl)})**, `char_error` **{tc} ({pc(tc/tl)})**, `unclear` **{tu} ({pc(tu/tl)})**, `source_damage` **{tsd}**.")
w(f"- **Residual character error rate** (char_error lines ÷ all OCR lines on the sampled pages): {tc} / {S['tocr']:,} = **{pc(tc/S['tocr'])}** (Wilson 95% {pc(lo)}–{pc(hi)}; ignores page clustering, so slightly narrow). Counting every `unclear` line as an error gives an upper bound of {pc((tc+tu)/S['tocr'])}.")
w(f"- Among OCR lines that were actually **retained in strict-table rows** on those pages: {S['ce_strict']} char_error lines / {S['tret']:,} retained lines = **{pc(S['ce_strict']/S['tret'])}** (Wilson 95% {pc(lo2)}–{pc(hi2)}).")
w(f"- Where the {tc} char_error lines landed: **{S['ce_strict']} in strict-table rows**, {S['ce_excl']} already excluded (covers, handwriting, short/boundary rows), {S['ce_neither']} in neither. Of all {tl} weak lines: {S['wl'][0]} in strict rows, {S['wl'][1]} excluded, {S['wl'][2]} in neither.")
w(f"- **Corrupted strict rows, extrapolated to the weak-page population:** {S['est_pt']:.0f} rows (95% stratified bootstrap {S['ci'][0]:.0f}–{S['ci'][1]:.0f}) of {S['TOTAL_ROWS']:,} = **{100*S['est_pt']/S['TOTAL_ROWS']:.2f}%**. Counting `unclear` retained lines too: {S['est_hi_pt']:.0f} ({S['ci_hi'][0]:.0f}–{S['ci_hi'][1]:.0f}). This is **roughly 35, not ~120**. It covers only pages the heuristic flagged `weak_ocr`; errors on high-confidence lines (or on the 110 non-weak flagged pages) were not measured, so treat it as a floor for that error class.")
w(f"- In the sample, {S['distinct_rows']} distinct strict rows contain a char_error line; in {S['distinct_rows_present']} of them the wrong OCR text is still present verbatim in the exported row (the other {S['distinct_rows']-S['distinct_rows_present']} were presumably changed by the earlier cleanup/verified-transcription passes).\n")
w("## Separate finding that matters more: stray `O`/`0` at row ends\n")
w(f"Most `artifact` lines are `。` or `”` misread as `O`, `,`, `9`… For lines that were **not** retained this is harmless, but **{S['D_strict']} retained weak lines differ from the image while counted `artifact`, and in {S['D_strict_present']} the wrong text is still in the exported row** (36 of them are a literal `O` for `。`; for short strings like `,` the substring test can match coincidentally). A full deterministic scan of the table (no sampling) finds **77 of 40,838 strict rows ending in a stray Latin `O`/`0` after a Chinese character or punctuation** (55 `。O`, 22 other, e.g. `了O`) — a sentence-final-punctuation corruption that a style model would learn. Rows by book: bc87fbaddf81 15, ba1ea6a1eb11 14, 36900f1061f3 11, 0d144c794cf6 9, 1c0e28a46b63 8, 1cc7f875ba10 8, d5f0f5c8c05e 6, b9b40c19e637 4, b551c3764d50 1, f81d523111a4 1. List: `weak_ocr_trailing_O_rows_20261011.csv`. A regex strip/replace is a **proposal for human approval**, not applied. The `。O` form may be a duplicate ring detection rather than a wrong glyph; I did not determine which.\n")
w("## Per-book results\n")
w("`n`/`N` = sampled / weak pages. Page rate = share of sampled weak pages with ≥1 char_error line (Wilson 95%). Books marked census were read in full (no sampling error). Books with ≤6 pages cannot support a rate, so only counts are given.\n")
w("| book | N | n | weak lines | artifact | char_error | unclear | page rate (≥1 char_error) | char_error / OCR lines | char_error in strict rows |")
w("|---|--:|--:|--:|--:|--:|--:|---|---|--:|")
for b,s in st.items():
    if s['N']<=6:
        rate="counts only (n too small)"; lr="—"
    else:
        a,c=wilson(s['ce_pages'],s['n']); rate=f"{pc(s['ce_pages']/s['n'])} ({pc(a)}–{pc(c)}){' census' if s['census'] else ''}"
        a,c=wilson(s['char_error'],s['ocr_lines']); lr=f"{pc(s['char_error']/s['ocr_lines'])} ({pc(a)}–{pc(c)})"
    w(f"| {b} {short(b)} | {s['N']} | {s['n']} | {s['lines']} | {s['artifact']} | {s['char_error']} | {s['unclear']} | {rate} | {lr} | {s['ce_strict']} |")
w("\nWith 30 pages per book the intervals are wide (e.g. 0/30 still allows up to ~11%); differences between most books are not statistically distinguishable. Clearly different: `f8cd13a4e2f9` (handwritten facsimile pages → 16 `unclear`), `4484675aeb7e` (handwriting, 5 unclear) and `55e91c672ed4` (1-bit scan, 4 unclear).\n")
w("## Method\n")
w("1. Queue: `quality_audit/queue.json` → 1,395 pages with a `weak_ocr` issue (1,707 lines scored <0.75; 649 lines on the sampled pages).\n2. Sample: per book `random.Random(20261011).sample(pages sorted by page, 30)`; books with ≤30 weak pages taken whole. Page numbers in the jsonl.\n3. Render: PyMuPDF at 200 DPI (dimensions matched the OCR frame for every page) into `/Volumes/EMC/stylevec_data/audit/weak_ocr_review_20261011/pages/`; each weak line cropped by its `box` (+70×28 px context, red outline) into 10-crop contact sheets (`sheets/`), shown with the OCR text alongside.\n4. Classify each line (`artifact`/`char_error`/`source_damage`/`unclear`), recording `image_reading` and `ocr_text_differs_from_image`. Retained status via `source_fragment_ids` in the lineage; exclusions from `books_scanned_exclusions.jsonl` + `books_scanned_strict_exclusions.jsonl`. Wrong text persisting in a row checked by substring against `books_scanned.parquet`.\n5. Rates: Wilson intervals; row extrapolation by per-book scaling (N/n × sample rows) with 10,000 stratified page bootstraps (numpy seed 20261011, no finite-population correction → slightly conservative).")
w("\nReviewer provenance (field `reviewer`): lines 0–569 `claude`; 570–648 first read by Haiku sub-models, of which 39 lines (580–599, 620–629, 640–648) were re-read by Claude and 7 verdicts changed (580, 583, 588, 590, 594, 598, 628); **40 Haiku lines (570–579, 600–619, 630–639) were not independently verified**, including ~20 handwriting `unclear` lines.\n")
w("## Step 6: `ce71b57bb15e` p.19 truncation\n")
s6=S['step6']
w(f"- Truncated paragraph `19:1:0` is still excluded (`{s6['f19_1_0'][0]}`); the page image shows the print cut mid-line after “我们应当从广大的民”.\n- Fragment `19:0:0` (top of p.19) is in the **same retained row as `18:2:0`** (pages [18,19]); the image shows it is the genuine, complete continuation of the p.18 paragraph (ends with 。), so that merge is correct.\n- Page 20 begins mid-sentence: `20:0:0` is excluded, **but `20:0:1` is retained** as row `{s6['f20_0_1_block']}` and starts “一新的现实主义一的作家们十分概念的产物…” — a headless continuation of the truncated text (see incidental findings).\n")
w("## Incidental findings (not acted on)\n")
w("1. `ce71b57bb15e` p.20: retained row `ce71b57bb15e:ba364402edfec025467b` (fragment `20:0:1`) begins mid-clause because its first line `20:0:0` and the preceding page body were lost to the confirmed p.19 truncation. Candidate for exclusion; needs human decision.\n2. Trailing `O`/`0` rows (above): 77 rows, deterministic scan.\n3. Many weak lines in `55e91c672ed4`, `4484675aeb7e`, `f8cd13a4e2f9` sit on handwriting/halftone facsimile pages; their OCR text is meaningless (marked `unclear`). None of their `unclear` lines were in strict rows.\n4. Cover/spine/contents char errors (`文集`→`文繁`, `徐訏`→`徐訂`, `人民文学出版社`→`…出服社` etc.) are all outside the strict table.\n")
w("## Strict-table rows with a confirmed/likely char_error (overlay for approval)\n")
w("| book | page | fragment id | OCR text | image reading | note |")
w("|---|--:|---|---|---|---|")
for r in lines:
    if r['verdict']=='char_error' and r['in_strict_row']:
        w(f"| {r['book']} | {r['page']} | `{r['line_id']}` | {r['ocr_text']} | {r['image_reading']} | {r['note']} |")
w("\nSeveral of these readings are moderate confidence (1-bit low-resolution scans, e.g. 俱/惧, ：/；, bold 。 vs ，). They need human confirmation before use.\n")
w("## Limitations\n")
w("- Model judgment on images, not human proofreading. Blurred, damaged or handwritten print remains unrecoverable however often it is re-read; such lines are `unclear`/`source_damage`, not resolved.\n- Only `weak_ocr` pages were sampled. The 110 pages flagged for other reasons and ~8,800 unflagged pages were not examined, and errors on lines OCR was confident about are invisible to this design.\n- Many residual errors are judgment calls on 1-bit scans (`unclear` = 29 lines; they could shift totals by ~±1 row).\n- Intervals ignore line clustering within pages and use no finite-population correction for the row extrapolation.\n- Reviewer = Claude for 570 lines and Haiku (partly spot-checked) for 79; Haiku-only verdicts are unverified.")
open(RB+'WEAK_OCR_AUDIT.md','w').write("\n".join(out)+"\n")
print(len("\n".join(out)))
