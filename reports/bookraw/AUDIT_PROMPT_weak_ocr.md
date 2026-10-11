# Prompt: audit weak-OCR pages in the strict scanned-books table

## Context you need before starting

`reports/bookraw/books_scanned.parquet` holds 40,838 author-body paragraph rows
from 20 scanned Chinese books. Roles, page boundaries, and cleanups were reviewed
against page images by GPT-6 Luna. **Character-level text was not verified.**
OCR line confidence is a recognition-model signal, not accuracy, so the table can
contain silently wrong characters inside otherwise correctly-scoped paragraphs.

An automated audit flagged 1,505 of 10,316 pages. The breakdown is:

| issue kind | pages | status |
| --- | ---: | --- |
| `weak_ocr` | 1,395 | **unreviewed** |
| `overlapping_retained_lines` | 109 | unreviewed |
| `no_retained_content` | 68 | unreviewed |
| `possible_embedded_page_number` | 22 | unreviewed |
| `possible_truncated_source` | 5 | **closed** (4 `false_positive`, 1 `confirmed_source_truncation`) |
| `archive_boilerplate` | 4 | unreviewed |
| `adjacent_duplicate_content` | 3 | closed (3 confirmed duplicates) |

So the real open work is the 1,395 `weak_ocr` pages, which is where character
errors most plausibly hide. Truncation and duplicate scans are already decided;
do not reopen them except as noted in step 5.

## Your task

Work the `weak_ocr` queue. For each sampled page, decide whether the weak lines
are a recognition artifact or a real character error, and emit a machine-readable
verdict file. **Do not modify any source artifact.** You are auditing, not
repairing — corrections go in an overlay file for a human to approve.

## Inputs (all read-only)

```
QUEUE   /Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/quality_audit/queue.json
PAGES   /Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/<book_id>/pages/<NNNNN>.json
PDFS    /Volumes/EMC/stylevec_data/raw/human_author/contemporary_authors/*.pdf
TABLE   reports/bookraw/books_scanned.parquet
LINEAGE reports/bookraw/books_scanned_lineage.jsonl
EXCL    reports/bookraw/books_scanned_exclusions.jsonl
VERIF   reports/bookraw/books_scanned_verification.json
```

`queue.json` is a list of 1,505 entries. Each has `book`, `page`, `pdf`,
`raw_json`, `image`, and `issues[]`. A `weak_ocr` issue carries `count` and
`examples[]` of `{id, text, score, box}` where `score < 0.75`.

Join to the strict table via lineage: `source_fragment_ids` in
`books_scanned_lineage.jsonl` use `<book_id>:<page>:<block>:<line>`, the same
form as the `examples[].id`. That tells you whether a weak line actually
reached a training row or was already excluded.

## Precedent to follow

`source_truncation_checks.json` and `duplicate_source_checks.json` are the
format precedent. Match their shape:

```json
{"book": "...", "page": 406, "verdict": "false_positive",
 "rationale": "one or two sentences citing what the image shows",
 "inspected_image": "...png"}
```

Render page images at 200 DPI for inspection; the existing audit images are only
20 sampled pages, so you will need to render your own. Do not overwrite files in
`quality_audit/images/`. Write to a new directory.

## Procedure

1. **Sample, don't boil the ocean.** 1,395 pages is too many to inspect one by
   one in a single run. Draw a stratified random sample by book, sized so each
   book's sample supports an error-rate estimate — target at least 30 pages per
   book for the 18 books with meaningful counts, and note that
   `24520e2f06ee` (5) and `bc43c71cc312` (6) cannot support per-book rates. Fix
   the seed and record it. Report per-book rates with intervals, not a pooled
   number that hides `95f2c9116a47` (138) versus `24520e2f06ee` (5).

2. **Render and inspect.** For each sampled page, render the PDF page at 200 DPI,
   crop to the `box` of each weak line, and compare against the OCR `text`.
   Reading the whole page is not enough; the boxes are small and the errors are
   character-level.

3. **Classify each weak line** into exactly one of:
   - `artifact` — punctuation, a digit read as a glyph, box misaligned, or a
     legitimately low-confidence line that the image confirms as correct text
   - `char_error` — the image clearly shows different text; record the correct
     reading
   - `source_damage` — blur, ink loss, or crop in the scan itself; the OCR is
     not at fault and the text is unrecoverable
   - `unclear` — you cannot decide from the image; say why

   Emit one row per weak line, plus a per-page verdict.

4. **Separate the artifact rate from the error rate.** `weak_ocr` fires on any
   line under 0.75, which over-selects punctuation and isolated glyphs. Report:
   - the share of weak lines that are `artifact`
   - the **residual character error rate**: `char_error` lines divided by total
     OCR lines on those pages, not by weak lines
   - how many `char_error` lines landed in actual strict-table rows, and how many
     were already excluded

   The second number is the one that matters for downstream use. A 60% weak-line
   artifact rate means little; a 0.3% residual row error rate on 40,838 rows
   means roughly 120 corrupted paragraphs.

5. **Do not silently expand scope.** If evidence turns up outside the queue —
   a truncation case not already adjudicated, a duplicate, a role error — record
   it in a separate `incidental_findings` list with page and image reference.
   Do not edit `queue.json`, the parquet, the lineage, or any exclusion file.

6. **Spot-check the one confirmed truncation.** `ce71b57bb15e` p.19 is
   `confirmed_source_truncation` and is excluded from the strict table. Confirm it
   is still excluded in the current export, and confirm no adjacent page absorbed
   a fragment that should have been left separate.

7. **State the limit.** Your review is model-assisted image inspection, exactly
   like the Luna role review. It is not human proofreading, and blurred or
   damaged print remains unrecoverable regardless of how many times it is
   reviewed. Say this in the report.

## Output

- `reports/bookraw/weak_ocr_review_<seed>.jsonl` — per line, then per page
- `reports/bookraw/WEAK_OCR_AUDIT.md` — method, sample design and seed, per-book
  rates with intervals, residual error rate, the count of corrupted strict rows,
  incidental findings, and the limitation statement
- New page renders in a new directory, referenced by absolute path

## Rules

- Read-only with respect to every existing file in `reports/bookraw/` and
  `/Volumes/EMC/stylevec_data/processed/`
- No claimed correction enters the table without human approval; your output is
  an overlay, not a repair
- Keep confidence-scored model judgments labeled as model judgments. A number
  with no interval and no sample size behind it is not a result here.
- If the sample is too small to support a per-book claim, say so instead of
  reporting the number anyway.