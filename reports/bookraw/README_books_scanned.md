# Scanned books strict-primary dataset

`books_scanned.parquet` is the scan-derived counterpart to
`books_work_primary_strict.parquet`. It uses the reference table's exact Arrow
schema and keeps only page blocks inside high-confidence author-primary ranges
reviewed by GPT-6 Luna against the source-page images. These are model-reviewed
role and page-boundary judgments, not human certification or character-level
proofreading. Editorial matter, contents, unverified appendices,
layout furniture, excluded duplicate pages, unresolved boundary fragments,
short blocks, and scan-cleanup removals are recorded in the exclusion/audit
sidecars rather than silently entering the table.

## Work and source granularity

For collected works and multi-work volumes, a `work_id` may represent one
volume-level group of works. The corresponding `work` value names that volume
or collection; it does not claim exact per-story segmentation unless the range
review explicitly separates works. Author-authored essays, diaries, letters,
and signed prefaces remain primary text. Editorial introductions, translation,
commentary, and appendices without verified author attribution are excluded.

`source_spine_index` stores the one-based PDF page position. `source_href` uses
`pdf-page:<book_id>:<page>`; it is not an EPUB spine reference.
`source_dom_position` stores the zero-based row position in that source
revision's assembled `paragraphs.json` before strict-role and length filtering.
`source_block_id` is a stable digest of the source book and contributing OCR
fragment IDs. The lineage JSONL records those fragment IDs, page numbers,
assembled paragraph position, text hash, range, and OCR revision.

## Per-book reviewed scope

The range manifest contains 20 source books and 25 reviewed primary ranges.
Collected volumes are represented at the volume level where the scan does not
support reliable story-level boundaries. The exact range evidence and excluded
roles are in `scanned_body_ranges.jsonl`.

| Book ID | Included PDF pages | Reviewed exclusions |
| --- | --- | --- |
| `95f2c9116a47` | 9–518 | pp. 1–8 cover/title/contents; p. 519 duplicate scan. |
| `55e91c672ed4` | 10–651 | pp. 1–9 cover/title/contents/dividers; p. 652 duplicate scan. |
| `b9b40c19e637` | 7–523 | pp. 1–6 cover/publication/contents; p. 524 source metadata. |
| `ba1ea6a1eb11` | 9–436 | pp. 1–8 cover/publication/editor note/contents; p. 437 source metadata. |
| `bc87fbaddf81` | 7–439 | pp. 1–6 title matter; p. 440 source metadata. |
| `0d144c794cf6` | 9–312 | pp. 1–8 title/photo/contents; p. 313 source metadata. |
| `d5f0f5c8c05e` | 11–340 | pp. 1–10 title/contents; p. 341 source metadata. |
| `36900f1061f3` | 8–435 | pp. 1–7 title/editor note/contents; p. 436 source metadata. |
| `1cc7f875ba10` | 11–342 | pp. 1–10 title/contents; p. 343–406 appendix play lacks scan-verifiable author attribution; p. 407 source metadata. |
| `94a5fb7e75e1` | 6–653 | pp. 1–5 cover/title/summary/contents; the signed author preface on pp. 6–9 remains included. |
| `1c0e28a46b63` | 14–251 | pp. 1–13 title matter; pp. 252–290 appended interviews/commentary; p. 291 scan metadata. |
| `24520e2f06ee` | 55–498, 505–509 | pp. 1–54 title/external forewords/contents; pp. 499–504 third-person account by another contributor. |
| `34139a2ac15b` | 4, 28–444 | pp. 1–3 cover/title; pp. 5–23 series/editor/critic forewords; pp. 24–27 contents/divider; p. 4 author handwriting is recorded but excluded from strict OCR because unverified; pp. 445–446 compiled bibliography/end matter. |
| `4484675aeb7e` | 14–574 | pp. 1–13 cover/publication/contents; pp. 575–599 editor’s afterword. |
| `91558afdf6a5` | 16–755 | pp. 1–15 cover/contents/divider; p. 756 editorial publication history. |
| `b551c3764d50` | 26–398 | pp. 1–25 cover/CIP/editorial/contents; author-signed creation chronology on pp. 396–398 remains included. |
| `bc43c71cc312` | 56–558 | pp. 1–7 cover/title/photos; pp. 8–46 publisher statement/external forewords/contents/divider. |
| `ce71b57bb15e` | 15–729, 754–757 | pp. 1–14 cover/contents; pp. 730–753 editor’s afterword/correction notes. The corrected author poem excerpt on pp. 754–757 is included as primary text. |
| `f81d523111a4` | 4–11, 14–565 | pp. 1–2 cover/publication; p. 3 unattributed epigraph; pp. 12–13 contents; p. 566 scan metadata. |
| `f8cd13a4e2f9` | 38–713 | pp. 1–17 title/editorial matter; pp. 18–37 contents/divider; pp. 709 note excluded as editorial annotation; p. 714 scan metadata. |

The rare-label audit reviewed 26 source-image examples: 10 were confirmed as
author-body context and retained, while 16 isolated examples were excluded
from the strict table (five headings and 11 interviewer prompts). Embedded
quotations, verse examples, tables, dialogue, the author-signed chronology,
and the author’s responses remain. A separate endpoint review excluded 14
exact aside/margin or printed-folio IDs from strict training rows; these are
not used as replacements for corrected body endpoints. The strict exclusion
manifest contains 29 exact IDs in total, including the original heading and
interviewer cases. All such exclusions are strict-table-only; source fragments
remain in the all-clean OCR view. Image evidence is in
`books_scanned_strict_exclusions.jsonl` and
`books_scanned_layout_label_reviews.jsonl`.

## Review and audit files

- `scanned_body_ranges_*.jsonl`: page ranges, author/work attribution,
  confidence, printed-page evidence, excluded page roles, and sampled scan
  quality notes and `reviewer_model: gpt-6-luna`. Ranges must have high role
  and boundary confidence to enter this strict export.
- `books_scanned_lineage.jsonl`: one row per exported block with source-fragment
  provenance and the cleaned assembled-text hash.
- `books_scanned_exclusions.jsonl`: paragraph and fragment exclusions with a
  concrete reason; scan-cleanup audit rows retain the affected fragment IDs
  and evidence.
- `books_scanned_source_accounting.json`: per-book counts of cleaned source
  fragments, saved paragraph sources, audited removals, and retained layout
  fragments that were never part of the saved paragraph chains, plus the OCR
  layout-label counts of selected rows (footnote/reference labels are excluded).
- `books_scanned_quality.json`: per-book retained-line confidence means and
  weak-line counts for the OCR revision used. These are triage signals, not
  measured OCR accuracy; genuinely missing or damaged scan content remains
  unavailable.
- `books_scanned_layout_label_reviews.jsonl`: source-image reviews of rare
  selected layout labels. Layout labels alone do not decide inclusion: embedded
  verse, dialogue, tables, and a signed chronology were retained as author
  prose context; isolated headings and interviewer questions were excluded
  from this strict table only. The all-clean fragment view is preserved.
- `books_scanned_strict_exclusions.jsonl`: exact source IDs for the reviewed
  heading/interviewer and non-body endpoint paragraphs, with page-image hashes
  and explicit `strict_export_only` scope. These role exclusions do not delete
  all-clean OCR fragments.
- `books_scanned_upstream_fragment_reviews.jsonl`: image disposition for a
  tiny edge fragment excluded before paragraph assembly. The source image
  confirms it duplicates punctuation already retained in its neighboring body
  block; the original raw OCR and saved paragraph text are unchanged.
- OCR server re-identification and old-to-new fragment mappings remain
  separate revision sidecars; the lineage's `ocr_revision` identifies which
  revision supplied each row.
- `books_scanned_verification.json` records the saved-output check, including
  the exact reference-schema fingerprint, per-book source coverage, source
  fragment uniqueness, and unchanged raw OCR hashes. The final table contains
  40,838 rows from 20 books, with 24 volume-level work IDs and 8 authors.
- Server-v1 blocks are re-cleaned after recognition, before paragraph text is
  rebuilt from the finalized continuation chains. Manual folios are matched by
  PDF page, recognized line text, line box, and the original inspected image;
  old fragment IDs are never reused across OCR revisions. A folio absent from
  the server blocks is recorded as not applicable.

The p. 44 `上。` fragment in `bc87fbaddf81` is a source-image-verified derived
transcription because mobile OCR read `0` and server OCR omitted the tiny text.
Its typed lineage points to the original page fragment and image evidence;
the mobile/server raw page files remain unchanged.

OCR confidence measures recognition-model confidence, not character accuracy.
Sampled visual quality notes therefore remain separate from author-role
confidence. Missing printed scan content cannot be recovered by OCR.

The OCR comparison used PP-OCRv5 server detection/recognition on CPU for 659
pages across five books: a complete 653-page run for `94a5fb7e75e1`, plus six
selected pages in four other books. Those partial-book outputs retain reviewed
mobile page references elsewhere and substitute server output only on the
selected pages; paragraph lineage records whether a row came from mobile,
server, or both revisions. A separately reviewed source-image transcription
overlay may preserve a verified tiny printed fragment while leaving both raw
OCR revisions unchanged. The server model weights add roughly 165 MB locally;
no paid GPU service was used. Weak OCR lines remain possible and were not
corrected character by character; blurred, cropped, or absent printed content
cannot be restored.

## Character-level quality audit (2026-10-11)

A sampled audit of the `weak_ocr` flag population is in `WEAK_OCR_AUDIT.md`, with
per-line readings in `weak_ocr_review_20261011.jsonl` and scripts in
`scripts/weak_ocr_audit/`. Headline: 551 of 1,395 weak-flagged pages and 649 weak
lines were reviewed against 200-DPI page crops. Of the weak lines, 572 were
artifacts, 47 were character errors, 29 unclear, and 1 source damage. The
residual character error rate is 47 / 14,893 OCR lines = 0.3%, and 16 of the 47
errors landed in exported strict rows. Stratified to the full weak-page
population that is roughly 35 corrupted rows (95% 18-54) of 40,838, or 0.09%.
This is a floor: it covers only pages the heuristic flagged, so errors on
high-confidence lines were not measured.

Separately, a deterministic scan of the whole table found **77 strict rows ending
in a stray `O` or `0`** after a Chinese character or punctuation (55 are the
`...O` form). Source-image inspection on sampled pages shows these are scan
specks or dust rings sitting next to the sentence-final period, recognized by OCR
as a separate low-score line, not a misread glyph. They are listed in
`weak_ocr_trailing_O_rows_20261011.csv`. Removing them is a proposal only and has
**not** been applied; a human should confirm none of the 77 are legitimate
before any strip runs.

All of these judgments are model-assisted image inspection, not human
proofreading. The audit changed no existing table, sidecar, or export.

## Validation

Tables and sidecars for this dataset live on the data volume under
`/Volumes/EMC/stylevec_data` (`processed/bookraw/` and `manifests/bookraw/`); see
[README.md](README.md) for the full map. The scripts default to that location and
honour `STYLEVEC_DATA` for another checkout.

The exporter is `scripts/bookraw/export_scanned.py`; server blocks are prepared
with `scripts/bookraw/prepare_server_clean_fragments.py`. Run both with `uv run`
and the reviewed body-range manifest. The exporter refuses incomplete assembled
server books or an upgraded book without its own cleaned fragment map and audit.
`scripts/bookraw/verify_scanned_export.py` is a saved-output-only check: it does
not export or infer rows. It compares the saved Parquet schema against
`books_work_primary_strict.parquet`, records the reference file's SHA-256 and
mtime, checks all 20 books and source PDFs, uniqueness, language/role/length,
lineage text hashes, source accounting, and raw OCR page hashes. Run it after
the final export to write `books_scanned_verification.json`.
