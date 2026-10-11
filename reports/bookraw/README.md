# Bookraw (scanned and EPUB corpora)

This directory holds **code and documentation only**. The datasets themselves
live on the external volume.

## Where the data lives

Root: `/Volumes/EMC/stylevec_data` (symlinked as `stylevec_data/` in the repo).

| What | Path |
|---|---|
| EPUB sources (20 books + 1 mobi) | `raw/human_author/bookraw_epub/` |
| Scanned PDFs (contemporary authors) | `raw/human_author/contemporary_authors/` |
| Labeled book tables (post-repair) | `processed/bookraw/` |
| Scanned-book tables | `processed/bookraw/books_scanned.parquet` |
| OCR revisions | `processed/pdf_ocr-v1/`, `pdf_ocr-reviewed-v1/`, `pdf_ocr-server-v1/` |
| Sidecars, lineage, audits, reviews | `manifests/bookraw/` |
| Legacy pre-repair parquets (epub20) | `manifests/bookraw/epub20_legacy/` |
| Pipeline scripts | `scripts/bookraw/` |
| OCR quality audit scripts | `scripts/weak_ocr_audit/` |

`/Volumes/EMC/stylevec_data/processed/bookraw/` is canonical. An earlier copy of
the same tables lived in `reports/bookraw/` and was stale — it predated the
repair pass described in the annotation schema. It has been removed here so
there is one source.

## What lives here

- `scripts/bookraw/` — extraction, the post-hoc repair pass, the scanned-book
  exporter, and the read-only verifier.
- `scripts/weak_ocr_audit/` — sampled character-level OCR audit.
- `docs/bookraw/ANNOTATION_SCHEMA.md` — **the source of truth** for what a row
  means, the repair that was applied, and the limits that remain.
- `reports/bookraw/` — narrative documentation only (see below).

## Documentation kept here

These are prose and are tracked in git, so they are duplicated from the volume
deliberately. The `.md` and small `.json` files stay; the large `.jsonl`
sidecars and all `.parquet` tables are gitignored and now live only on the
volume.

- `README_books_scanned.md` — the 20-book page-range table, field semantics,
  sidecar index, and the character-level audit results.
- `TECHNICAL_REPORT_books_scanned.md` — OCR pipeline, performance experiments,
  and boundary review.
- `WEAK_OCR_AUDIT.md` — sampled OCR character-error audit and the stray `O`/`0`
  finding.

## Regenerating

All scripts are run with `uv run`. They take explicit `--input` / `--output`
paths; the defaults point at the volume layout above. See each script's
`--help`.

`verify_scanned_export.py` is read-only: it checks a saved Parquet and its
sidecars and never re-exports.