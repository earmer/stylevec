# Bookraw source and text-role annotations

## Current data and provenance

The work catalog is assembled from `work_catalog_english.jsonl`,
`work_catalog_chinese_a.jsonl`, and `work_catalog_chinese_b.jsonl`. The current
passage extraction is `books_work_segments.parquet`; its catalog is
`works_catalog.jsonl`. The richer passage table has a work-level start pointer
(`source_href`) but does not retain the XHTML paragraph positions used to make
each passage. It also merges adjacent paragraphs while chunking.

Run `python3 scripts/bookraw/enrich_work_catalog.py` from the repository root
to create the source and work-level metadata sidecars under `reports/bookraw/`.
Then run `python3 scripts/bookraw/extract_labeled_bookraw.py` to re-read all
local EPUBs and create:

- `books_work_blocks_labeled.parquet`: one source paragraph/block per row with
  its label, evidence, source EPUB, spine file, DOM position, and catalog row.
- `books_work_passages_labeled.parquet`: role-labeled blocks, excluding
  `unknown`.
- `books_work_primary_clean.parquet`: primary-text blocks with medium/high
  role and boundary confidence, filtered to at least 30 Chinese or 80 English
  characters.
- `books_work_primary_strict.parquet`: the same length-filtered subset with
  both role and boundary confidence required to be high.
- `books_work_translations.parquet`: translated blocks kept separate from
  primary text.
- `works_catalog_labeled.jsonl` and `work_boundary_manifest.jsonl`: per-work
  extraction counts, role counts, and resolved-boundary evidence.
- `passage_relations.jsonl`: numbered-note links and local original/translation
  links with their alignment method and confidence.

Provenance hints are marked as filename-derived, not verified publisher or
license metadata. The work catalog retains translation and boundary evidence.
Passage roles are rule-labeled with confidence and evidence; most general
primary-text labels use a medium-confidence author/language heuristic. The
strict primary subset limits use to high-confidence roles and boundaries.

## Passage-level role vocabulary

Labels are assigned before any new chunks are formed, while EPUB block and
section structure are still available. Each row represents one paragraph/block
and receives one role:

| `text_role` | Meaning |
| --- | --- |
| `primary_text` | Work text in its source language, where verified |
| `translation` | A translated rendering of identifiable source text |
| `annotation` | Numbered/inline explanatory notes or glosses |
| `commentary` | Appreciation, criticism, analysis, or modern paraphrase |
| `editorial_matter` | Preface, introduction, editor biography, or other framing |
| `paratext` | Contents, headings, publication data, or navigation material |
| `unknown` | Evidence does not support a reliable role |

Each block carries `role_confidence`, `role_source`, and `role_evidence`.
`role_source` distinguishes EPUB heading rules, source-specific CSS rules,
catalog metadata, note markers, and author/language heuristics. Heuristics
should not silently convert unsupported text into primary text.

## Relationships and paired text

Use stable `source_block_id` values based on source file, spine item, DOM block
identity, and role; do not use chunk index alone. Represent relationships as
separate rows in `passage_relations.jsonl`, with `from_passage_id`,
`to_passage_id`, `relation_type`, `alignment_method`, and confidence. The
passage IDs refer to source block IDs in this paragraph-level extraction. Supported relations
are `translates`, `annotates`, `comments_on`, and `quotes`. A relation may be
many-to-many; do not assume adjacent chunks or equal passage indexes are
translations of each other. Only create translation pairs when EPUB markup,
clear source/translation headings, or manual review identifies both sides.

## Known contamination and limits

The legacy extractor gathers paragraph-like blocks between work boundaries,
then merges them into size-limited chunks. This new pass re-reads the original
EPUBs and saves one block per row, resolving empty cover anchors to nearby exact
work-title headings where possible. It uses explicit role headings such as
`【注释】` and `【译文】`; for the reviewed Zhonghua Classics EPUB it also uses
the edition's body classes (`kindle-cn-hei1`, `x17`, and
`kindle-cn-para-left1`) and commentary class (`kindle-cn-kai1`). These CSS rules
are edition-specific. Other author/language-based primary labels are medium
confidence and must be filtered by confidence for strict use.

The boundary manifest makes unresolved starts visible. `六韬`, `战国策`, and
`楚辞` now retain distinct commentary, primary text, note, and translation
blocks where the EPUB supplies those structures. Their old Parquet rows remain
untouched. Pair relations are evidence-based: numbered notes link by printed
reference IDs; translated blocks link to recent primary blocks within the
same section and are marked medium confidence. This is not sentence-level
alignment.

The old `books_cleaned.parquet` is still consumed by
`expandtraining/2606/src/import_bookraw.py` and has no role fields. Any later
training use of the new annotations must explicitly select eligible roles;
unknown, editorial matter, notes, and commentary should not enter author-style
training by default.
