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

## Post-hoc repair (`scripts/bookraw/repair_labeled_parquet.py`)

The source EPUBs referenced by `extract_labeled_bookraw.py` are no longer
present, so the extractor cannot be re-run. `repair_labeled_parquet.py`
repairs the existing exports from the evidence they already carry and rebuilds
every derived view from the repaired master. It is deterministic and
idempotent. Repairs, in the order applied:

1. **Whitespace injection.** `dom_blocks()` used `tag.get_text(" ")`, which
   inserts a space at every inline-tag boundary. That stranded spaces before
   fullwidth punctuation (`〔1〕 。`) and split Chinese words at inline-markup
   seams. 49,378 Chinese rows were corrected by collapsing only
   typographically impossible spacing. Spaces between two CJK characters are
   left alone, because those are genuine (`二十七日 晴。` in a Lu Xun diary).
   English rows are never touched.
2. **The `后记` role-state leak.** `classify_blocks()` set `role_state` from a
   role cue and never reset it, so a stray marker near a book's start
   relabelled everything after it. 76,795 blocks of Jin Yong novel prose were
   carried as `editorial_matter` at `high` confidence. Where the leaked region
   began within the first 5% of a work, those blocks are restored to
   `primary_text` with `role_evidence=role_state_leak_reverted_to_primary_text`
   and `role_source=post_hoc_repair` — 71,502 surviving rows across 12 Jin Yong
   novels. Works where the cue landed on a real, short, terminal afterword
   (Ba Jin's `寒夜`, `憩园`, and Jin Yong's `袁崇焕评传`) are left alone,
   because there the inherited label is correct. Navigation stubs such as
   `返回总目录` are skipped, since relabelling them would only swap one wrong
   label for another.
3. **The Zhonghua-classics heading state, and the editorial voice.** In the
   reviewed 中华经典藏书 edition, `classify_blocks` set `role_state =
   "commentary"` at every unlabelled `<h>` heading, so the poems and essays
   themselves were labelled commentary under `chapter_intro_before_body_marker`
   — 李白's 月下独酌 and 孟郊's 游子吟 among them — which left 289 authors with
   no `primary_text` at all. A block is restored to `primary_text` only when the
   reference marks it carries are glossed by the apparatus block of the *same*
   section, which commentary about a poem never does. Three bracket styles are
   recognised, each only when the section's notes use that same style: `〔n〕` in
   中华经典藏书, `[n]` elsewhere, and `(n)` where a note block *opens* with
   `(n)`. The test runs against *every* commentary evidence, not just the heading
   state, because `introductory_material_after_work_title` and
   `reviewed_publisher_commentary_style` also swallow poem bodies — the first
   lines of 关雎 carry the second of those. The EPUB's author-credit lines
   (`白居易 (2)`, `程颢 (2)`) are excluded by shape: their number resolves to a
   biographical note rather than a gloss, so they would otherwise pass. 6,756
   rows / 618,045 chars were restored, tagged
   `role_evidence=classics_heading_state_restored_to_primary_text`.

   Separately, every `commentary` block in this corpus comes from that one
   volume, whose modern editors — 胡平生、王国轩、陈晓芬、于天池、陈曦 — are
   named in the source filename. None of the commentary is by the classical
   author it discusses: 304 rows of 王国轩's prose on 王国维's 人间词话 were
   recorded as author=王国维, and the same rows also swept up the analysed
   work's own text. 7,414 commentary rows now carry `author=(现代编者)` and
   `role_evidence` suffixed `|attribution_to_modern_editor`, while the genuine
   primary text of those authors (王国维's 人间词话, 王永彬's 围炉夜话) is
   restored to them. A named editor is not guessed, because the volume credits
   five of them for one file.

   Blocks with no reference marks at all, in sections whose notes define none,
   remain unclassified. Most genuine commentary and many genuine poems fall in
   that bucket, and separating them needs a judgement this repair does not make.
4. **Wrong-author prefaces.** Sections that credit a different writer
   (`OSCAR WILDE: AN IDLER'S IMPRESSION by Edgar Saltus`) are not that
   catalog author's prose. 2,805 rows in `TELENY` were re-attributed to the
   writer the section names (Frank Harris, G. Bernard Shaw, Edgar Saltus).
5. **Language mismatches.** `lang` was copied from the catalog, not measured
   from the text. 4,829 rows marked `zh` that carry Latin prose with no CJK
   (Hu Shi's English writings) were corrected to `en`.
6. **Junk rows.** 3,961 rows whose text is only punctuation, only a page number,
   or only index furniture (`Ⅱ`, `331119 331220 ③ 340522 ①`, the Cyrillic sort
   keys of a notes index) were dropped, plus 37 Project Gutenberg licence
   paragraphs, which ship inside the book body and otherwise survive the role
   and length gates as apparent `primary_text`. The licence filter matches
   Gutenberg's own vocabulary (`Project Gutenberg`, `PGLAF`, `gutenberg.org`),
   so the three genuine literary mentions of the Gutenberg Bible survive.

Counts above are rows surviving the deletions, not pre-deletion totals. The
pass is deterministic and idempotent: running it twice produces byte-identical
files.

`books_work_passages_labeled.parquet` is byte-identical to the master by
construction: its filter keeps all six roles the extractor emits and no
`unknown` row survives extraction.

### Remaining limits after repair

These are known and are *not* fixed by the repair:

- **`primary_clean` is still 96% one weak heuristic.** 208,721 of 278,709 rows
  rest only on `catalog_author_and_*_text_language`, which infers a role from
  whether the catalog author name is CJK or Latin. Sampling those rows finds
  genuine novel prose but also table-of-contents entries, index fragments, and
  editor biographies filed under the author's name.
- **Author coverage is narrow.** `primary_clean` has 44 authors, `strict` only
  14, and 27.6% of `strict` is placeholder attribution (`多位作者`, `佚名`).
  `strict` is entirely Chinese classical text, so it is not a balanced
  benchmark.
- **A small number of works are entirely wrong-author.** The catalog names the
  biographer, not the subject, for some biographies (`MARK TWAIN A BIOGRAPHY`
  is Paine), so the `author` column describes the book, not necessarily its
  prose.
- **Cross-work duplicate text.** 1,385 texts in `primary_clean` appear under
  more than one `work_id`, including long verbatim reprints shared across
  works. Group splits by text, not by row, to avoid leakage.
- **Reference ids miss fullwidth markers.** `reference_ids` is built from
  ASCII `[n]` only, so 32,538 rows numbered `〔n〕` populate nothing, and ~74% of
  annotation references dangle in `passage_relations.jsonl`.
- **`section_title` is a running heading, not a per-work label.** In the
  中华经典藏书 volume it carries forward across consecutive poems, so 343 of
  3,500 works contain a section titled with a *different* poet — 李白's 春思
  contains a section called 杜甫, which is the next poem bleeding in. It must not
  be used to re-attribute a work's author, even though it looks like the obvious
  signal: doing so would have mislabelled 2,458 rows, including 蜀道难 as 高适
  and 登岳阳楼 as 王湾. The `author` column is wrong for those Tang poems
  (the catalog carries the previous entry forward), and that is left uncorrected
  here. Resolving it needs the poem's own title matched to a poet, which this
  data does not carry.

