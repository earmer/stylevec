# Prompt: download and clean a ~5 GB subset of CNNovel125K

## What this dataset is

[RyokoAI/CNNovel125K](https://huggingface.co/datasets/RyokoAI/CNNovel125K) is a
Chinese web-novel corpus, ~210.6 GB across 65 JSONL shards
(`cnnovel125k-00.jsonl` ... `cnnovel125k-64.jsonl`), roughly 3.2 GB and ~150k
books per shard. It is the T0 dataset for direction 1 (human author style) in
`automating.md`.

Each row is one JSON object per line:

```json
{"text": "<full book text>",
 "meta": {"subset": "cnnovel.ibiquw", "id": "100067", "q": 0.9,
          "lang": "zh_cn", "title": "为爱入局：嫁给秦先生", "author": "奥德萨"}}
```

`text` averages ~437k characters; the largest observed row is 10.77M characters.
`meta.author` is the identity label the ArcFace classifier needs. Verified by
ranged read of `cnnovel125k-00.jsonl` — first row is 183,391 bytes with
`text` length 66,558.

## Facts already established — do not re-derive these

- **Random shard = random book.** IDs are unordered within a shard and spread
  across the full table (shard 00 opens around id 100008, has 10065-100739 at an
  800 MB offset; shard 64 spans 1482-150554 in a 90 MB window). A single shard's
  ID range already covers all ~150k books. Random shard selection is therefore
  unbiased with respect to author, and no author-balancing is needed.
- **`meta.q` is useless.** Every value is the constant 0.9. It cannot be used for
  quality filtering, regardless of what the dataset card implies.
- **The dataset card self-describes as incomplete.**
- **HF viewer is unusable** — `/size` returns 0 rows, `/filter` hangs past 60s.
  Remote SQL filtering is not an option. Everything must be local.
- **Bandwidth ceiling ~11.7 MB/s** single connection. Multi-connection is
  *worse* (8 streams aggregated to 1.5 MB/s; `aria2c -x8` gave 12 MB/s).
  **Use a single sequential connection.** 5 GB ≈ 8 minutes.
- **`HF_HUB_DISABLE_XET=1` is required.** The installed `hf_xet` transport fails
  TLS against the xet-bridge-us CDN; plain HTTPS works. Set it before importing
  `huggingface_hub`. This bit the earlier downloads — see
  `/Volumes/EMC/stylevec_data/scripts/download_full.py`.
- **exFAT writes AppleDouble stubs.** A `._name` sidecar appears next to every
  file. Any directory listing or glob must skip names starting with `._` or `.`,
  or it will read 4 KB stubs that look like corrupt data. See the `data_files()`
  helper in `/Volumes/EMC/stylevec_data/scripts/audit.py`.

## Target

Land roughly 5 GB of raw shards, then produce a cleaned, deduplicated,
identity-labelled subset in `selected/`.

Budget guidance: **2 shards ≈ 6.4 GB**, which lands just over 5 GB and is the
simplest option that gives a statistically honest sample. One shard (~3.2 GB)
is under budget but halves the sample for ~8 minutes of download. Pick one,
state which, and note that 2 shards is the better choice for author-diversity
purposes. Do not download 3 shards (9.6 GB) without asking.

## Steps

### 1. Write the download script first

Follow the conventions in `/Volumes/EMC/stylevec_data/scripts/download_full.py`:
stdlib `urllib.request`, single sequential connection, sha256 per file, manifest
written to `manifests/`, progress to `logs/`. Put it at
`/Volumes/EMC/stylevec_data/scripts/download_cnnovel.py`.

Resolve shard sizes from the HF tree API before downloading, so the manifest
records the *expected* byte count and you can verify what actually landed:

```
https://huggingface.co/api/datasets/RyokoAI/CNNovel125K/tree/main?recursive=1
```

Shards must keep their upstream names — exFAT has no symlinks or hardlinks and
the convention is to preserve the HF filename exactly.

Target directory: `/Volumes/EMC/stylevec_data/raw/human_author/cnnovel125k/`

### 2. Download and verify

Report actual vs expected bytes and sha256 per shard. Confirm free space on the
volume first (exFAT, ~357 GiB was free as of 2026-09-28; re-check).

### 3. Stream-parse, never load a shard into memory

A shard is 3.2 GB with rows up to 10.77M characters. Read line by line.
`json.loads` on each complete line works — do not write a parser that tries to
handle the `meta`-at-end-of-row ordering as a special case. It is a single JSON
object per line and parses normally.

Malformed-line handling: count and log every line that fails to parse rather
than silently skipping. Report the count.

### 4. Filter what cannot serve the classifier

The goal is **author style**, so keep authors with multiple independent works.
The plan's rule is "多作者多样本优先" — prioritize authors with many independent
texts — and `automating.md:252` is explicit: 100 segments cut from one long
article are not 100 independent style observations.

Drop, with counts reported for each:
- rows missing or empty `meta.author`, or where author is a placeholder
- duplicate `meta.id` (keep first, log the rest)
- non-`zh_cn` `lang`
- `text` shorter than a threshold you choose and state (suggest 100k characters,
  since a full book is the unit here; report the length distribution first so the
  threshold is a decision you can defend)
- any row whose `text` is truncated or fails a structural sanity check

**Do not** filter on `meta.q` — it is a constant.

### 5. Build the author index

Report: total books parsed, books kept, distinct authors surviving, and the
distribution of works-per-author. This distribution is the actual deliverable —
it decides whether the subset can support identity training at all.

If most surviving authors have exactly one work, the subset is unusable for
ArcFace identity supervision and you must say so plainly rather than producing
the parquet anyway. That would be the single most important finding.

### 6. Write outputs

- `selected/cnnovel125k.parquet` or `.jsonl` — cleaned rows with at minimum
  `identity_id` (= `meta.author`), `document_id` (= `meta.id`), `title`, `text`.
  The field list at `automating.md:217-225` is the intended schema: add
  `dataset`, `subset`, and an identity-confidence note rather than inventing
  fields.
- `manifests/cnnovel125k_audit.md` — matching the existing `*_audit.md` format
  used by the other T0 datasets
- `manifests/cnnovel125k_download.json` — shard names, byte counts, sha256

### 7. Apply the leakage rule from the plan

`automating.md:256` and the docs/bookraw precedent both require: split before
slicing. All works by one author stay in the same partition, or the test set
leaks author identity. If you emit any train/test split alongside the parquet,
group by `identity_id`. If you do not emit a split, say so.

## Rules

- Do not delete or modify anything already in `raw/` or `selected/`. This dataset
  has never been downloaded before — `raw/human_author/cnnovel125k/` does not
  exist yet. If you find it already populated, stop and report.
- Never write to the HF cache as if it were the corpus. Raw downloads go to
  `raw/`, cleaned output to `selected/`.
- Report the download size, the parse success/failure counts, the filter
  decision counts, and the author distribution. Those numbers are the deliverable;
  a parquet path alone is not.
- If the works-per-author distribution is too thin for identity training, say so
  and stop there. Do not pad the result with single-work authors to make the
  count look better.
- `automating.md:17` flags that CNNovel125K's quality fields cannot be trusted
  for filtering. Any quality judgment you make beyond the structural checks above
  is your own, and must be labeled as such.