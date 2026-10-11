#!/usr/bin/env python3
"""Add conservative source and translation metadata to the EPUB work catalog.

This is an additive export. It never modifies the source catalogs or passage parquet.
Only explicit translation statements in catalog notes produce a translation label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Sources and outputs live on the data volume; see reports/bookraw/README.md.
DATA_ROOT = Path(os.environ.get("STYLEVEC_DATA", "/Volumes/EMC/stylevec_data"))
CATALOG_DIR = DATA_ROOT / "manifests/bookraw/epub20_legacy"
ACTIVE_DATA_ROOT = DATA_ROOT / "raw/human_author/bookraw_epub"
CATALOGS = [
    CATALOG_DIR / "work_catalog_english.jsonl",
    CATALOG_DIR / "work_catalog_chinese_a.jsonl",
    CATALOG_DIR / "work_catalog_chinese_b.jsonl",
]
WORK_OUTPUT = DATA_ROOT / "manifests/bookraw/works_catalog_enriched.jsonl"
SOURCE_OUTPUT = DATA_ROOT / "manifests/bookraw/epub_sources.jsonl"
TRANSLATION_RE = re.compile(r"\btranslation by\s+([^.;,]+)", re.I)
TITLE_ROLE_RULES = [
    ("annotation", re.compile(r"^(?:注释条目|注释|注解|校注|译注|集注)")),
    ("commentary", re.compile(r"(?:赏析|评析|鉴赏|解读|评论)$|^(?:赏析|评析|作品鉴赏|导读)")),
    (
        "editorial_matter",
        re.compile(r"^(?:编者按|出版说明|译者序|译者前言|前言|序言|导言|引言|后记)$"),
    ),
]


def read_catalogs(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["catalog_source"] = path.name
                row["catalog_line"] = line_no
                rows.append(row)
    return rows


def stable_work_id(row: dict) -> str:
    key = "\0".join(str(row.get(k, "")) for k in ("source_file", "work_title", "author", "source_href"))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def provenance_hints(source_file: str) -> dict[str, str]:
    lowered = source_file.casefold()
    content_source = "project_gutenberg" if "project gutenberg" in lowered else "unspecified"
    if "anna" in lowered and "archive" in lowered:
        acquisition_source = "annas_archive"
    elif "z-library" in lowered or "z-lib" in lowered or "1lib" in lowered:
        acquisition_source = "z_library_family"
    else:
        acquisition_source = "unspecified"
    return {
        "content_source": content_source,
        "acquisition_source": acquisition_source,
        "provenance_evidence": "source filename; not independently verified",
    }


def enrich_work(row: dict) -> dict:
    match = TRANSLATION_RE.search(str(row.get("note", "")))
    title = str(row.get("work_title", "")).strip()
    text_type = "translation" if match else "unknown"
    confidence = "high" if match else "unknown"
    evidence = "explicit catalog note" if match else "not established at work level"
    if not match:
        for role, pattern in TITLE_ROLE_RULES:
            if pattern.search(title):
                text_type = role
                confidence = "high" if role == "annotation" else "medium"
                evidence = f"work title matched {role} cue"
                break
    enriched = dict(row)
    enriched.update(
        {
            "work_id": stable_work_id(row),
            "text_type": text_type,
            "text_type_confidence": confidence,
            "text_type_evidence": evidence,
            "translator": match.group(1).strip() if match else None,
            "original_language": None,
            "translation_pair_id": None,
            "annotation_status": "work_level_only; passage_roles_unreviewed",
        }
    )
    return enriched


def source_records(rows: list[dict]) -> list[dict]:
    sources: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("source_file", ""))
        if name not in sources:
            path = ACTIVE_DATA_ROOT / name
            info = {
                "source_file": name,
                "source_format": path.suffix.lower().lstrip(".") or "unknown",
                **provenance_hints(name),
                "source_exists": path.is_file(),
                "source_sha256": None,
                "catalog_record_count": 0,
            }
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                info["source_sha256"] = digest.hexdigest()
            sources[name] = info
        sources[name]["catalog_record_count"] += 1
    return [sources[name] for name in sorted(sources)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--work-output", type=Path, default=WORK_OUTPUT)
    parser.add_argument("--source-output", type=Path, default=SOURCE_OUTPUT)
    args = parser.parse_args()
    global ACTIVE_DATA_ROOT
    ACTIVE_DATA_ROOT = args.data_root
    rows = read_catalogs([args.data_root / path.name for path in CATALOGS])
    enriched = [enrich_work(row) for row in rows]
    sources = source_records(rows)
    args.work_output.parent.mkdir(parents=True, exist_ok=True)
    args.source_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.work_output, enriched)
    write_jsonl(args.source_output, sources)
    roles = Counter(row["text_type"] for row in enriched)
    print(
        f"Wrote {len(enriched)} enriched work records ({dict(roles)}) to {args.work_output}; "
        f"wrote {len(sources)} EPUB source records to {args.source_output}."
    )


if __name__ == "__main__":
    main()
