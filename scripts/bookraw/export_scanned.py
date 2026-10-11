#!/usr/bin/env python3
"""Export reviewed scanned-PDF paragraphs to the books_work_primary_strict schema."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path("/Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/clean_export")
SCHEMA_COLUMNS = [
    "text", "author", "work", "work_id", "genre", "lang", "text_role",
    "role_confidence", "role_evidence", "role_source", "boundary_confidence",
    "boundary_evidence", "section_id", "section_title", "source_file",
    "source_href", "source_spine_file", "source_spine_index",
    "source_dom_position", "source_block_id", "source_tag", "source_html_id",
    "source_html_classes", "source_epub_type", "catalog_source", "catalog_line",
    "reference_ids",
]
STRING_COLUMNS = {
    "text", "author", "work", "work_id", "genre", "lang", "text_role",
    "role_confidence", "role_evidence", "role_source", "boundary_confidence",
    "boundary_evidence", "section_id", "section_title", "source_file",
    "source_href", "source_spine_file", "source_block_id", "source_tag",
    "source_html_id", "source_epub_type", "catalog_source",
}
LIST_COLUMNS = {"source_html_classes", "reference_ids"}
INT_COLUMNS = {"source_spine_index", "source_dom_position", "catalog_line"}
SCANNED_SCHEMA = pa.schema(
    [
        pa.field(name, pa.large_string()) if name in STRING_COLUMNS else
        pa.field(name, pa.list_(pa.string())) if name in LIST_COLUMNS else
        pa.field(name, pa.int64())
        for name in SCHEMA_COLUMNS
    ]
)
OCR_NON_BODY_LABELS = {"footnote", "vision_footnote", "reference_content"}
ID_PAGE_RE = re.compile(r"^[^:]+:(\d+):")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fragment_ids(paragraph: dict[str, Any]) -> list[str]:
    ids = [str(source.get("id")) for source in paragraph.get("sources", []) if source.get("id")]
    if not ids and paragraph.get("id"):
        ids = [str(paragraph["id"])]
    return ids


def source_pages(paragraph: dict[str, Any]) -> list[int]:
    pages = {int(source["page"]) for source in paragraph.get("sources", []) if source.get("page") is not None}
    if not pages and paragraph.get("id"):
        match = ID_PAGE_RE.match(str(paragraph["id"]))
        if match:
            pages.add(int(match.group(1)))
    return sorted(pages)


def stable_id(*parts: str) -> str:
    value = "\0".join(parts)
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def join_fragment_texts(texts: list[str]) -> str:
    cjk = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]")
    result = ""
    for value in texts:
        value = value.strip()
        if not value:
            continue
        if result and not (cjk.search(result[-1]) or cjk.search(value[0])):
            result += " "
        result += value
    return result


def reassemble_cleaned_paragraphs(
    paragraphs: list[dict[str, Any]], fragments_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Apply cleaned fragment text while preserving the saved paragraph chains."""
    output: list[dict[str, Any]] = []
    expected = {
        str(source["id"])
        for paragraph in paragraphs
        for source in paragraph.get("sources", [])
        if source.get("id")
    }
    seen: list[str] = []
    for paragraph in paragraphs:
        sources = []
        texts = []
        for source in paragraph.get("sources", []):
            source_id = str(source.get("id", ""))
            fragment = fragments_by_id.get(source_id)
            if fragment is None:
                continue
            updated_source = copy.deepcopy(source)
            if fragment.get("box") is not None:
                updated_source["box"] = fragment["box"]
            if fragment.get("page") is not None:
                updated_source["page"] = int(fragment["page"])
            sources.append(updated_source)
            texts.append(str(fragment.get("text") or ""))
            seen.append(source_id)
        if not sources:
            continue
        rebuilt = copy.deepcopy(paragraph)
        rebuilt["sources"] = sources
        rebuilt["text"] = join_fragment_texts(texts)
        if rebuilt["text"].strip():
            output.append(rebuilt)
    expected_retained = expected.intersection(fragments_by_id)
    if len(seen) != len(set(seen)):
        raise ValueError("Cleaned source fragment appears in multiple assembled paragraphs")
    if set(seen) != expected_retained:
        raise ValueError("Cleaned paragraph reassembly lost or duplicated source fragments")
    return output


def unresolved_new_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    result: set[str] = set()
    id_keys = {
        "tail_id", "head_id", "new_tail_id", "new_head_id", "tail_new_id", "head_new_id",
        "candidate_tail_id", "candidate_head_id", "candidate_new_ids", "candidate_ids",
        "new_source_ids", "mapped_new_ids", "new_fragment_ids", "new_ids", "source_ids",
    }

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key in id_keys:
                result.update(str(item) for item in value if isinstance(item, str) and ":" in item)
            else:
                for child in value:
                    collect(child, key)
        elif key in id_keys and isinstance(value, str) and ":" in value:
            result.add(value)

    for row in read_jsonl(path):
        if row.get("status") == "review":
            collect(row)
    return result


def review_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    id_keys = {
        "tail_id", "head_id", "new_tail_id", "new_head_id", "tail_new_id", "head_new_id",
        "candidate_tail_id", "candidate_head_id", "candidate_new_ids", "candidate_ids",
        "new_source_ids", "mapped_new_ids", "new_fragment_ids", "new_ids", "source_ids",
    }

    def collect(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key))
        elif isinstance(value, list):
            if key in id_keys:
                ids.update(str(item) for item in value if isinstance(item, str) and ":" in item)
            else:
                for child in value:
                    collect(child, key)
        elif key in id_keys and isinstance(value, str) and ":" in value:
            ids.add(value)

    for row in rows:
        if row.get("status") == "review":
            collect(row)
    return ids


def selected_server_pages(selection: dict[str, Any], server_verification: dict[str, Any]) -> dict[str, set[int]]:
    """Return the pages supplied by the completed server revision per book."""
    result: dict[str, set[int]] = {}
    full_ids = {str(row["book"]) for row in selection.get("full_books", [])}
    for row in selection.get("full_books", []):
        result[str(row["book"])] = set(range(1, int(row["pages"]) + 1))
    partial = {str(row["book"]): row for row in server_verification.get("partial_books", [])}
    for row in selection.get("selected_pages", []):
        book_id = str(row["book"])
        if book_id in full_ids:
            continue
        if book_id not in partial:
            raise ValueError(f"Selected server page has no completed assembled book entry: {book_id}")
        result.setdefault(book_id, set()).add(int(row["page"]))
    expected_partial_ids = set(result) - full_ids
    if expected_partial_ids != set(partial):
        raise ValueError(
            f"Server selection and completed partial-book manifests differ: "
            f"selected={sorted(expected_partial_ids)} completed={sorted(partial)}"
        )
    for book_id, row in partial.items():
        if row.get("page_reference_coverage") is not True or row.get("paragraph_fragment_coverage_verified") is not True:
            raise ValueError(f"Partial server book {book_id} failed assembled coverage checks")
        selected = {int(page) for page in row.get("server_reprocessed_pages", [])}
        if selected != result[book_id]:
            raise ValueError(f"Partial server page set mismatch for {book_id}: {selected} != {result[book_id]}")
    return result


def paragraph_ocr_revision(paragraph: dict[str, Any], book_id: str, server_pages: dict[str, set[int]]) -> str:
    pages = set(source_pages(paragraph))
    pages_used = pages.intersection(server_pages.get(book_id, set()))
    if not pages_used:
        return "pdf_ocr-reviewed-v1"
    if pages_used == pages:
        return "pdf_ocr-server-v1"
    return "pdf_ocr-reviewed-v1+pdf_ocr-server-v1"


def load_cleaned_fragment_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cleaned fragment file: {path}")
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fragment = json.loads(line)
        fragment_id = str(fragment.get("id", ""))
        if not fragment_id or fragment_id in result:
            raise ValueError(f"Missing or duplicate cleaned fragment ID in {path}: {fragment_id}")
        result[fragment_id] = fragment
    return result


def find_body_range(pages: list[int], ranges: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not pages:
        return None
    matches = [
        item for item in ranges
        if item.get("role") == "primary_text"
        and all(int(item["start_page"]) <= page <= int(item["end_page"]) for page in pages)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def high_confidence_range(body_range: dict[str, Any]) -> bool:
    return (
        body_range.get("role_confidence") == "high"
        and body_range.get("boundary_confidence") == "high"
        and body_range.get("reviewer_model") == "gpt-6-luna"
    )


def selection_reason(
    paragraph: dict[str, Any],
    ranges: list[dict[str, Any]],
    role_ranges: list[dict[str, Any]],
    unresolved: set[str],
    strict_exclusions: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str, dict[str, Any] | None]:
    ids = fragment_ids(paragraph)
    manually_excluded = [source_id for source_id in ids if strict_exclusions and source_id in strict_exclusions]
    if manually_excluded:
        review = strict_exclusions[manually_excluded[0]]
        return None, f"manual_source_image_role_exclusion:{review['reason_code']}", review
    if any(source_id in unresolved for source_id in ids):
        return None, "boundary_reidentification_requires_review", None
    if paragraph.get("label") in OCR_NON_BODY_LABELS:
        return None, f"ocr_layout_role:{paragraph.get('label')}", None
    pages = source_pages(paragraph)
    body_range = find_body_range(pages, ranges)
    if body_range is None:
        if not pages:
            return None, "missing_pdf_page_provenance", None
        explicit = [
            item for item in role_ranges
            if all(int(item["start_page"]) <= page <= int(item["end_page"]) for page in pages)
        ]
        if len(explicit) == 1:
            return None, f"reviewed_role_exclusion:{explicit[0].get('role')}:{explicit[0].get('reason', '')}", explicit[0]
        return None, "outside_one_reviewed_primary_body_range", None
    strict_exclusion = body_range.get("strict_exclusion_reason")
    if strict_exclusion or body_range.get("strict_export_eligible") is False:
        strict_exclusion = strict_exclusion or "reviewed_range_not_strict_export_eligible"
        return None, f"reviewed_primary_text_excluded:{strict_exclusion}", body_range
    if not high_confidence_range(body_range):
        return None, "body_range_not_high_confidence", body_range
    text = str(paragraph.get("text") or "").strip()
    if len(text) < 30:
        return None, "strict_chinese_minimum_length_under_30", body_range
    return body_range, "selected", body_range


def strict_row(
    paragraph: dict[str, Any],
    body_range: dict[str, Any],
    book: dict[str, Any],
    source_id: str,
    pages: list[int],
    paragraph_position: int,
    range_file: str,
) -> dict[str, Any]:
    author = str(body_range["author"])
    work = str(body_range["work_title"])
    source_file = str(book["source_pdf"])
    source_href = f"pdf-page:{book['book_id']}:{pages[0]}"
    work_href = f"pdf-page:{book['book_id']}:{body_range['start_page']}"
    work_id = stable_id(source_file, work, author, work_href)
    range_id = stable_id(str(book["book_id"]), str(body_range["start_page"]), str(body_range["end_page"]), work)
    return {
        "text": str(paragraph["text"]).strip(),
        "author": author,
        "work": work,
        "work_id": work_id,
        "genre": str(body_range.get("genre") or ""),
        "lang": "zh",
        "text_role": "primary_text",
        "role_confidence": "high",
        "role_evidence": str(body_range["evidence"]),
        "role_source": "gpt_6_luna_reviewed_pdf_page_range",
        "boundary_confidence": "high",
        "boundary_evidence": "source-page layout block; cross-page boundary review is represented by the boundary re-identification audit",
        "section_id": range_id,
        "section_title": work,
        "source_file": source_file,
        "source_href": source_href,
        "source_spine_file": None,
        # This column records the one-based PDF page for scans; source_href and
        # role_evidence identify it as PDF page provenance, not EPUB spine order.
        "source_spine_index": pages[0],
        # For page-image sources, this is the zero-based position in the
        # reviewed paragraphs.json sequence, not an EPUB DOM offset.
        "source_dom_position": paragraph_position,
        "source_block_id": source_id,
        "source_tag": None,
        "source_html_id": None,
        "source_html_classes": [],
        "source_epub_type": None,
        "catalog_source": range_file,
        "catalog_line": None,
        "reference_ids": [],
    }


def export(
    input_root: Path,
    ranges_path: Path,
    output_path: Path,
    audit_path: Path,
    lineage_path: Path,
    boundary_path: Path | None = None,
    cleaning_path: Path | None = None,
    strict_exclusion_path: Path | None = None,
    server_root: Path | None = None,
    server_selection_path: Path | None = None,
    cleaned_fragments_root: Path | None = None,
    server_cleaned_fragments_root: Path | None = None,
    server_cleaning_path: Path | None = None,
    verified_transcription_path: Path | None = None,
    book_ids: set[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force to replace it")
    all_ranges = read_jsonl(ranges_path)
    by_book: dict[str, list[dict[str, Any]]] = {}
    for row in all_ranges:
        by_book.setdefault(str(row["book_id"]), []).extend(row.get("body_ranges", []))
    books = {str(row["book_id"]): row for row in all_ranges}
    if book_ids:
        missing = book_ids - books.keys()
        if missing:
            raise ValueError(f"No body-range review for requested books: {sorted(missing)}")
    unresolved = unresolved_new_ids(boundary_path)
    verified_transcriptions: dict[str, dict[str, Any]] = {}
    if verified_transcription_path is not None:
        for row in read_jsonl(verified_transcription_path):
            derived_id = str(row.get("derived_fragment_id", ""))
            if not derived_id or derived_id in verified_transcriptions:
                raise ValueError(f"Missing or duplicate verified transcription ID: {derived_id}")
            if row.get("kind") != "source_image_verified_transcription" or row.get("reviewer_model") != "gpt-6-luna" or row.get("confidence") != "high":
                raise ValueError(f"Verified transcription lacks required provenance: {derived_id}")
            for path_key, hash_key in (("source_pdf", "source_pdf_sha256"), ("source_page_image", "source_page_image_sha256"), ("source_crop_image", "source_crop_image_sha256")):
                evidence_path = Path(str(row.get(path_key, "")))
                if not evidence_path.is_file() or hashlib.sha256(evidence_path.read_bytes()).hexdigest() != row.get(hash_key):
                    raise ValueError(f"Verified transcription evidence is missing or changed: {evidence_path}")
            verified_transcriptions[derived_id] = row
    strict_exclusions: dict[str, dict[str, Any]] = {}
    if strict_exclusion_path is not None:
        for row in read_jsonl(strict_exclusion_path):
            if row.get("decision") != "exclude_from_strict" or row.get("reviewer_model") != "gpt-6-luna":
                raise ValueError(f"Manual strict exclusion is not source-image reviewed: {row}")
            images = row.get("inspected_images", [])
            if not images:
                raise ValueError(f"Manual strict exclusion has no inspected images: {row}")
            for evidence in images:
                image_path = Path(str(evidence.get("path") or ""))
                if not image_path.is_file():
                    raise ValueError(f"Manual strict-exclusion image is missing: {image_path}")
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if evidence.get("sha256") != image_hash:
                    raise ValueError(f"Manual strict-exclusion image changed: {image_path}")
            for source_id in row.get("source_fragment_ids", []):
                source_id = str(source_id)
                if source_id in strict_exclusions:
                    raise ValueError(f"Duplicate manual strict exclusion: {source_id}")
                strict_exclusions[source_id] = row
    server_books: dict[str, dict[str, Any]] = {}
    server_pages_by_book: dict[str, set[int]] = {}
    if server_selection_path is not None:
        if server_root is None:
            raise ValueError("--server-root is required with --server-selection")
        selection = json.loads(server_selection_path.read_text(encoding="utf-8"))
        server_verification_path = server_root / "verification.json"
        if not server_verification_path.is_file():
            raise FileNotFoundError(f"Missing completed server verification: {server_verification_path}")
        server_verification = json.loads(server_verification_path.read_text(encoding="utf-8"))
        if server_verification.get("complete") is not True:
            raise ValueError("Server revision is not marked complete")
        server_pages_by_book = selected_server_pages(selection, server_verification)
        full_ids = {str(item["book"]) for item in selection.get("full_books", [])}
        verified_books = {
            str(row["book"]): row
            for row in [*server_verification.get("full_books", []), *server_verification.get("partial_books", [])]
        }
        for selected_id, pages in server_pages_by_book.items():
            selected = verified_books.get(selected_id)
            if selected is None:
                raise ValueError(f"Server selection lacks verification entry for {selected_id}")
            expected_pages = int(selected.get("expected_pages", selected.get("pages", 0)))
            server_books[selected_id] = {
                "book": selected_id,
                "pages": expected_pages,
                "server_pages": sorted(pages),
                "full_book_replacement": selected_id in full_ids,
            }
        if server_books and (server_cleaned_fragments_root is None or server_cleaning_path is None):
            raise ValueError(
                "Full-book server revisions require both --server-cleaned-fragments-root "
                "and --server-cleaning-audit"
            )
        for selected_id, selected in server_books.items():
            server_dir = server_root / selected_id
            required = ["complete.json", "paragraphs.json", "continuation.json", "review.json"]
            missing_files = [name for name in required if not (server_dir / name).is_file()]
            if missing_files:
                raise ValueError(
                    f"Selected server book {selected_id} is incomplete; missing {missing_files}"
                )
            complete = json.loads((server_dir / "complete.json").read_text(encoding="utf-8"))
            if complete.get("status") not in (None, "complete", "completed", "success"):
                raise ValueError(f"Selected server book {selected_id} has noncomplete marker: {complete}")
            expected_pages = int(selected["pages"])
            complete_pages = complete.get("pages", complete.get("page_count"))
            if complete_pages is not None and int(complete_pages) != expected_pages:
                raise ValueError(
                    f"Selected server book {selected_id} completed {complete_pages} pages; expected {expected_pages}"
                )
            page_dir = server_dir / "pages"
            page_files = [
                path for path in page_dir.glob("*.json")
                if path.stem.isdigit()
            ] if page_dir.is_dir() else []
            if len(page_files) != expected_pages:
                raise ValueError(
                    f"Selected server book {selected_id} has {len(page_files)} page JSONs; expected {expected_pages}"
                )
            for page_number in selected["server_pages"]:
                page_path = page_dir / f"{page_number:05d}.json"
                if not page_path.is_file():
                    raise ValueError(f"Selected server page is missing for {selected_id}: {page_path}")
        for selected_id in server_books:
            current_paragraphs = json.loads((server_root / selected_id / "paragraphs.json").read_text(encoding="utf-8"))
            current_ids = {
                str(source.get("id"))
                for paragraph in current_paragraphs
                for source in paragraph.get("sources", [])
                if source.get("id")
            }
            stale = [
                source_id
                for source_id in strict_exclusions
                if source_id.startswith(f"{selected_id}:") and source_id not in current_ids
            ]
            if stale:
                raise ValueError(
                    f"Image-reviewed strict exclusion IDs became stale after server assembly for {selected_id}: {stale[:8]}"
                )
    base_cleaning_audit = read_jsonl(cleaning_path) if cleaning_path is not None else []
    server_cleaning_audit = read_jsonl(server_cleaning_path) if server_cleaning_path is not None else []
    server_book_ids = set(server_books)
    cleaning_audit = [
        row for row in base_cleaning_audit
        if str(row.get("book_id", row.get("book", ""))) not in server_book_ids
    ] + server_cleaning_audit
    audit_ids_by_fragment: dict[str, list[str]] = {}
    cleaning_audit_ids: list[str] = []
    for row in cleaning_audit:
        if row.get("status") == "review_required":
            raise ValueError(f"Unresolved scan cleanup review remains for {row.get('book') or row.get('book_id')}: {row}")
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
        audit_id = str(row.get("audit_id") or stable_id("cleanup-audit", encoded))
        cleaning_audit_ids.append(audit_id)
        for fragment_id in [
            *row.get("removed_source_ids", []), *row.get("modified_source_ids", []),
            *row.get("derived_source_ids", []),
        ]:
            audit_ids_by_fragment.setdefault(str(fragment_id), []).append(audit_id)
    strict_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    source_accounting: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_source_blocks: set[str] = set()
    selected_books: set[str] = set()
    seen_source_fragments: set[str] = set()

    for book_id, book in books.items():
        if book_ids and book_id not in book_ids:
            continue
        book_dir = input_root / book_id
        if book_id in server_books:
            assert server_root is not None
            paragraph_path = server_root / book_id / "paragraphs.json"
            ocr_revision = (
                "pdf_ocr-server-v1"
                if server_books[book_id]["full_book_replacement"]
                else "pdf_ocr-reviewed-v1+pdf_ocr-server-v1"
            )
            unresolved.update(review_ids(server_root / book_id / "review.json"))
        else:
            paragraph_path = book_dir / "paragraphs.json"
            ocr_revision = input_root.parent.name
            unresolved.update(review_ids(input_root.parent / book_id / "review.json"))
        if not paragraph_path.is_file():
            raise FileNotFoundError(f"Missing reviewed paragraphs for {book_id}: {paragraph_path}")
        ranges = by_book[book_id]
        selected_label_counts: Counter[str] = Counter()
        role_ranges = book.get("role_ranges", [])
        excluded_path = book_dir / "excluded_fragments.json" if book_id not in server_books else None
        if excluded_path is not None and excluded_path.is_file():
            for source_id in json.loads(excluded_path.read_text(encoding="utf-8")):
                match = ID_PAGE_RE.match(str(source_id))
                audit_rows.append({
                    "book_id": book_id,
                    "source_fragment_ids": [str(source_id)],
                    "pages": [int(match.group(1))] if match else [],
                    "decision": "excluded_upstream",
                    "reason": "confirmed_duplicate_page_removed_by_clean_export",
                })
                counts["confirmed_duplicate_page_removed_by_clean_export"] += 1
        paragraphs = json.loads(paragraph_path.read_text(encoding="utf-8"))
        raw_text_hashes = {
            str(paragraph.get("id")): hashlib.sha256(str(paragraph.get("text") or "").strip().encode("utf-8")).hexdigest()
            for paragraph in paragraphs
        }
        cleaned_fragments: dict[str, dict[str, Any]] = {}
        fragment_root = (
            server_cleaned_fragments_root if book_id in server_books and server_cleaned_fragments_root is not None
            else cleaned_fragments_root
        )
        if fragment_root is not None:
            if book_id in server_books and server_cleaning_path is None:
                raise ValueError(f"Selected server book {book_id} requires --server-cleaning-audit")
            cleaned_fragments = load_cleaned_fragment_map(fragment_root / f"{book_id}.jsonl")
            line_scores = [
                float(line["score"])
                for fragment in cleaned_fragments.values()
                for line in fragment.get("lines", [])
                if isinstance(line.get("score"), (int, float))
            ]
            weak_line_count = sum(score < 0.75 for score in line_scores)
            quality_rows.append({
                "book_id": book_id,
                "page_count": int(book.get("page_count", 0)),
                "ocr_revision": ocr_revision,
                "retained_fragment_count": len(cleaned_fragments),
                "scored_line_count": len(line_scores),
                "mean_line_confidence": sum(line_scores) / len(line_scores) if line_scores else None,
                "weak_line_threshold": 0.75,
                "weak_line_count": weak_line_count,
                "weak_line_fraction": weak_line_count / len(line_scores) if line_scores else None,
                "interpretation": "OCR confidence is a model-specific triage signal, not an estimate of character accuracy.",
            })
            # The cleaning inventory includes layout furniture and duplicate
            # rows that were never paragraph sources. Only assembled source IDs
            # are eligible for strict-output reassembly.
            source_ids = {
                str(source.get("id"))
                for paragraph in paragraphs
                for source in paragraph.get("sources", [])
                if source.get("id")
            }
            removed_for_book = {
                str(fragment_id)
                for row in cleaning_audit
                if str(row.get("book_id", row.get("book", ""))) == book_id
                for fragment_id in row.get("removed_source_ids", [])
            }
            upstream_excluded_ids = (
                set(json.loads(excluded_path.read_text(encoding="utf-8")))
                if excluded_path is not None and excluded_path.is_file()
                else set()
            )
            missing_cleaned = source_ids - cleaned_fragments.keys()
            if not missing_cleaned.issubset(removed_for_book):
                raise ValueError(
                    f"Cleaned-fragment source coverage has unexplained missing IDs for {book_id}: "
                    f"{sorted(missing_cleaned - removed_for_book)[:8]}"
                )
            unassembled = set(cleaned_fragments) - source_ids
            unexplained_unassembled = unassembled - upstream_excluded_ids
            if unexplained_unassembled:
                raise ValueError(
                    f"Cleaned retained fragments are absent from saved paragraph chains for {book_id}: "
                    f"{sorted(unexplained_unassembled)[:8]}"
                )
            source_accounting.append({
                "book_id": book_id,
                "cleaned_fragment_count": len(cleaned_fragments),
                "reviewed_paragraph_source_count": len(source_ids),
                "removed_paragraph_source_count": len(missing_cleaned),
                "removed_ids_audited": len(removed_for_book),
                "cleaned_fragments_not_in_assembled_paragraphs": len(unassembled),
                "not_assembled_but_excluded_upstream_count": len(unassembled.intersection(upstream_excluded_ids)),
                "not_assembled_sample_ids": sorted(unassembled)[:20],
                "not_assembled_reason": "retained layout fragments absent from the saved paragraph source chains are not silently added to training rows",
            })
            paragraph_fragments = {key: value for key, value in cleaned_fragments.items() if key in source_ids}
            paragraphs = reassemble_cleaned_paragraphs(paragraphs, paragraph_fragments)
        for paragraph_position, paragraph in enumerate(paragraphs):
            ids = fragment_ids(paragraph)
            pages = source_pages(paragraph)
            text = str(paragraph.get("text") or "").strip()
            body_range, reason, matched_range = selection_reason(
                paragraph, ranges, role_ranges, unresolved, strict_exclusions
            )
            if body_range is None:
                counts[reason] += 1
                audit_rows.append({
                    "book_id": book_id,
                    "paragraph_id": paragraph.get("id"),
                    "source_fragment_ids": ids,
                    "pages": pages,
                    "ocr_label": paragraph.get("label"),
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text_length": len(text),
                    "decision": "excluded",
                    "reason": reason,
                    "reviewed_range": matched_range,
                })
                continue
            source_id = f"{book_id}:{stable_id(book_id, *ids)}"
            duplicate_sources = seen_source_fragments.intersection(ids)
            if duplicate_sources:
                raise ValueError(f"Source fragments selected more than once: {sorted(duplicate_sources)}")
            seen_source_fragments.update(ids)
            if source_id in seen_source_blocks:
                raise ValueError(f"Duplicate source_block_id after assembly: {source_id}")
            seen_source_blocks.add(source_id)
            strict_rows.append(strict_row(
                paragraph, body_range, book, source_id, pages,
                paragraph_position, ranges_path.name,
            ))
            selected_label_counts[str(paragraph.get("label") or "unknown")] += 1
            selected_books.add(book_id)
            paragraph_revision = paragraph_ocr_revision(paragraph, book_id, server_pages_by_book)
            lineage_rows.append({
                "source_block_id": source_id,
                "book_id": book_id,
                "paragraph_id": paragraph.get("id"),
                "paragraph_position": paragraph_position,
                "ocr_label": paragraph.get("label"),
                "source_fragment_ids": ids,
                "pages": pages,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "pre_cleanup_paragraph_text_sha256": raw_text_hashes.get(
                    str(paragraph.get("id")), hashlib.sha256(text.encode("utf-8")).hexdigest()
                ),
                "cleanup_audit_ids": sorted({
                    audit_id
                    for fragment_id in ids
                    for audit_id in audit_ids_by_fragment.get(fragment_id, [])
                }),
                "source_fragment_text_sha256": {
                    fragment_id: hashlib.sha256(
                        str(cleaned_fragments[fragment_id].get("text") or "").encode("utf-8")
                    ).hexdigest()
                    for fragment_id in ids
                    if fragment_id in cleaned_fragments
                },
                "verified_transcriptions": [
                    verified_transcriptions[source_id]
                    for source_id in ids
                    if source_id in verified_transcriptions
                ],
                "work_id": strict_rows[-1]["work_id"],
                "range_start_page": body_range["start_page"],
                "range_end_page": body_range["end_page"],
                "boundary_status": "reviewed_or_same_page_layout_block",
                "ocr_revision": paragraph_revision,
                "source_page_revisions": {
                    str(page): (
                        "pdf_ocr-server-v1"
                        if page in server_pages_by_book.get(book_id, set())
                        else "pdf_ocr-reviewed-v1"
                    )
                    for page in pages
                },
            })
            counts["selected"] += 1
        for accounting in source_accounting:
            if accounting["book_id"] == book_id:
                accounting["selected_layout_label_counts"] = dict(selected_label_counts)

    if cleaning_path is not None:
        for row, audit_id in zip(cleaning_audit, cleaning_audit_ids, strict=True):
            affected = [*row.get("removed_source_ids", []), *row.get("modified_source_ids", [])]
            affected.extend(row.get("derived_source_ids", []))
            decision = (
                "manual_folio_not_applicable"
                if row.get("status") == "not_applicable"
                else "excluded_during_scan_cleanup"
                if row.get("removed_source_ids")
                else "modified_during_scan_cleanup"
                if row.get("modified_source_ids")
                else "scan_cleanup_audit"
            )
            audit_rows.append({
                "audit_id": audit_id,
                "book_id": row.get("book_id", row.get("book")),
                "source_fragment_ids": affected or row.get("source_ids", []),
                "pages": row.get("pages", []),
                "decision": decision,
                "reason": row.get("reason", row.get("rule", "scan_cleanup")),
                "cleaning_evidence": row.get("evidence"),
            })
            counts[audit_rows[-1]["decision"]] += 1

    expected_books = book_ids or set(books)
    missing_books = expected_books - selected_books
    if missing_books:
        raise ValueError(f"Books with no eligible strict rows: {sorted(missing_books)}")

    if not strict_rows:
        raise ValueError("No eligible strict rows were selected")
    table = pa.Table.from_pylist(strict_rows, schema=SCANNED_SCHEMA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_parquet = output_path.with_name(output_path.name + ".tmp")
    pq.write_table(table, temporary_parquet)
    os.replace(temporary_parquet, output_path)
    write_jsonl(audit_path, audit_rows)
    write_jsonl(lineage_path, lineage_rows)
    source_accounting_path = output_path.with_name(output_path.stem + "_source_accounting.json")
    write_json(source_accounting_path, source_accounting)
    selected_by_book: dict[str, list[dict[str, Any]]] = {}
    for row in strict_rows:
        book_id = row["source_href"].split(":", 2)[1]
        selected_by_book.setdefault(book_id, []).append(row)
    for quality in quality_rows:
        selected = selected_by_book.get(quality["book_id"], [])
        authors = sorted({str(row["author"]) for row in selected if row.get("author")})
        works = sorted({str(row["work"]) for row in selected if row.get("work")})
        work_ids = sorted({str(row["work_id"]) for row in selected if row.get("work_id")})
        languages = sorted({str(row["lang"]) for row in selected if row.get("lang")})
        quality.update({
            "selected_rows": len(selected),
            "authors": authors,
            "unique_author_count": len(authors),
            "work_titles": works,
            "unique_work_id_count": len(work_ids),
            "languages": languages,
            "strict_row_layout_label_counts": dict(
                Counter(str(row.get("ocr_label") or "unknown") for row in lineage_rows if row["book_id"] == quality["book_id"])
            ),
        })
    quality_path = output_path.with_name(output_path.stem + "_quality.json")
    write_json(quality_path, quality_rows)
    reread = pq.read_table(output_path)
    if reread.schema != SCANNED_SCHEMA:
        raise AssertionError(f"Export schema drifted:\n{reread.schema}")
    if reread.num_rows != len(strict_rows):
        raise AssertionError("Parquet reread row count mismatch")
    ids = reread["source_block_id"].to_pylist()
    work_ids = reread["work_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate source_block_id in strict export")
    if not set(ids).issubset({row["source_block_id"] for row in lineage_rows}):
        raise AssertionError("Lineage sidecar does not cover every strict source_block_id")
    if any(not value for value in work_ids):
        raise AssertionError("Empty work_id in strict export")
    excluded_source_ids = {
        str(source_id)
        for row in audit_rows
        if row.get("decision") in {"excluded", "excluded_upstream", "excluded_during_scan_cleanup"}
        for source_id in row.get("source_fragment_ids", [])
    }
    modified_cleanup_ids = {
        str(source_id)
        for row in cleaning_audit
        for source_id in row.get("modified_source_ids", [])
    }
    return {
        "rows": reread.num_rows,
        "books": len({row["source_file"] for row in strict_rows}),
        "unique_work_ids": len(set(work_ids)),
        "unique_source_block_ids": len(set(ids)),
        "selected_book_ids": sorted(selected_books),
        "audit_records": len(audit_rows),
        "excluded_source_fragments": len(excluded_source_ids),
        "modified_cleanup_source_fragments": len(modified_cleanup_ids),
        "exclusion_reasons": dict(counts),
        "schema": str(reread.schema),
        "output": str(output_path),
        "audit": str(audit_path),
        "lineage": str(lineage_path),
        "source_accounting": str(source_accounting_path),
        "quality": str(quality_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--body-ranges", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bookraw/books_scanned.parquet")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/bookraw/books_scanned_exclusions.jsonl")
    parser.add_argument("--lineage", type=Path, default=ROOT / "reports/bookraw/books_scanned_lineage.jsonl")
    parser.add_argument("--boundary-reidentification", type=Path)
    parser.add_argument("--strict-exclusion-manifest", type=Path)
    parser.add_argument("--cleaning-audit", type=Path)
    parser.add_argument("--server-root", type=Path)
    parser.add_argument("--server-selection", type=Path)
    parser.add_argument("--cleaned-fragments-root", type=Path)
    parser.add_argument("--server-cleaned-fragments-root", type=Path)
    parser.add_argument("--server-cleaning-audit", type=Path)
    parser.add_argument("--verified-transcriptions", type=Path)
    parser.add_argument("--book-id", action="append", dest="book_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = export(
        args.input_root,
        args.body_ranges,
        args.output,
        args.audit,
        args.lineage,
        args.boundary_reidentification,
        args.cleaning_audit,
        args.strict_exclusion_manifest,
        args.server_root,
        args.server_selection,
        args.cleaned_fragments_root,
        args.server_cleaned_fragments_root,
        args.server_cleaning_audit,
        args.verified_transcriptions,
        set(args.book_ids) if args.book_ids else None,
        args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
