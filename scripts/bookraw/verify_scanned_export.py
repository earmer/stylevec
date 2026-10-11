#!/usr/bin/env python3
"""Verify saved scanned strict export and provenance sidecars without rewriting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from export_scanned import SCANNED_SCHEMA, join_fragment_texts, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
# Tables and sidecars live on the data volume, not under reports/bookraw/; see
# reports/bookraw/README.md. Override with STYLEVEC_DATA for another checkout.
DATA = Path(os.environ.get("STYLEVEC_DATA", "/Volumes/EMC/stylevec_data"))
DEFAULT_REPORTS = DATA / "manifests/bookraw"
DEFAULT_TABLES = DATA / "processed/bookraw"
DEFAULT_INPUT_ROOT = DATA / "processed/pdf_ocr-reviewed-v1/clean_export"
DEFAULT_SERVER_ROOT = DATA / "processed/pdf_ocr-server-v1"
DEFAULT_REVIEWED_VERIFICATION = DATA / "processed/pdf_ocr-reviewed-v1/verification.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_raw_integrity(raw_root: Path, expected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in expected:
        book_id = str(item["book"])
        page_dir = raw_root / book_id / "pages"
        paths = sorted(page_dir.glob("*.json"))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        observed = {"book": book_id, "pages": len(paths), "sha256": digest.hexdigest()}
        if observed != {"book": book_id, "pages": int(item["pages"]), "sha256": item["sha256"]}:
            raise ValueError(f"Raw OCR integrity mismatch for {book_id}: {observed}")
        result.append(observed)
    return result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fragment_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        source_id = str(item.get("id", ""))
        if not source_id or source_id in result:
            raise ValueError(f"Missing/duplicate source fragment ID in {path}: {source_id}")
        result[source_id] = item
    return result


def verify(args: argparse.Namespace) -> dict[str, Any]:
    required_paths = [
        args.parquet, args.reference, args.body_ranges, args.lineage, args.exclusions,
        args.source_accounting, args.quality, args.reviewed_verification,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    ref_stat = args.reference.stat()
    ref_hash = sha256_file(args.reference)
    reference = pq.read_table(args.reference)
    table = pq.read_table(args.parquet)
    if table.schema.remove_metadata() != reference.schema.remove_metadata():
        raise ValueError("Saved Parquet schema differs from strict reference schema")
    if table.schema.remove_metadata() != SCANNED_SCHEMA.remove_metadata():
        raise ValueError("Saved Parquet schema differs from the exporter’s declared 27-field schema")
    rows = table.to_pylist()
    lineages = read_jsonl(args.lineage)
    exclusions = read_jsonl(args.exclusions)
    ranges_rows = read_jsonl(args.body_ranges)
    strict_exclusion_manifest = read_jsonl(args.strict_exclusion_manifest) if args.strict_exclusion_manifest else []
    source_accounting = read_json(args.source_accounting)
    quality_rows = read_json(args.quality)
    reviewed_verification = read_json(args.reviewed_verification)
    server_verification = read_json(args.server_verification) if args.server_verification else None
    server_book_ids = set()
    server_pages_by_book: dict[str, set[int]] = {}
    selected_server_page_count = 0
    if server_verification is not None:
        server_book_ids = {
            str(row["book"])
            for row in [*server_verification.get("full_books", []), *server_verification.get("partial_books", [])]
        }
        selection = read_json(args.server_selection)
        full_ids = {str(row["book"]) for row in selection.get("full_books", [])}
        for row in selection.get("full_books", []):
            server_pages_by_book[str(row["book"])] = set(range(1, int(row["pages"]) + 1))
        for row in selection.get("selected_pages", []):
            if str(row["book"]) not in full_ids:
                server_pages_by_book.setdefault(str(row["book"]), set()).add(int(row["page"]))
        if set(server_pages_by_book) != server_book_ids:
            raise ValueError("Server selection and assembled server book coverage differ")
        selected_server_page_count = sum(len(pages) for pages in server_pages_by_book.values())
        partial_selected_page_count = sum(
            len(row.get("server_reprocessed_pages", []))
            for row in server_verification.get("partial_books", [])
        )
        if partial_selected_page_count != int(server_verification.get("selected_page_count", partial_selected_page_count)):
            raise ValueError("Partial server selection page count differs from completed verification")
        manifest_partial_page_count = sum(1 for row in selection.get("selected_pages", []) if str(row["book"]) not in full_ids)
        if manifest_partial_page_count != partial_selected_page_count:
            raise ValueError("Partial selection manifest differs from completed verification")
    if reviewed_verification.get("raw_ocr_unchanged") is not True:
        raise ValueError("Reviewed-v1 verification does not certify unchanged raw OCR")

    range_by_book = {str(item["book_id"]): item for item in ranges_rows}
    expected_books = set(range_by_book)
    selected_books = {str(row["source_href"]).split(":", 2)[1] for row in rows}
    if selected_books != expected_books:
        raise ValueError(
            f"Parquet book coverage differs from body-range manifest; missing={sorted(expected_books-selected_books)}, extra={sorted(selected_books-expected_books)}"
        )
    if len(lineages) != len(rows):
        raise ValueError(f"Lineage row count {len(lineages)} != Parquet row count {len(rows)}")
    lineage_by_id = {str(item["source_block_id"]): item for item in lineages}
    block_ids = [str(row["source_block_id"]) for row in rows]
    if len(block_ids) != len(set(block_ids)) or set(block_ids) != set(lineage_by_id):
        raise ValueError("Parquet and lineage source_block_id sets are not a unique one-to-one match")

    excluded_ids = {
        str(source_id)
        for item in exclusions
        if item.get("decision") in {"excluded", "excluded_upstream", "excluded_during_scan_cleanup"}
        for source_id in item.get("source_fragment_ids", [])
    }
    manual_exclusion_ids: set[str] = set()
    for item in strict_exclusion_manifest:
        if (
            item.get("decision") != "exclude_from_strict"
            or item.get("reviewer_model") != "gpt-6-luna"
            or item.get("scope") != "strict_export_only"
            or item.get("body_range_review_complete") is not True
            or item.get("all_clean_source_action") != "retain_original_fragments"
        ):
            raise ValueError(f"Strict role exclusion is incomplete or has unsafe scope: {item}")
        for evidence in item.get("inspected_images", []):
            image_path = Path(str(evidence.get("path", "")))
            if not image_path.is_file() or sha256_file(image_path) != evidence.get("sha256"):
                raise ValueError(f"Strict-exclusion source image is missing or changed: {image_path}")
        for fragment_id in item.get("source_fragment_ids", []):
            fragment_id = str(fragment_id)
            if fragment_id in manual_exclusion_ids:
                raise ValueError(f"Duplicate strict-only manual exclusion ID: {fragment_id}")
            manual_exclusion_ids.add(fragment_id)
    audit_excluded_ids = {
        str(fragment_id)
        for item in exclusions
        if item.get("decision") == "excluded"
        and str(item.get("reason", "")).startswith("manual_source_image_role_exclusion:")
        for fragment_id in item.get("source_fragment_ids", [])
    }
    if args.strict_exclusion_manifest is not None and manual_exclusion_ids != audit_excluded_ids:
        raise ValueError(
            f"Strict-only image-reviewed exclusions differ from saved exclusion audit: "
            f"manifest={len(manual_exclusion_ids)} audit={len(audit_excluded_ids)}"
        )
    used_fragments: set[str] = set()
    authors: set[str] = set()
    works: set[str] = set()
    rows_by_book: Counter[str] = Counter()
    row_hashes = {}
    paragraph_maps: dict[str, dict[str, dict[str, Any]]] = {}
    fragment_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        source_id = str(row["source_block_id"])
        lineage = lineage_by_id[source_id]
        book_id = str(row["source_href"]).split(":", 2)[1]
        pages = [int(value) for value in lineage.get("pages", [])]
        if not pages or int(row["source_spine_index"]) != pages[0]:
            raise ValueError(f"Invalid PDF page provenance for {source_id}")
        if str(row["source_href"]) != f"pdf-page:{book_id}:{pages[0]}":
            raise ValueError(f"source_href does not match lineage page for {source_id}")
        source_pdf = Path(str(row["source_file"]))
        if not source_pdf.is_file():
            raise FileNotFoundError(f"Source PDF missing for {source_id}: {source_pdf}")
        text = str(row["text"] or "")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if len(text) < 30 or lineage.get("text_sha256") != text_hash:
            raise ValueError(f"Text length/hash mismatch for {source_id}")
        if row["lang"] != "zh" or row["text_role"] != "primary_text":
            raise ValueError(f"Unexpected language or role for {source_id}")
        if row["role_confidence"] != "high" or row["boundary_confidence"] != "high":
            raise ValueError(f"Non-high strict confidence in {source_id}")
        if row["role_source"] != "gpt_6_luna_reviewed_pdf_page_range":
            raise ValueError(f"Unexpected reviewer provenance for {source_id}")
        body_ranges = range_by_book[book_id].get("body_ranges", [])
        eligible_ranges = [
            item for item in body_ranges
            if item.get("role") == "primary_text"
            and item.get("role_confidence") == "high"
            and item.get("boundary_confidence") == "high"
            and item.get("reviewer_model") == "gpt-6-luna"
            and item.get("strict_export_eligible") is not False
            and not item.get("strict_exclusion_reason")
            and all(int(item["start_page"]) <= page <= int(item["end_page"]) for page in pages)
        ]
        if len(eligible_ranges) != 1:
            raise ValueError(f"No unique high-confidence reviewed primary range for {source_id}")
        if row["author"] != eligible_ranges[0]["author"] or row["work"] != eligible_ranges[0]["work_title"]:
            raise ValueError(f"Author/work differs from reviewed range for {source_id}")
        if lineage.get("work_id") != row["work_id"]:
            raise ValueError(f"work_id differs from lineage for {source_id}")
        expected_page_revisions = {
            str(page): (
                "pdf_ocr-server-v1"
                if page in server_pages_by_book.get(book_id, set())
                else "pdf_ocr-reviewed-v1"
            )
            for page in pages
        }
        if server_verification is not None and lineage.get("source_page_revisions") != expected_page_revisions:
            raise ValueError(f"Per-page OCR revision lineage mismatch for {source_id}")
        expected_revision = (
            "pdf_ocr-reviewed-v1"
            if not set(pages).intersection(server_pages_by_book.get(book_id, set()))
            else "pdf_ocr-server-v1"
            if set(pages).issubset(server_pages_by_book.get(book_id, set()))
            else "pdf_ocr-reviewed-v1+pdf_ocr-server-v1"
        )
        if server_verification is not None and lineage.get("ocr_revision") != expected_revision:
            raise ValueError(f"Paragraph OCR revision does not match contributing pages for {source_id}")
        fragments = [str(item) for item in lineage.get("source_fragment_ids", [])]
        if not fragments or len(fragments) != len(set(fragments)):
            raise ValueError(f"Missing or duplicate source fragments for {source_id}")
        if used_fragments.intersection(fragments):
            raise ValueError(f"Source fragment selected in multiple rows: {sorted(used_fragments.intersection(fragments))[:4]}")
        used_fragments.update(fragments)
        fragment_hashes = lineage.get("source_fragment_text_sha256", {})
        if not set(fragments).issubset(fragment_hashes):
            raise ValueError(f"Missing source-fragment text hashes for {source_id}")
        fragment_root = args.server_cleaned_fragments_root if book_id in server_book_ids else args.cleaned_fragments_root
        if fragment_root is not None:
            if book_id not in fragment_maps:
                fragment_maps[book_id] = load_fragment_map(fragment_root / f"{book_id}.jsonl")
            fragment_map = fragment_maps[book_id]
            source_texts = []
            for fragment_id in fragments:
                fragment = fragment_map.get(fragment_id)
                if fragment is None:
                    raise ValueError(f"Lineage source fragment is missing from cleaned source map: {fragment_id}")
                actual_fragment_hash = hashlib.sha256(str(fragment.get("text") or "").encode("utf-8")).hexdigest()
                if fragment_hashes[fragment_id] != actual_fragment_hash:
                    raise ValueError(f"Source fragment text hash mismatch: {fragment_id}")
                source_texts.append(str(fragment.get("text") or ""))
            rebuilt_hash = hashlib.sha256(join_fragment_texts(source_texts).encode("utf-8")).hexdigest()
            if rebuilt_hash != text_hash:
                raise ValueError(f"Cleaned source fragments do not rebuild strict row text: {source_id}")
        if set(fragments).intersection(excluded_ids):
            raise ValueError(f"Selected fragment is also recorded excluded: {source_id}")
        source_root = args.server_root if book_id in server_book_ids else args.input_root
        paragraph_path = source_root / book_id / "paragraphs.json"
        if not paragraph_path.is_file():
            raise FileNotFoundError(f"Lineage source paragraphs missing: {paragraph_path}")
        if book_id not in paragraph_maps:
            paragraph_rows = json.loads(paragraph_path.read_text(encoding="utf-8"))
            paragraph_maps[book_id] = {str(item.get("id")): item for item in paragraph_rows}
        position = int(lineage["paragraph_position"])
        if int(row["source_dom_position"]) != position or position < 0:
            raise ValueError(f"source_dom_position mismatch for {source_id}")
        source_paragraph = paragraph_maps[book_id].get(str(lineage.get("paragraph_id")))
        if source_paragraph is None:
            raise ValueError(f"Source paragraph ID is missing for {source_id}")
        source_ids = {str(item.get("id")) for item in source_paragraph.get("sources", []) if item.get("id")}
        if not set(fragments).issubset(source_ids):
            raise ValueError(f"Lineage source fragments are absent from paragraph chain for {source_id}")
        row_hashes[source_id] = text_hash
        rows_by_book[book_id] += 1
        authors.add(str(row["author"]))
        works.add(str(row["work_id"]))

    if manual_exclusion_ids.intersection(used_fragments):
        raise ValueError("A manually excluded source fragment appears in a strict training row")

    expected_raw = reviewed_verification.get("raw_ocr_integrity", [])
    raw_integrity = current_raw_integrity(args.raw_root, expected_raw)
    if server_verification is not None:
        if server_verification.get("complete") is not True:
            raise ValueError("Server-v1 verification is incomplete")
        if server_verification.get("selection_manifest") and server_verification.get("original_raw_changed") is True:
            raise ValueError("Server-v1 verification reports modified original raw OCR")
        if len(server_book_ids) != 5 or selected_server_page_count != 659:
            raise ValueError(
                f"Expected five upgraded books and 659 server pages; found {len(server_book_ids)} books/{selected_server_page_count} pages"
            )
        if server_book_ids - expected_books:
            raise ValueError(f"Server-upgraded books are absent from export manifest: {sorted(server_book_ids - expected_books)}")
    if args.verified_transcriptions is not None:
        transcription_rows = read_jsonl(args.verified_transcriptions)
        transcription_by_id = {str(row["derived_fragment_id"]): row for row in transcription_rows}
        seen_transcriptions: Counter[str] = Counter()
        for derived_id, transcription in transcription_by_id.items():
            if transcription.get("kind") != "source_image_verified_transcription" or transcription.get("reviewer_model") != "gpt-6-luna" or transcription.get("confidence") != "high":
                raise ValueError(f"Invalid verified-transcription provenance: {derived_id}")
            for path_key, hash_key in (("source_pdf", "source_pdf_sha256"), ("source_page_image", "source_page_image_sha256"), ("source_crop_image", "source_crop_image_sha256")):
                evidence_path = Path(str(transcription.get(path_key, "")))
                if not evidence_path.is_file() or sha256_file(evidence_path) != transcription.get(hash_key):
                    raise ValueError(f"Verified-transcription evidence missing or changed: {evidence_path}")
            raw_page = args.input_root.parent / str(transcription["book_id"]) / "pages" / f"{int(transcription['page']):05d}.json"
            if not raw_page.is_file():
                raise FileNotFoundError(f"Original reviewed page for transcription is missing: {raw_page}")
            page_data = read_json(raw_page)
            source_matches = [item for item in page_data.get("blocks", []) if str(item.get("id")) == str(transcription.get("source_fragment_id"))]
            if len(source_matches) != 1 or str(source_matches[0].get("text")) != str(transcription.get("source_text")):
                raise ValueError(f"Verified-transcription before-text/source ID mismatch: {derived_id}")
        for item in lineages:
            for transcription in item.get("verified_transcriptions", []):
                derived_id = str(transcription.get("derived_fragment_id", ""))
                if derived_id not in transcription_by_id or transcription_by_id[derived_id] != transcription:
                    raise ValueError(f"Lineage transcription differs from source-image audit: {derived_id}")
                seen_transcriptions[derived_id] += 1
        if set(seen_transcriptions) != set(transcription_by_id) or any(count != 1 for count in seen_transcriptions.values()):
            raise ValueError("Verified transcription IDs do not appear exactly once in strict lineage")

    accounting_by_book = {str(row["book_id"]): row for row in source_accounting}
    if set(accounting_by_book) != expected_books:
        raise ValueError("Source-accounting book coverage differs from body-range manifest")
    for book_id, accounting in accounting_by_book.items():
        source_root = args.server_root if book_id in server_book_ids else args.input_root
        paragraph_path = source_root / book_id / "paragraphs.json"
        if not paragraph_path.is_file():
            raise FileNotFoundError(paragraph_path)
        if book_id not in paragraph_maps:
            paragraph_rows = json.loads(paragraph_path.read_text(encoding="utf-8"))
            paragraph_maps[book_id] = {str(item.get("id")): item for item in paragraph_rows}
        all_source_ids = {
            str(source.get("id"))
            for paragraph in paragraph_maps[book_id].values()
            for source in paragraph.get("sources", [])
            if source.get("id")
        }
        if len(all_source_ids) != int(accounting["reviewed_paragraph_source_count"]):
            raise ValueError(f"Paragraph source count differs from source accounting for {book_id}")
        fragment_root = args.server_cleaned_fragments_root if book_id in server_book_ids else args.cleaned_fragments_root
        if fragment_root is not None:
            if book_id not in fragment_maps:
                fragment_maps[book_id] = load_fragment_map(fragment_root / f"{book_id}.jsonl")
            all_clean_ids = set(fragment_maps[book_id])
            if len(all_clean_ids) != int(accounting["cleaned_fragment_count"]):
                raise ValueError(f"Cleaned source fragment count differs from source accounting for {book_id}")
            missing_ids = all_source_ids - all_clean_ids
            extra_ids = all_clean_ids - all_source_ids
            cleanup_removed = {
                str(source_id)
                for item in exclusions
                if str(item.get("book_id")) == book_id and item.get("decision") == "excluded_during_scan_cleanup"
                for source_id in item.get("source_fragment_ids", [])
            }
            if missing_ids != cleanup_removed.intersection(all_source_ids):
                raise ValueError(
                    f"Paragraph sources absent from cleaned map do not match cleanup exclusions for {book_id}: "
                    f"missing={len(missing_ids)} audited_paragraph_ids={len(cleanup_removed.intersection(all_source_ids))}"
                )
            excluded_file = args.input_root / book_id / "excluded_fragments.json"
            upstream_ids = set(json.loads(excluded_file.read_text(encoding="utf-8"))) if excluded_file.is_file() and book_id not in server_book_ids else set()
            if extra_ids != extra_ids.intersection(upstream_ids):
                raise ValueError(f"Cleaned fragments absent from source paragraph chains are unexplained for {book_id}")
            if len(missing_ids) != int(accounting.get("removed_paragraph_source_count", 0)):
                raise ValueError(f"Missing cleaned-source count differs from accounting for {book_id}")
            audited_cleanup_ids = {
                str(source_id)
                for item in exclusions
                if str(item.get("book_id")) == book_id
                and item.get("decision") == "excluded_during_scan_cleanup"
                for source_id in item.get("source_fragment_ids", [])
            }
            if len(audited_cleanup_ids) != int(accounting.get("removed_ids_audited", 0)):
                raise ValueError(f"Cleanup audit source-ID count differs from accounting for {book_id}")
            modified_cleanup_ids = {
                str(source_id)
                for item in exclusions
                if str(item.get("book_id")) == book_id and item.get("decision") == "modified_during_scan_cleanup"
                for source_id in item.get("source_fragment_ids", [])
            }
            if not modified_cleanup_ids.issubset(all_clean_ids):
                raise ValueError(f"Modified cleanup fragments are missing from the retained source map for {book_id}")
        if int(accounting["reviewed_paragraph_source_count"]) < int(accounting["cleaned_fragment_count"]):
            allowed = int(accounting.get("cleaned_fragments_not_in_assembled_paragraphs", 0))
            if int(accounting["cleaned_fragment_count"]) - int(accounting["reviewed_paragraph_source_count"]) > allowed:
                raise ValueError(f"Source accounting has unexplained extra fragments for {book_id}")
        missing = int(accounting.get("removed_paragraph_source_count", 0))
        audited = int(accounting.get("removed_ids_audited", 0))
        if missing > audited:
            raise ValueError(f"Source accounting has unaudited missing paragraph IDs for {book_id}")
    if {str(row["book_id"]) for row in quality_rows} != expected_books:
        raise ValueError("OCR quality sidecar book coverage differs from body-range manifest")
    if any("not an estimate of character accuracy" not in str(row.get("interpretation", "")) for row in quality_rows):
        raise ValueError("OCR quality sidecar omits confidence interpretation")

    reference_identity = {
        "path": str(args.reference.resolve()),
        "sha256": ref_hash,
        "size_bytes": ref_stat.st_size,
        "mtime_ns": ref_stat.st_mtime_ns,
        "schema": str(reference.schema.remove_metadata()),
    }
    result = {
        "verified": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "parquet": str(args.parquet.resolve()),
        "rows": len(rows),
        "book_count": len(rows_by_book),
        "book_rows": dict(sorted(rows_by_book.items())),
        "unique_author_count": len(authors),
        "unique_work_id_count": len(works),
        "unique_source_block_id_count": len(block_ids),
        "unique_source_fragment_id_count": len(used_fragments),
        "lineage_rows": len(lineages),
        "exclusion_audit_records": len(exclusions),
        "reference": reference_identity,
        "raw_ocr_integrity_verified_books": len(raw_integrity),
        "raw_ocr_integrity": raw_integrity,
        "checks": [
            "exact 27-field Arrow schema matches the saved strict reference",
            "all manifest books and source PDFs are represented",
            "unique strict block IDs and unique source fragments",
            "row text hashes, source paragraph IDs/positions, ranges, and source fragments match lineage",
            "strict role/language/minimum length/confidence invariants",
            "source-accounting and OCR quality sidecars cover every book",
            "raw OCR file hashes match the reviewed baseline",
        ],
    }
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to replace verification report: {args.output}; pass --force")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_TABLES / "books_scanned.parquet")
    parser.add_argument("--reference", type=Path, default=DEFAULT_TABLES / "books_work_primary_strict.parquet")
    parser.add_argument("--body-ranges", type=Path, default=DEFAULT_REPORTS / "scanned_body_ranges.jsonl")
    parser.add_argument("--lineage", type=Path, default=DEFAULT_REPORTS / "books_scanned_lineage.jsonl")
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_REPORTS / "books_scanned_exclusions.jsonl")
    parser.add_argument("--strict-exclusion-manifest", type=Path)
    parser.add_argument("--source-accounting", type=Path, default=DEFAULT_REPORTS / "books_scanned_source_accounting.json")
    parser.add_argument("--quality", type=Path, default=DEFAULT_REPORTS / "books_scanned_quality.json")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--server-root", type=Path, default=DEFAULT_SERVER_ROOT)
    parser.add_argument("--cleaned-fragments-root", type=Path)
    parser.add_argument("--server-cleaned-fragments-root", type=Path)
    parser.add_argument("--server-verification", type=Path)
    parser.add_argument("--server-selection", type=Path, default=DEFAULT_SERVER_ROOT / "selection.json")
    parser.add_argument("--verified-transcriptions", type=Path)
    parser.add_argument("--reviewed-verification", type=Path, default=DEFAULT_REVIEWED_VERIFICATION)
    parser.add_argument("--raw-root", type=Path, default=Path("/Volumes/EMC/stylevec_data/processed/pdf_ocr-v1"))
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORTS / "books_scanned_verification.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
