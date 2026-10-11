#!/usr/bin/env python3
"""Clean full-book server-v1 fragments using reviewed page-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PDF_OCR = ROOT / "stylevec_data/scripts/pdf_ocr"
sys.path.insert(0, str(PDF_OCR))
from scanned_cleanup import clean_scanned_fragments  # noqa: E402
from continuation import content as page_content  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def box_overlap_fraction(left: list[float], right: list[float]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    area = min((left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1]))
    return width * height / area if area > 0 else 0.0


def match_manual_folio(fragment_rows: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    page = int(item["page"])
    target = [float(value) for value in item["line_box"]]
    matches = []
    for fragment in fragment_rows:
        if int(fragment.get("page", 0)) != page:
            continue
        lines = fragment.get("lines") or []
        if len(lines) < 2:
            continue
        line_box = [float(value) for value in lines[-1].get("box", [])]
        if len(line_box) != 4 or box_overlap_fraction(line_box, target) < 0.65:
            continue
        line_text = str(lines[-1].get("text", "")).strip()
        if not re.fullmatch(r"\d{1,4}", line_text):
            continue
        if not str(fragment.get("text", "")).rstrip().endswith(line_text):
            continue
        matches.append(fragment)
    return matches


def clean_server_books(
    server_root: Path,
    selection_path: Path,
    source_clean_root: Path,
    destination_root: Path,
    output_audit: Path,
    not_applicable_path: Path,
    transcription_path: Path | None = None,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    server_verification = json.loads((server_root / "verification.json").read_text(encoding="utf-8"))
    if server_verification.get("complete") is not True:
        raise ValueError("Server revision is not marked complete")
    complete_by_book = {
        str(row["book"]): row
        for row in [*server_verification.get("full_books", []), *server_verification.get("partial_books", [])]
    }
    selected_page_sets: dict[str, set[int]] = {
        str(row["book"]): set(range(1, int(row["pages"]) + 1))
        for row in selection.get("full_books", [])
    }
    for row in selection.get("selected_pages", []):
        selected_page_sets.setdefault(str(row["book"]), set()).add(int(row["page"]))
    missing_selected = set(selected_page_sets) - set(complete_by_book)
    if missing_selected:
        raise ValueError(f"Selected server books are not fully assembled: {sorted(missing_selected)}")
    for book_id, row in complete_by_book.items():
        if book_id not in selected_page_sets:
            continue
        if row.get("page_reference_coverage") is False or row.get("coverage_verified") is False:
            raise ValueError(f"Server page reference coverage failed for {book_id}")
        server_pages = set(int(page) for page in row.get("server_reprocessed_pages", []))
        if server_pages and server_pages != selected_page_sets[book_id]:
            raise ValueError(f"Server page selection differs from completed revision for {book_id}")
    full_books = [
        {"book": book_id, "pages": int(complete_by_book[book_id].get("expected_pages", complete_by_book[book_id].get("pages")))}
        for book_id in selected_page_sets
    ]
    source_verification = json.loads((source_clean_root / "verification.json").read_text(encoding="utf-8"))
    duplicate_pages = source_verification.get("confirmed_duplicates", [])
    cleanup_root = source_clean_root.parent / "scanned_cleanup"
    manual_path = cleanup_root / "manual_page_number_review/confirmed_embedded_folios.json"
    nonbody_path = cleanup_root / "manual_page_number_review/confirmed_nonbody_pages.jsonl"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.is_file() else []
    nonbody = read_jsonl(nonbody_path) if nonbody_path.is_file() else []
    transcriptions = read_jsonl(transcription_path) if transcription_path is not None else []
    transcriptions_by_book: dict[str, list[dict[str, Any]]] = {}
    for item in transcriptions:
        book_id = str(item.get("book_id", ""))
        derived_id = str(item.get("derived_fragment_id", ""))
        page = int(item.get("page", 0))
        if not book_id or not derived_id or page < 1:
            raise ValueError(f"Malformed verified-transcription record: {item}")
        if item.get("kind") != "source_image_verified_transcription" or item.get("reviewer_model") != "gpt-6-luna" or item.get("confidence") != "high":
            raise ValueError(f"Unverified transcription provenance: {derived_id}")
        for path_key, hash_key in (("source_pdf", "source_pdf_sha256"), ("source_page_image", "source_page_image_sha256"), ("source_crop_image", "source_crop_image_sha256")):
            evidence_path = Path(str(item.get(path_key, "")))
            if not evidence_path.is_file() or sha256_file(evidence_path) != item.get(hash_key):
                raise ValueError(f"Verified-transcription evidence is missing or changed: {evidence_path}")
        if not str(item.get("verified_text", "")).strip() or not str(item.get("source_text", "")).strip():
            raise ValueError(f"Verified transcription must preserve before/after text: {derived_id}")
        if not derived_id.startswith(f"{book_id}:{page}:verified_transcription:"):
            raise ValueError(f"Derived transcription ID is outside its isolated namespace: {derived_id}")
        transcriptions_by_book.setdefault(book_id, []).append(item)

    destination_root.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    summaries = []
    for selection_row in full_books:
        book_id = str(selection_row["book"])
        book_root = server_root / book_id
        page_count = int(selection_row["pages"])
        page_dir = book_root / "pages"
        page_paths = [page_dir / f"{page:05d}.json" for page in range(1, page_count + 1)]
        missing = [str(path) for path in page_paths if not path.is_file()]
        if missing:
            raise ValueError(f"Server page set is incomplete for {book_id}: {missing[:4]}")
        fragments: list[dict[str, Any]] = []
        page_sizes: dict[int, tuple[float, float]] = {}
        for page_number, page_path in enumerate(page_paths, 1):
            page = json.loads(page_path.read_text(encoding="utf-8"))
            if page.get("book") != book_id or int(page.get("page", 0)) != page_number:
                raise ValueError(f"Server page identity mismatch: {page_path}")
            page_sizes[page_number] = (float(page["width"]), float(page["height"]))
            fragments.extend({**block, "page": page_number} for block in page_content(page))

        book_transcriptions = transcriptions_by_book.get(book_id, [])
        if any(int(item["page"]) not in selected_page_sets[book_id] for item in book_transcriptions):
            raise ValueError(f"Verified transcription is outside this book's server-selected pages: {book_id}")
        for item in book_transcriptions:
            derived_id = str(item["derived_fragment_id"])
            if any(str(fragment.get("id")) == derived_id for fragment in fragments):
                raise ValueError(f"Verified transcription ID collides with OCR source fragment: {derived_id}")
            mobile_page = source_clean_root.parent / book_id / "pages" / f"{int(item['page']):05d}.json"
            mobile_data = json.loads(mobile_page.read_text(encoding="utf-8"))
            original = [block for block in mobile_data.get("blocks", []) if str(block.get("id")) == str(item["source_fragment_id"])]
            if len(original) != 1 or str(original[0].get("text")) != str(item["source_text"]):
                raise ValueError(f"Verified transcription before-text/source ID mismatch: {derived_id}")
            box = [float(value) for value in item["box"]]
            fragments.append({
                "id": derived_id,
                "page": int(item["page"]),
                "label": "text",
                "text": str(item["verified_text"]),
                "box": box,
                "region_box": [float(value) for value in item.get("region_box", box)],
                "score": 1.0,
                "lines": [{"text": str(item["verified_text"]), "box": box, "score": 1.0}],
                "derived_source": {
                    "kind": item["kind"],
                    "source_fragment_id": item["source_fragment_id"],
                    "source_text": item["source_text"],
                    "reviewer_model": item["reviewer_model"],
                },
            })

        confirmed_folios = []
        for item in manual:
            if not Path(str(item.get("inspected_image", ""))).name.startswith(f"{book_id}_"):
                continue
            matches = match_manual_folio(fragments, item)
            if len(matches) == 1:
                fragment = matches[0]
                server_line_text = str(fragment["lines"][-1].get("text", "")).strip()
                printed_folio = str(item.get("printed_folio", item["line_text"])).strip()
                if not re.fullmatch(r"\d{1,4}", server_line_text):
                    raise ValueError(
                        f"Verified folio location on {book_id} page {item['page']} has nonnumeric OCR text {server_line_text!r}; visual review is needed"
                    )
                applied = {
                    **item,
                    "line_text": server_line_text,
                    "printed_folio": printed_folio,
                    "visual_evidence": (
                        f"The inspected original source image shows printed folio {printed_folio}; "
                        f"the server OCR terminal line reads {server_line_text!r} at the same page and line-box location."
                    ),
                }
                confirmed_folios.append(applied)
            elif not matches:
                not_applicable.append({
                    "book": book_id,
                    "pages": [int(item["page"])],
                    "removed_source_ids": [],
                    "modified_source_ids": [],
                    "status": "not_applicable",
                    "rule": "manual_folio_not_present_in_server_ocr",
                    "reason": "The reviewed page/text/line-box confirmation did not match a server fragment; no old fragment ID was reused.",
                    "evidence": {key: item.get(key) for key in ("line_text", "printed_folio", "line_box", "reason", "visual_evidence", "inspected_image")},
                })
            else:
                raise ValueError(f"Manual folio matched {len(matches)} server fragments for {book_id} PDF page {item['page']}; refusing ambiguous removal")

        duplicates = [row for row in duplicate_pages if row.get("book") == book_id]
        nonbody_pages = [row for row in nonbody if row.get("book") == book_id]
        retained, book_audit = clean_scanned_fragments(
            fragments,
            book_id=book_id,
            page_count=page_count,
            page_sizes=page_sizes,
            confirmed_duplicate_pages=duplicates,
            confirmed_embedded_folios=confirmed_folios,
            confirmed_nonbody_pages=nonbody_pages,
        )
        for row in book_audit:
            row["book_id"] = book_id
        audit_rows.extend(book_audit)
        for item in book_transcriptions:
            paragraphs = json.loads((book_root / "paragraphs.json").read_text(encoding="utf-8"))
            appearances = sum(
                1
                for paragraph in paragraphs
                for source in paragraph.get("sources", [])
                if str(source.get("id")) == str(item["derived_fragment_id"])
            )
            if appearances != 1:
                raise ValueError(
                    f"Assembled paragraph chains must contain each verified transcription exactly once: {item['derived_fragment_id']} ({appearances})"
                )
            audit_rows.append({
                "audit_id": hashlib.sha1(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:20],
                "book_id": book_id,
                "pages": [int(item["page"])],
                "status": "derived_source_transcription",
                "rule": item["kind"],
                "reason": item["evidence"],
                "derived_source_ids": [item["derived_fragment_id"]],
                "original_source_ids": [item["source_fragment_id"]],
                "before_text": item["source_text"],
                "verified_text": item["verified_text"],
                "reviewer_model": item["reviewer_model"],
                "source_pdf_sha256": item["source_pdf_sha256"],
                "source_page_image_sha256": item["source_page_image_sha256"],
                "source_crop_image_sha256": item["source_crop_image_sha256"],
                "source_page_image": item["source_page_image"],
                "source_crop_image": item["source_crop_image"],
            })
        fragment_path = destination_root / f"{book_id}.jsonl"
        with fragment_path.open("w", encoding="utf-8") as stream:
            for row in retained:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries.append({
            "book_id": book_id,
            "pages": page_count,
            "input_fragments": len(fragments),
            "retained_fragments": len(retained),
            "removed_fragments": sum(len(row.get("removed_source_ids", [])) for row in book_audit),
            "modified_fragments": sum(len(row.get("modified_source_ids", [])) for row in book_audit),
            "manual_folios_applied": len(confirmed_folios),
            "verified_transcriptions": len(book_transcriptions),
        })

    destination_root.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    with output_audit.open("w", encoding="utf-8") as stream:
        for row in [*audit_rows, *not_applicable]:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    not_applicable_path.parent.mkdir(parents=True, exist_ok=True)
    not_applicable_path.write_text(json.dumps(not_applicable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    unassembled_overlay_ids = {
        str(item["derived_fragment_id"])
        for items in transcriptions_by_book.values()
        for item in items
        if item["book_id"] in complete_by_book
    }
    report = {
        "books": summaries,
        "not_applicable_manual_folios": len(not_applicable),
        "verified_transcriptions": len(transcriptions),
        "transcription_ids": sorted(unassembled_overlay_ids),
        "fragments": str(destination_root),
        "audit": str(output_audit),
    }
    (destination_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--server-selection", type=Path, required=True)
    parser.add_argument("--source-clean-root", type=Path, default=Path("/Volumes/EMC/stylevec_data/processed/pdf_ocr-reviewed-v1/clean_export"))
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--not-applicable", type=Path, required=True)
    parser.add_argument("--verified-transcriptions", type=Path, default=Path("/Volumes/EMC/stylevec_data/processed/pdf_ocr-server-v1/verified_transcriptions/verified_transcriptions_luna2.jsonl"))
    args = parser.parse_args()
    report = clean_server_books(args.server_root, args.server_selection, args.source_clean_root, args.destination_root, args.audit, args.not_applicable, args.verified_transcriptions)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
