#!/usr/bin/env python3
"""Re-extract EPUB text blocks with conservative role labels and source pointers.

Existing Parquet files are never overwritten. Outputs are written under
reports/bookraw/ and include all labeled blocks plus clean role-specific views.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile

import pandas as pd
from bs4 import BeautifulSoup
from ebooklib import epub


ROOT = Path(__file__).resolve().parents[2]
# Sources and outputs live on the data volume; see reports/bookraw/README.md.
DATA_ROOT = Path(os.environ.get("STYLEVEC_DATA", "/Volumes/EMC/stylevec_data"))
DATA = DATA_ROOT / "raw/human_author/bookraw_epub"
OUT = DATA_ROOT / "processed/bookraw"
MANIFEST_OUT = DATA_ROOT / "manifests/bookraw/epub20_legacy"
sys.path.insert(0, str(DATA))
CATALOGS = [
    MANIFEST_OUT / "work_catalog_english.jsonl",
    MANIFEST_OUT / "work_catalog_chinese_a.jsonl",
    MANIFEST_OUT / "work_catalog_chinese_b.jsonl",
]
ROLE_MARKERS = {
    "注释": "annotation",
    "注解": "annotation",
    "译文": "translation",
    "白话": "translation",
    "今译": "translation",
    "原文": "primary_text",
    "正文": "primary_text",
    "赏析": "commentary",
    "评析": "commentary",
    "题解": "commentary",
    "导读": "commentary",
    "前言": "editorial_matter",
    "序言": "editorial_matter",
    "译者序": "editorial_matter",
    "引言": "editorial_matter",
    "后记": "editorial_matter",
    "目录": "paratext",
    "版权": "paratext",
}
ROLE_CONFIDENCE = {
    "annotation": "high",
    "translation": "high",
    "primary_text": "high",
    "commentary": "medium",
    "editorial_matter": "high",
    "paratext": "high",
}
REF_RE = re.compile(r"\[(\d{1,6})\]")
# `get_text(" ")` splits on inline tags, so Chinese text arrives with a space at
# every inline boundary: `有蕡 其实 〔1〕` for `有蕡其实〔1〕`, and `〔1〕 。` with a
# stranded space. Both forms are typographically impossible, so they are safe to
# collapse. A space between two CJK characters is left alone: `二十七日 晴。` is a
# genuine Lu Xun diary heading, not an extraction artifact.
ZH_PUNCT_AFTER = re.compile(r"(?<=[，。、；：？！）」』】〕《〉·]) +")
ZH_SPACE_BEFORE_OPEN = re.compile(r" +(?=[（「『【〔《〈])")
ZH_SPACE_BEFORE_CLOSE = re.compile(r"(?<=[（「『【〔《〈]) +(?=[）」』】〕《〉])")
ZH_SPACE_AFTER_CLOSE = re.compile(r"(?<=[）」』】〕《〉]) +(?=[㐀-䶿（「『【〔《])")


def fix_zh_spacing(text: str) -> str:
    text = ZH_PUNCT_AFTER.sub("", text)
    text = ZH_SPACE_BEFORE_OPEN.sub("", text)
    text = ZH_SPACE_BEFORE_CLOSE.sub("", text)
    return ZH_SPACE_AFTER_CLOSE.sub("", text)


NOTE_START_RE = re.compile(r"^\s*\[\d{1,6}\]")
BOILERPLATE_RE = re.compile(
    r"^(?:copyright|all rights reserved|project gutenberg|gutenberg license|"
    r"电子版|版权所有|出版说明|本书.*?版权|isbn\b)|"
    r"this ebook is for the use|included with this ebook|gutenberg license",
    re.I,
)
EDITORIAL_CUE_RE = re.compile(
    r"^(?:译注|译者|注者|编者|整理者|校注者|作者简介|译者简介|编辑说明|"
    r"translated by|edited by|translated and annotated by|by\s+[A-Z])\b",
    re.I,
)
CHINESE_CREDIT_RE = re.compile(r"^[\u3400-\u9fff]{2,8}\s*(?:译注|校注|注释|译者|编者)(?:\s|$)")


def read_catalogs() -> list[dict]:
    rows = []
    for path in CATALOGS:
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["catalog_source"] = path.name
                row["catalog_line"] = line_no
                row["work_id"] = work_id(row)
                rows.append(row)
    return rows


def work_id(row: dict) -> str:
    key = "\0".join(str(row.get(k, "")) for k in ("source_file", "work_title", "author", "source_href"))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def norm(text: str) -> str:
    return re.sub(r"[\s\u3000《》〈〉「」『』【】]", "", text or "").casefold()


def href_parts(href: str) -> tuple[str, str]:
    parts = urlsplit(str(href or ""))
    return unquote(parts.path).lstrip("/"), unquote(parts.fragment)


def dom_blocks(html: bytes, lang: str) -> list[dict]:
    soup = BeautifulSoup(html, "xml")
    tags = soup.find_all(True)
    positions = {id(tag): i for i, tag in enumerate(tags)}
    headings = []
    paragraphs = []
    has_p = bool(soup.find("p"))
    for tag in tags:
        if re.fullmatch(r"h[1-6]", tag.name or ""):
            text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
            if text:
                headings.append({"pos": positions[id(tag)], "text": text, "tag": tag.name})
        if tag.name != "p" and tag.name != "div":
            continue
        if tag.name == "div" and tag.find(["p", "div", "table", "h1", "h2", "h3", "h4"]):
            continue
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text) < (1 if lang == "zh" else 3) and not tag.get("id"):
            continue
        classes = tag.get("class", [])
        if isinstance(classes, str):
            classes = classes.split()
        attrs = {str(k).casefold(): str(v) for k, v in tag.attrs.items()}
        paragraphs.append(
            {
                "pos": positions[id(tag)],
                # `get_text(" ")` joins every inline element with a space, which
                # splits Chinese words at inline-markup seams and strands a space
                # before fullwidth punctuation. Collapse only the spacing that is
                # typographically impossible; spaces between two CJK characters may
                # be genuine (diary dates, verse) and are kept.
                "text": fix_zh_spacing(text) if lang == "zh" else text,
                "tag": tag.name,
                "id": tag.get("id"),
                "classes": [str(x) for x in classes],
                "epub_type": attrs.get("epub:type", attrs.get("type", "")),
                "role_attr": " ".join(
                    [attrs.get("role", ""), attrs.get("epub:type", ""), attrs.get("type", "")]
                ).casefold(),
            }
        )
    return sorted(headings + paragraphs, key=lambda block: block["pos"])


def open_spine(path: Path) -> list[tuple[str, bytes]]:
    try:
        book = epub.read_epub(str(path), {"ignore_ncx": False})
        by_name = {item.file_name: item for item in book.get_items_of_type(9)}
        ordered = []
        for ref in book.spine:
            item = book.get_item_with_id(ref[0])
            if item and item.file_name in by_name:
                ordered.append((item.file_name, item.get_content()))
        return ordered
    except Exception:
        # Use the existing manual EPUB reader for older books with broken manifests.
        from extract_epub import _get_content, parse_epub_manual

        parsed = parse_epub_manual(path)
        return [
            (entry["href"], (_get_content(parsed, entry) or "").encode("utf-8"))
            for entry in parsed["spine"]
        ]


def locate_starts(rows: list[dict], filenames: list[str], page_blocks: list[list[dict]]) -> list[dict]:
    requested_pages = []
    for row in rows:
        requested, _ = href_parts(row.get("source_href", ""))
        matched = next((i for i, name in enumerate(filenames) if name.lstrip("/").endswith(requested)), None)
        if matched is None:
            basename = Path(requested).name
            candidates = [i for i, name in enumerate(filenames) if Path(name).name == basename]
            matched = candidates[0] if len(candidates) == 1 else None
        requested_pages.append(matched)
    ordered_pages = sorted(i for i in requested_pages if i is not None)
    starts = []
    for row_index, row in enumerate(rows):
        requested, fragment = href_parts(row.get("source_href", ""))
        page_index = requested_pages[row_index]
        if page_index is None:
            starts.append({"row": row, "spine_index": 0, "position": -1, "boundary_confidence": "low", "boundary_evidence": "source_href_unresolved"})
            continue
        following = [i for i in ordered_pages if i > page_index]
        search_end = min(following[0] if following else len(filenames) - 1, page_index + 50)
        page = page_blocks[page_index]
        anchor_pos = -1
        if fragment:
            for block in page:
                if block.get("id") == fragment:
                    anchor_pos = block["pos"]
                    break
        title = norm(str(row.get("work_title", "")))
        exact = []
        # Prefer exact title blocks at or immediately after the catalog pointer.
        for si in range(page_index, min(len(filenames), search_end + 1)):
            for block in page_blocks[si]:
                if norm(block.get("text", "")) == title:
                    exact.append((si, block["pos"]))
        if exact:
            si, pos = min(exact, key=lambda p: (max(0, p[0] - page_index), abs(p[1] - anchor_pos)))
            starts.append({"row": row, "spine_index": si, "position": pos, "boundary_confidence": "high", "boundary_evidence": "exact_work_heading"})
            continue
        # A headingless anchor is usable only when it points into real text.
        content_after = [b for b in page if b["pos"] >= anchor_pos and b.get("text")]
        if content_after and anchor_pos >= 0:
            starts.append({"row": row, "spine_index": page_index, "position": anchor_pos, "boundary_confidence": "medium", "boundary_evidence": "catalog_anchor_with_text"})
            continue
        first_text = next(
            ((si, b["pos"]) for si in range(page_index, search_end + 1) for b in page_blocks[si] if b.get("text")),
            None,
        )
        if first_text:
            si, pos = first_text
            confidence = "medium" if si <= page_index + 2 else "low"
            evidence = "first_text_after_empty_title_or_cover_page" if confidence == "medium" else "distant_text_after_unresolved_anchor"
            starts.append({"row": row, "spine_index": si, "position": pos, "boundary_confidence": confidence, "boundary_evidence": evidence})
        else:
            starts.append({"row": row, "spine_index": page_index, "position": 10**8, "boundary_confidence": "low", "boundary_evidence": "empty_or_cover_anchor_without_exact_heading"})
    return starts


def explicit_role(text: str) -> str | None:
    cleaned = re.sub(r"^[\s【〔\[（(]+|[\s】〕\]）)]+$", "", text).strip()
    cleaned = re.sub(r"[：:。]+$", "", cleaned).strip()
    for cue, role in ROLE_MARKERS.items():
        if cleaned == cue or cleaned.startswith(cue + "：") or cleaned.startswith(cue + ":"):
            return role
    return None


def author_language_role(row: dict) -> tuple[str, str, str]:
    author = str(row.get("author", "")).strip()
    lang = str(row.get("language", "zh"))
    if not author or any(x in author for x in ("多位作者", "佚名", "Unknown", "编者")):
        return "unknown", "unknown", "author_or_language_not_specific"
    if lang == "zh" and re.search(r"[\u3400-\u9fff]", author):
        return "primary_text", "medium", "catalog_author_and_chinese_text_language"
    if lang == "en" and not re.search(r"[\u3400-\u9fff]", author):
        return "primary_text", "medium", "catalog_author_and_english_text_language"
    if lang == "zh" and re.search(r"[A-Za-z]", author):
        return "translation", "medium", "foreign_author_with_chinese_text_language"
    return "unknown", "unknown", "language_role_not_established"


def catalog_title_role(row: dict) -> tuple[str, str, str] | None:
    title = str(row.get("work_title", "")).strip()
    if re.match(r"^(?:注释条目|注释|注解|校注|译注|集注)", title):
        return "annotation", "high", "catalog_title_identifies_notes"
    if re.match(r"^(?:赏析|评析|作品鉴赏|导读|题解)", title) or re.search(r"(?:赏析|评析|鉴赏)$", title):
        return "commentary", "high", "catalog_title_identifies_commentary"
    if title in {"序", "序言", "前言", "引言", "译者序", "译者前言", "后记", "编者按", "出版说明"}:
        return "editorial_matter", "high", "catalog_title_identifies_editorial_matter"
    return None


def classify_blocks(rows: list[dict], filenames: list[str], page_blocks: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    starts = locate_starts(rows, filenames, page_blocks)
    starts.sort(key=lambda x: (x["spine_index"], x["position"], x["row"]["catalog_source"], x["row"]["catalog_line"]))
    records = []
    for wi, start in enumerate(starts):
        row = start["row"]
        if start["boundary_confidence"] == "low":
            continue
        end = starts[wi + 1] if wi + 1 < len(starts) else None
        start_order = start["spine_index"] * 10**9 + start["position"]
        end_order = (end["spine_index"] * 10**9 + end["position"]) if end else 10**30
        base_role, base_conf, base_evidence = catalog_title_role(row) or author_language_role(row)
        if re.search(r"\btranslation by\s+([^.;,]+)", str(row.get("note", "")), re.I):
            base_role, base_conf, base_evidence = "translation", "high", "explicit_catalog_translation_note"
        source_profile = "zhonghua_classics" if "中华经典藏书" in str(row.get("source_file", "")) else "generic_epub"
        role_state = "unknown"
        role_state_conf = "unknown"
        role_state_evidence = "no explicit role heading"
        section_title = ""
        section_id = ""
        section_counter = 0
        has_chapter = False
        block_number = 0
        for si in range(start["spine_index"], len(filenames)):
            for block in page_blocks[si]:
                order = si * 10**9 + block["pos"]
                if order < start_order or order >= end_order:
                    continue
                text = block.get("text", "")
                if not text:
                    continue
                tag = block.get("tag", "")
                if norm(text) == norm(str(row.get("work_title", ""))):
                    section_title = text[:200]
                    if source_profile == "zhonghua_classics":
                        role_state = "commentary"
                        role_state_conf = "medium"
                        role_state_evidence = "introductory_material_after_work_title"
                    continue
                if tag.startswith("h"):
                    cue_role = explicit_role(text)
                    if cue_role:
                        role_state = cue_role
                        role_state_conf = ROLE_CONFIDENCE.get(cue_role, "medium")
                        role_state_evidence = f"heading:{text[:100]}"
                    else:
                        section_title = text[:200]
                        section_counter += 1
                        section_id = hashlib.sha1(
                            f"{row['work_id']}\0{si}\0{block['pos']}\0{section_title}".encode("utf-8")
                        ).hexdigest()[:16]
                        has_chapter = True
                        role_state = "commentary" if source_profile == "zhonghua_classics" else "unknown"
                        role_state_conf = "medium" if role_state == "commentary" else "unknown"
                        role_state_evidence = "chapter_intro_before_body_marker" if role_state == "commentary" else "unlabeled_section"
                    continue
                cue_role = explicit_role(text)
                if cue_role:
                    role_state = cue_role
                    role_state_conf = ROLE_CONFIDENCE.get(cue_role, "medium")
                    role_state_evidence = f"inline_heading:{text[:100]}"
                    if len(text) <= 30:
                        continue

                attrs = block.get("role_attr", "")
                classes = set(block.get("classes", []))
                if "footnote" in attrs or "endnote" in attrs or "fnote1" in classes or NOTE_START_RE.match(text) or re.match(r"^注[：:]", text):
                    role, confidence, evidence = "annotation", "high", "epub_note_semantics_or_numbered_note"
                elif classes.intersection({"fig", "figure", "caption", "cover", "image"}):
                    role, confidence, evidence = "paratext", "high", "image_caption_or_placeholder_class"
                elif cue_role:
                    role, confidence, evidence = cue_role, ROLE_CONFIDENCE.get(cue_role, "medium"), f"inline_role_label:{text[:100]}"
                elif EDITORIAL_CUE_RE.match(text) or CHINESE_CREDIT_RE.match(text) or BOILERPLATE_RE.search(text):
                    role, confidence, evidence = "paratext", "high", "publication_or_credit_marker"
                elif source_profile == "zhonghua_classics" and ("x17" in classes or "kindle-cn-para-left1" in classes or "kindle-cn-hei1" in classes):
                    role, confidence, evidence = "primary_text", "high", "reviewed_publisher_body_style"
                    role_state = role
                    role_state_conf = confidence
                    role_state_evidence = evidence
                elif source_profile == "zhonghua_classics" and "kindle-cn-kai1" in classes and role_state not in {"translation", "annotation"}:
                    role, confidence, evidence = "commentary", "medium", "reviewed_publisher_commentary_style"
                    role_state = role
                    role_state_conf = confidence
                    role_state_evidence = evidence
                elif source_profile == "zhonghua_classics" and "x1" in classes and role_state == "unknown":
                    role, confidence, evidence = "commentary", "medium", "unlabeled_introductory_paragraph_in_classics_edition"
                elif role_state != "unknown":
                    role, confidence, evidence = role_state, role_state_conf, role_state_evidence
                    # A cue marker labels the block that follows it, not the rest
                    # of the work. Consuming it once here stops a stray `后记`
                    # near the start of a book from relabelling the whole novel
                    # as editorial matter, which is what leaked before.
                    role_state = "unknown"
                    role_state_conf = "unknown"
                    role_state_evidence = "role_state_cleared_after_single_use"
                elif BOILERPLATE_RE.match(text):
                    role, confidence, evidence = "paratext", "high", "publication_boilerplate_pattern"
                elif base_role != "unknown":
                    role, confidence, evidence = base_role, base_conf, base_evidence
                else:
                    role, confidence, evidence = "unknown", "unknown", "no_role_evidence"

                wid = row["work_id"]
                if "reviewed_publisher_" in evidence:
                    role_source = "source_specific_css_rule"
                elif evidence.startswith("heading:") or evidence.startswith("inline_heading:") or evidence.startswith("inline_role_label:"):
                    role_source = "epub_heading_rule"
                elif evidence.startswith("catalog_author_") or evidence.startswith("foreign_author_"):
                    role_source = "author_language_heuristic"
                elif evidence.startswith("explicit_catalog_") or evidence.startswith("catalog_"):
                    role_source = "curated_catalog"
                elif "numbered_note" in evidence or "epub_note" in evidence:
                    role_source = "epub_semantics_or_note_marker"
                elif "pattern" in evidence or "marker" in evidence:
                    role_source = "text_pattern_rule"
                else:
                    role_source = "section_state_or_default_rule"
                block_id = hashlib.sha1(
                    f"{row['source_file']}\0{filenames[si]}\0{block['pos']}\0{block_number}".encode("utf-8")
                ).hexdigest()[:20]
                block_number += 1
                records.append(
                    {
                        "text": text,
                        "author": row.get("author") or "Unknown",
                        "work": row.get("work_title", ""),
                        "work_id": wid,
                        "genre": row.get("genre", "unknown"),
                        "lang": row.get("language", "zh"),
                        "text_role": role,
                        "role_confidence": confidence,
                        "role_evidence": evidence,
                        "role_source": role_source,
                        "boundary_confidence": start["boundary_confidence"],
                        "boundary_evidence": start["boundary_evidence"],
                        "section_id": section_id,
                        "section_title": section_title,
                        "source_file": row["source_file"],
                        "source_href": row.get("source_href", ""),
                        "source_spine_file": filenames[si],
                        "source_spine_index": si,
                        "source_dom_position": block["pos"],
                        "source_block_id": block_id,
                        "source_tag": tag,
                        "source_html_id": block.get("id"),
                        "source_html_classes": block.get("classes", []),
                        "source_epub_type": block.get("epub_type", ""),
                        "catalog_source": row["catalog_source"],
                        "catalog_line": row["catalog_line"],
                        "reference_ids": REF_RE.findall(text),
                    }
                )
            if (si + 1) * 10**9 >= end_order:
                break
    return records, starts


def build_relations(blocks: list[dict]) -> list[dict]:
    relations = []
    by_work: dict[str, list[dict]] = defaultdict(list)
    for row in blocks:
        by_work[row["work_id"]].append(row)
    for work, rows in by_work.items():
        references: dict[tuple[str, str], list[str]] = defaultdict(list)
        pending_primary: list[str] = []
        active_targets: list[str] = []
        previous_role = ""
        for row in rows:
            rid = row["source_block_id"]
            section = row.get("section_id", "")
            if row["text_role"] == "primary_text":
                if previous_role in {"translation", "annotation", "commentary", "editorial_matter"}:
                    pending_primary = []
                pending_primary.append(rid)
                active_targets = []
                for ref in row["reference_ids"]:
                    references[(section, ref)].append(rid)
            elif row["text_role"] == "annotation":
                refs = set(row["reference_ids"])
                for ref in refs:
                    for target in references.get((section, ref), []):
                        relations.append({
                            "from_passage_id": target,
                            "to_passage_id": rid,
                            "relation_type": "annotates",
                            "alignment_method": "printed_reference_number",
                            "confidence": "high",
                            "work_id": work,
                        })
            elif row["text_role"] == "translation":
                if pending_primary:
                    active_targets = list(dict.fromkeys(pending_primary))
                for target in active_targets:
                    relations.append({
                        "from_passage_id": target,
                        "to_passage_id": rid,
                        "relation_type": "translates",
                        "alignment_method": "same_section_order",
                        "confidence": "medium",
                        "work_id": work,
                    })
            previous_role = row["text_role"]
    unique = {}
    for row in relations:
        key = (row["from_passage_id"], row["to_passage_id"], row["relation_type"])
        unique[key] = row
    return list(unique.values())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    global DATA, OUT, CATALOGS, MANIFEST_OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    args = parser.parse_args()
    DATA = args.data_root
    OUT = args.output_dir
    MANIFEST_OUT = args.manifest_dir or OUT
    CATALOGS = [DATA / path.name for path in CATALOGS]
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(DATA))
    catalog = read_catalogs()
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in catalog:
        if row.get("work_title") and row.get("source_href"):
            by_source[row["source_file"]].append(row)

    start_manifest = [
        {
            "work_id": row["work_id"],
            "work": row.get("work_title", ""),
            "author": row.get("author", ""),
            "source_file": row.get("source_file", ""),
            "source_href": row.get("source_href", ""),
            "boundary_confidence": "low",
            "boundary_evidence": "missing_work_title_or_source_href",
            "block_count": 0,
        }
        for row in catalog
        if not row.get("work_title") or not row.get("source_href")
    ]

    all_blocks = []
    for source, works in sorted(by_source.items()):
        path = DATA / source
        if not path.is_file() or path.suffix.lower() != ".epub":
            for row in works:
                start_manifest.append({"work_id": row["work_id"], "work": row["work_title"], "source_file": source, "boundary_confidence": "low", "boundary_evidence": "source_file_missing_or_not_epub", "block_count": 0})
            continue
        try:
            spine = open_spine(path)
        except (BadZipFile, OSError, ValueError) as exc:
            for row in works:
                start_manifest.append({"work_id": row["work_id"], "work": row["work_title"], "source_file": source, "boundary_confidence": "low", "boundary_evidence": f"epub_read_error:{type(exc).__name__}", "block_count": 0})
            continue
        filenames = [name for name, _ in spine]
        page_blocks = [dom_blocks(payload, "zh") for _, payload in spine]
        source_blocks, starts = classify_blocks(works, filenames, page_blocks)
        all_blocks.extend(source_blocks)
        block_counts = Counter(block["work_id"] for block in source_blocks)
        for start in starts:
            row = start["row"]
            count = block_counts[row["work_id"]]
            start_manifest.append({
                "work_id": row["work_id"],
                "work": row["work_title"],
                "author": row.get("author", ""),
                "source_file": source,
                "source_href": row.get("source_href", ""),
                "resolved_spine_index": start["spine_index"],
                "boundary_confidence": start["boundary_confidence"],
                "boundary_evidence": start["boundary_evidence"],
                "block_count": count,
            })
        print(f"{source}: {len(source_blocks)} tagged blocks for {len(works)} catalog works")

    blocks_frame = pd.DataFrame(all_blocks)
    role_counts_by_work: dict[str, dict[str, int]] = defaultdict(dict)
    for (wid, role), count in Counter((b["work_id"], b["text_role"]) for b in all_blocks).items():
        role_counts_by_work[wid][role] = count
    boundary_by_work = {row["work_id"]: row for row in start_manifest}
    catalog_rows = []
    for row in catalog:
        boundary = boundary_by_work.get(row["work_id"], {})
        catalog_rows.append({
            **{key: value for key, value in row.items() if key not in {"catalog_source", "catalog_line"}},
            "catalog_source": row["catalog_source"],
            "catalog_line": row["catalog_line"],
            "block_count": boundary.get("block_count", 0),
            "text_role_counts": role_counts_by_work.get(row["work_id"], {}),
            "boundary_confidence": boundary.get("boundary_confidence", "low"),
            "boundary_evidence": boundary.get("boundary_evidence", "no_resolved_start"),
        })
    if not blocks_frame.empty:
        blocks_frame.to_parquet(OUT / "books_work_blocks_labeled.parquet", index=False)
        clean_roles = {"primary_text", "translation", "annotation", "commentary", "editorial_matter", "paratext"}
        # `blocks_frame` only ever emits those six roles -- anything else is
        # dropped as `unknown` before a record is built -- so this filter is a
        # no-op today and the two files are byte-identical. Keep the subset so
        # the contract holds if a seventh role is ever introduced.
        passages = blocks_frame[blocks_frame["text_role"].isin(clean_roles)]
        passages.to_parquet(OUT / "books_work_passages_labeled.parquet", index=False)
        primary_mask = (
            (blocks_frame["text_role"] == "primary_text")
            & (blocks_frame["role_confidence"].isin(["high", "medium"]))
            & (blocks_frame["boundary_confidence"].isin(["high", "medium"]))
            & (
                ((blocks_frame["lang"] == "zh") & (blocks_frame["text"].str.len() >= 30))
                | ((blocks_frame["lang"] != "zh") & (blocks_frame["text"].str.len() >= 80))
            )
        )
        blocks_frame[primary_mask].to_parquet(OUT / "books_work_primary_clean.parquet", index=False)
        strict_mask = (
            (blocks_frame["text_role"] == "primary_text")
            & (blocks_frame["role_confidence"] == "high")
            & (blocks_frame["boundary_confidence"] == "high")
            & (
                ((blocks_frame["lang"] == "zh") & (blocks_frame["text"].str.len() >= 30))
                | ((blocks_frame["lang"] != "zh") & (blocks_frame["text"].str.len() >= 80))
            )
        )
        blocks_frame[strict_mask].to_parquet(OUT / "books_work_primary_strict.parquet", index=False)
        blocks_frame[blocks_frame["text_role"] == "translation"].to_parquet(
            OUT / "books_work_translations.parquet", index=False
        )
    write_jsonl(MANIFEST_OUT / "work_boundary_manifest.jsonl", start_manifest)
    write_jsonl(MANIFEST_OUT / "works_catalog_labeled.jsonl", catalog_rows)
    write_jsonl(MANIFEST_OUT / "passage_relations.jsonl", build_relations(all_blocks))
    print(f"Wrote {len(all_blocks)} source blocks: {OUT / 'books_work_blocks_labeled.parquet'}")
    print("Role counts:", dict(Counter(row["text_role"] for row in all_blocks)))


if __name__ == "__main__":
    main()
