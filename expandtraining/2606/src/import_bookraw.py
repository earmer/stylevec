from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from config import resolve_repo_path
from records import SampleRecord, normalize_text, stable_id


def load_bookraw_samples(config: dict[str, Any]) -> tuple[list[SampleRecord], dict[str, int]]:
    if not bool(config["data"].get("bookraw_enabled", False)):
        return [], {}

    path = resolve_repo_path(str(config["paths"]["bookraw_cleaned"]))
    frame = pd.read_parquet(path)
    required = {"text", "author", "work", "lang"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bookraw parquet missing columns {sorted(missing)}: {path}")

    min_chars = int(config["data"].get("bookraw_min_chars", config["data"]["min_chars"]))
    max_chars = int(config["data"].get("bookraw_max_chars", config["data"]["max_chars"]))
    author_cap = int(config["data"]["bookraw_authors"])
    per_author = int(config["data"]["bookraw_segments_per_author"])
    per_work = int(config["data"]["bookraw_segments_per_work"])
    min_author_segments = int(config["data"].get("bookraw_min_segments_per_author", 2))
    min_work_segments = int(config["data"].get("bookraw_min_segments_per_work", 2))

    filter_counts: defaultdict[str, int] = defaultdict(int)
    rows_by_author: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_texts: set[str] = set()

    for row_index, row in frame.iterrows():
        author = normalize_text(str(row["author"]))
        work = normalize_text(str(row["work"]))
        lang = normalize_text(str(row["lang"]))
        text = normalize_text(str(row["text"]))
        if not author or not work or not lang or not text:
            filter_counts["empty_field"] += 1
            continue
        if not min_chars <= len(text) <= max_chars:
            filter_counts["bad_length"] += 1
            continue
        text_key = text.casefold()
        if text_key in seen_texts:
            filter_counts["duplicate_text"] += 1
            continue
        seen_texts.add(text_key)
        rows_by_author[author].append(
            {
                "row_index": row_index,
                "author": author,
                "work": work,
                "lang": lang,
                "text": text,
            }
        )

    usable_authors: list[tuple[str, int, int]] = []
    for author, rows in rows_by_author.items():
        works = defaultdict(int)
        for row in rows:
            works[str(row["work"])] += 1
        usable_works = sum(1 for count in works.values() if count >= min_work_segments)
        usable_rows = sum(1 for row in rows if works[str(row["work"])] >= min_work_segments)
        if usable_rows < min_author_segments or usable_works == 0:
            filter_counts["author_shortage"] += 1
            continue
        usable_authors.append((author, usable_works, usable_rows))

    usable_authors.sort(key=lambda item: (item[1] < 2, -item[1], -item[2], item[0]))
    selected_author_rows = usable_authors if author_cap <= 0 else usable_authors[:author_cap]
    selected_authors = {author for author, _works, _rows in selected_author_rows}
    samples: list[SampleRecord] = []

    for author in sorted(selected_authors):
        rows = rows_by_author[author]
        rows_by_work: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_work[str(row["work"])].append(row)
        for work_rows in rows_by_work.values():
            work_rows.sort(key=lambda row: int(row["row_index"]))

        selected_rows: list[dict[str, Any]] = []
        works = sorted(rows_by_work, key=lambda work: (-len(rows_by_work[work]), work))
        for work in works:
            work_rows = rows_by_work[work]
            if len(work_rows) < min_work_segments:
                filter_counts["work_shortage"] += 1
                continue
            room = per_author - len(selected_rows) if per_author > 0 else len(work_rows)
            if per_author > 0 and room <= 0:
                filter_counts["author_cap"] += 1
                break
            work_limit = len(work_rows) if per_work <= 0 else per_work
            selected_rows.extend(work_rows[: min(work_limit, room)])

        if len(selected_rows) < min_author_segments:
            filter_counts["selected_author_shortage"] += 1
            continue

        val_count = min(max(2, len(selected_rows) // 5), max(1, len(selected_rows) - 2))
        train_cutoff = len(selected_rows) - val_count
        for selected_index, row in enumerate(selected_rows):
            split = "train" if selected_index < train_cutoff else "val"
            text = str(row["text"])
            work = str(row["work"])
            lang = str(row["lang"])
            samples.append(
                SampleRecord(
                    sample_id=stable_id("bookraw", row["row_index"]),
                    text=text,
                    style_id=author,
                    style_type="author",
                    domain="bookraw",
                    content_id=stable_id("bookraw_work", author, work),
                    source_id=f"{path.name}:{row['row_index']}",
                    language=lang,
                    split=split,
                    quality=0.85,
                    meta={
                        "author": author,
                        "work": work,
                        "lang": lang,
                        "length": len(text),
                        "length_bucket": len(text) // 200,
                    },
                )
            )

    if not samples:
        raise ValueError("bookraw importer produced no samples")
    return samples, dict(filter_counts)
