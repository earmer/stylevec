from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any

from config import resolve_repo_path
from records import SampleRecord, normalize_text, stable_id


PLACEHOLDER_SPEAKERS = {"？？？", "？？", "???", "？"}


def load_top_speakers(config: dict[str, Any]) -> list[str]:
    speaker_limit = int(config["data"]["speakers"])
    path = resolve_repo_path(str(config["paths"]["top_speakers"]))
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"top speakers file must contain a list: {path}")
    speakers: list[str] = []
    for row in rows:
        speaker = row.get("speaker") if isinstance(row, dict) else row
        if isinstance(speaker, str) and speaker.strip() and speaker not in PLACEHOLDER_SPEAKERS:
            speakers.append(speaker)
    if len(speakers) < speaker_limit:
        raise ValueError(f"top speakers file has {len(speakers)} speakers, need {speaker_limit}")
    return speakers[:speaker_limit]


def load_genshin_samples(config: dict[str, Any], speakers: list[str]) -> tuple[list[SampleRecord], dict[str, int]]:
    db_path = resolve_repo_path(str(config["paths"]["genshin_db"]))
    min_chars = int(config["data"]["min_chars"])
    max_chars = int(config["data"]["max_chars"])
    per_speaker = int(config["data"]["genshin_lines_per_speaker"])
    samples: list[SampleRecord] = []
    filter_counts: defaultdict[str, int] = defaultdict(int)
    seen_texts: set[str] = set()

    with sqlite3.connect(db_path) as connection:
        for speaker in speakers:
            rows = connection.execute(
                """
                select id, origin_text, para_text
                from dialogues
                where speaker = ?
                  and origin_text is not null
                  and trim(origin_text) != ''
                  and para_text is not null
                  and trim(para_text) != ''
                order by id
                """,
                (speaker,),
            ).fetchall()
            selected_rows: list[tuple[Any, Any, Any]] = []
            for row_id, origin_text, para_text in rows:
                origin = normalize_text(origin_text)
                para = normalize_text(para_text)
                if not min_chars <= len(origin) <= max_chars:
                    filter_counts["bad_origin_length"] += 1
                    continue
                if not min_chars <= len(para) <= max_chars:
                    filter_counts["bad_para_length"] += 1
                    continue
                text_key = origin.casefold()
                if text_key in seen_texts:
                    filter_counts["duplicate_origin"] += 1
                    continue
                seen_texts.add(text_key)
                selected_rows.append((row_id, origin, para))
                if per_speaker > 0 and len(selected_rows) >= per_speaker:
                    break
            val_per_speaker = min(max(2, len(selected_rows) // 5), max(1, len(selected_rows) - 2))
            train_cutoff = len(selected_rows) - val_per_speaker
            for speaker_kept, (row_id, origin, para) in enumerate(selected_rows):
                split = "val" if speaker_kept >= train_cutoff else "train"
                sample = SampleRecord(
                    sample_id=stable_id("genshin", row_id, "origin"),
                    text=origin,
                    style_id=speaker,
                    style_type="speaker",
                    domain="genshin",
                    content_id=stable_id("genshin_content", row_id),
                    source_id=str(row_id),
                    language="zh",
                    split=split,
                    quality=1.0,
                    meta={
                        "para_text": para,
                        "speaker": speaker,
                        "length": len(origin),
                        "length_bucket": len(origin) // 20,
                    },
                )
                samples.append(sample)
            if len(selected_rows) < 2:
                filter_counts["speaker_shortage"] += 1
    if not samples:
        raise ValueError("Genshin importer produced no samples")
    return samples, dict(filter_counts)
