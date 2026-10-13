from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from bisect import bisect_left
from typing import Any

import numpy as np

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


JUNK_MARKERS = ("测试对话", "占位")


def is_junk(text: str) -> bool:
    return any(marker in text for marker in JUNK_MARKERS)


def heldout_speakers(config: dict[str, Any], speakers: list[str]) -> set[str]:
    """Last `data.heldout_speakers` speakers: every line goes to val, none to train."""
    count = int(config["data"].get("heldout_speakers", 0))
    if count < 0 or count >= len(speakers):
        raise ValueError(f"data.heldout_speakers={count} invalid for {len(speakers)} speakers")
    return set(speakers[len(speakers) - count :]) if count else set()


def load_genshin_samples(config: dict[str, Any], speakers: list[str]) -> tuple[list[SampleRecord], dict[str, int]]:
    db_path = resolve_repo_path(str(config["paths"]["genshin_db"]))
    min_chars = int(config["data"]["min_chars"])
    max_chars = int(config["data"]["max_chars"])
    per_speaker = int(config["data"]["genshin_lines_per_speaker"])
    samples: list[SampleRecord] = []
    filter_counts: defaultdict[str, int] = defaultdict(int)
    seen_texts: set[str] = set()
    heldout = heldout_speakers(config, speakers)

    scene_window = int(config["data"].get("genshin_scene_window", 12))
    scene_neighbors_max = int(config["data"].get("genshin_scene_neighbors", 3))
    scene_lines: list[tuple[int, str, str]] = []  # (db id, speaker, text) for every usable top-speaker line
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
            usable: list[tuple[Any, Any, Any]] = []
            for row_id, origin_text, para_text in rows:
                origin = normalize_text(origin_text)
                para = normalize_text(para_text)
                if is_junk(origin):
                    filter_counts["junk_placeholder"] += 1
                    continue
                if not min_chars <= len(origin) <= max_chars:
                    filter_counts["bad_origin_length"] += 1
                    continue
                if not min_chars <= len(para) <= max_chars:
                    filter_counts["bad_para_length"] += 1
                    continue
                usable.append((row_id, origin, para))
                scene_lines.append((int(row_id), speaker, origin))
            selected_rows: list[tuple[Any, Any, Any]] = []
            # Spread the per-speaker cap over the whole id range; "first N" would be one early quest.
            if per_speaker > 0 and len(usable) > per_speaker:
                picks = sorted({int(i) for i in np.linspace(0, len(usable) - 1, per_speaker)})
                candidates = [usable[i] for i in picks]
            else:
                candidates = usable
            for row_id, origin, para in candidates:
                text_key = origin.casefold()
                if text_key in seen_texts:
                    filter_counts["duplicate_origin"] += 1
                    continue
                seen_texts.add(text_key)
                selected_rows.append((row_id, origin, para))
            val_per_speaker = min(max(2, len(selected_rows) // 5), max(1, len(selected_rows) - 2))
            train_cutoff = len(selected_rows) - val_per_speaker
            is_heldout = speaker in heldout
            for speaker_kept, (row_id, origin, para) in enumerate(selected_rows):
                if is_heldout:
                    split = "val"
                else:
                    split = "val" if speaker_kept >= train_cutoff else "train"
                samples.append(
                    SampleRecord(
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
                            "heldout": is_heldout,
                            "order": speaker_kept,
                            "db_id": int(row_id),
                            "length": len(origin),
                            "length_bucket": len(origin) // 20,
                        },
                    )
                )
            if len(selected_rows) < 2:
                filter_counts["speaker_shortage"] += 1
    attach_scene_neighbors(samples, scene_lines, heldout, scene_window, scene_neighbors_max)
    if not samples:
        raise ValueError("Genshin importer produced no samples")
    return samples, dict(filter_counts)


def attach_scene_neighbors(
    samples: list[SampleRecord],
    scene_lines: list[tuple[int, str, str]],
    heldout: set[str],
    window: int,
    limit: int,
) -> None:
    """Give each line the nearest other-speaker lines by DB id (same scene, same topic).

    A neighbour is only offered if using it cannot leak across splits: train anchors never get a
    held-out speaker's line or a validation line; validation anchors may use anything.
    """
    ordered = sorted(scene_lines)
    ids = [row[0] for row in ordered]
    split_by_id = {int(s.meta["db_id"]): s.split for s in samples}
    for sample in samples:
        center = int(sample.meta["db_id"])
        lo = bisect_left(ids, center - window)
        near: list[tuple[int, str, str]] = []
        for row_id, speaker, text in ordered[lo:]:
            if row_id > center + window:
                break
            if speaker == sample.style_id or row_id == center:
                continue
            if sample.split == "train" and (speaker in heldout or split_by_id.get(row_id) == "val"):
                continue
            near.append((abs(row_id - center), speaker, text))
        near.sort()
        sample.meta["scene_neighbors"] = [[sp, tx, d] for d, sp, tx in near[:limit]]
