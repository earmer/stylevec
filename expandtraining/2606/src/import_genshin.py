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
# Present in almost every quest scene: as the nearest scene neighbour they were 74% of
# same_speaker_topic_matched negatives (d-fe32ff5dc617), so the rule mostly taught "not Paimon".
HUB_SPEAKERS = ("派蒙", "旅行者")


def hub_speakers(config: dict[str, Any]) -> frozenset[str]:
    return frozenset(config["data"].get("genshin_hub_speakers", HUB_SPEAKERS))


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


GENSHIN_SOURCES = ("v1", "v2")


def genshin_source(config: dict[str, Any]) -> str:
    """`data.genshin_source`: "v1" (genshin.db, scenes guessed from nearby DB ids; the default) or "v2"
    (genshin_v2.db, whose talk_id is the real conversation)."""
    source = str(config["data"].get("genshin_source", "v1"))
    if source not in GENSHIN_SOURCES:
        raise ValueError(f"data.genshin_source must be one of {GENSHIN_SOURCES}, got {source!r}")
    return source


def load_genshin_samples(config: dict[str, Any], speakers: list[str]) -> tuple[list[SampleRecord], dict[str, int]]:
    if genshin_source(config) == "v2":
        return load_genshin_v2_samples(config, speakers)
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
            is_heldout = speaker in heldout
            selected_rows = select_speaker_rows(usable, per_speaker, seen_texts, filter_counts)
            for speaker_kept, ((row_id, origin, para), split) in enumerate(
                zip(selected_rows, speaker_splits(len(selected_rows), is_heldout))
            ):
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
    attach_scene_neighbors(samples, scene_lines, heldout, scene_window, scene_neighbors_max, hub_speakers(config))
    if not samples:
        raise ValueError("Genshin importer produced no samples")
    return samples, dict(filter_counts)


def select_speaker_rows(
    usable: list[tuple[Any, ...]],
    per_speaker: int,
    seen_texts: set[str],
    filter_counts: defaultdict[str, int],
) -> list[tuple[Any, ...]]:
    """A speaker's usable rows (row[1] is the line) capped at `per_speaker`, minus lines already taken."""
    # Spread the per-speaker cap over the whole id range; "first N" would be one early quest.
    if per_speaker > 0 and len(usable) > per_speaker:
        picks = sorted({int(i) for i in np.linspace(0, len(usable) - 1, per_speaker)})
        candidates = [usable[i] for i in picks]
    else:
        candidates = usable
    selected: list[tuple[Any, ...]] = []
    for row in candidates:
        text_key = row[1].casefold()
        if text_key in seen_texts:
            filter_counts["duplicate_origin"] += 1
            continue
        seen_texts.add(text_key)
        selected.append(row)
    return selected


def speaker_splits(count: int, is_heldout: bool) -> list[str]:
    """A held-out speaker's lines all go to val; otherwise the last fifth (at least 2) does."""
    if is_heldout:
        return ["val"] * count
    val_per_speaker = min(max(2, count // 5), max(1, count - 2))
    return ["val" if kept >= count - val_per_speaker else "train" for kept in range(count)]


def load_genshin_v2_samples(config: dict[str, Any], speakers: list[str]) -> tuple[list[SampleRecord], dict[str, int]]:
    """genshin_v2.db: lines carry their conversation (talk_id) and turn. Rows marked dup_of are skipped.

    Same selection and splits as v1. Scene neighbours become other speakers' lines from the same
    conversation (any speaker, not only the top ones), and the far positive a line from another
    conversation; see attach_conversation_neighbors."""
    db_path = resolve_repo_path(str(config["paths"].get("genshin_v2_db", "data/genshin/genshin_v2.db")))
    data = config["data"]
    min_chars, max_chars = int(data["min_chars"]), int(data["max_chars"])
    per_speaker = int(data["genshin_lines_per_speaker"])
    samples: list[SampleRecord] = []
    filter_counts: defaultdict[str, int] = defaultdict(int)
    seen_texts: set[str] = set()
    heldout = heldout_speakers(config, speakers)
    wanted = set(speakers)
    by_speaker: defaultdict[str, list[tuple[int, str, str, int, int]]] = defaultdict(list)
    talk_lines: defaultdict[int, list[tuple[int, str, str, int]]] = defaultdict(list)  # (turn, speaker, text, id)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            """
            select id, talk_id, turn_id, speaker, origin_text, para_text
            from dialogues
            where dup_of is null and origin_text is not null and trim(origin_text) != ''
            order by id
            """
        ).fetchall()
    for row_id, talk_id, turn_id, speaker, origin_text, para_text in rows:
        origin = normalize_text(origin_text)
        if is_junk(origin) or speaker in PLACEHOLDER_SPEAKERS or not min_chars <= len(origin) <= max_chars:
            if speaker in wanted:
                filter_counts["junk_placeholder" if is_junk(origin) else "bad_origin_length"] += 1
            continue
        talk_lines[int(talk_id)].append((int(turn_id), speaker, origin, int(row_id)))
        if speaker not in wanted:
            continue
        para = normalize_text(para_text or "")
        if not min_chars <= len(para) <= max_chars:
            filter_counts["bad_para_length"] += 1
            continue
        by_speaker[speaker].append((int(row_id), origin, para, int(talk_id), int(turn_id)))
    for speaker in speakers:
        is_heldout = speaker in heldout
        selected = select_speaker_rows(by_speaker[speaker], per_speaker, seen_texts, filter_counts)
        for kept, ((row_id, origin, para, talk_id, turn_id), split) in enumerate(
            zip(selected, speaker_splits(len(selected), is_heldout))
        ):
            samples.append(
                SampleRecord(
                    sample_id=stable_id("genshin_v2", row_id, "origin"),
                    text=origin,
                    style_id=speaker,
                    style_type="speaker",
                    domain="genshin",
                    content_id=stable_id("genshin_v2_content", row_id),
                    source_id=f"v2:{row_id}",
                    language="zh",
                    split=split,
                    quality=1.0,
                    meta={
                        "para_text": para,
                        "speaker": speaker,
                        "heldout": is_heldout,
                        "order": kept,
                        "db_id": row_id,
                        "talk_id": talk_id,
                        "turn_id": turn_id,
                        "length": len(origin),
                        "length_bucket": len(origin) // 20,
                    },
                )
            )
        if len(selected) < 2:
            filter_counts["speaker_shortage"] += 1
    attach_conversation_neighbors(
        samples, talk_lines, heldout, int(data.get("genshin_scene_neighbors", 3)), hub_speakers(config),
        float(data.get("genshin_echo_jaccard", 0.5)),
    )
    if not samples:
        raise ValueError("Genshin v2 importer produced no samples")
    return samples, dict(filter_counts)


def char_bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)} or {text}


def attach_conversation_neighbors(
    samples: list[SampleRecord],
    talk_lines: dict[int, list[tuple[int, str, str, int]]],
    heldout: set[str],
    limit: int,
    hubs: frozenset[str] = frozenset(),
    echo_jaccard: float = 0.5,
) -> None:
    """v2's scene neighbours: other speakers' lines from the anchor's own conversation, nearest turn first.

    Same leak rules as attach_scene_neighbors (train anchors never get a held-out speaker's line or a
    validation line) and the same non-hub-then-hub order. A line that echoes the anchor (character-bigram
    Jaccard above `echo_jaccard`, e.g. a reply repeating the question) is dropped: it would share more
    words with the anchor than any far positive can. The distance kept is in turns."""
    split_by_id = {int(s.meta["db_id"]): s.split for s in samples}
    for sample in samples:
        anchor_grams = char_bigrams(sample.text)
        turn = int(sample.meta["turn_id"])
        near: list[tuple[int, str, str]] = []
        for other_turn, speaker, text, row_id in talk_lines.get(int(sample.meta["talk_id"]), []):
            if speaker == sample.style_id or row_id == int(sample.meta["db_id"]):
                continue
            if sample.split == "train" and (speaker in heldout or split_by_id.get(row_id) == "val"):
                continue
            grams = char_bigrams(text)
            if len(anchor_grams & grams) / len(anchor_grams | grams) > echo_jaccard:
                continue
            near.append((abs(other_turn - turn), speaker, text))
        near.sort()
        others = [row for row in near if row[1] not in hubs][:limit]
        fallback = [row for row in near if row[1] in hubs][:limit]
        sample.meta["scene_neighbors"] = [[sp, tx, d] for d, sp, tx in others + fallback]


def attach_scene_neighbors(
    samples: list[SampleRecord],
    scene_lines: list[tuple[int, str, str]],
    heldout: set[str],
    window: int,
    limit: int,
    hubs: frozenset[str] = frozenset(),
) -> None:
    """Give each line the nearest other-speaker lines by DB id (same scene, same topic).

    A neighbour is only offered if using it cannot leak across splits: train anchors never get a
    held-out speaker's line or a validation line; validation anchors may use anything.
    Keeps the `limit` nearest non-hub lines, then the `limit` nearest hub lines as a fallback, so the
    triplet builder can prefer a non-hub speaker without losing anchors whose scene has only hubs.
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
        others = [row for row in near if row[1] not in hubs][:limit]
        fallback = [row for row in near if row[1] in hubs][:limit]
        sample.meta["scene_neighbors"] = [[sp, tx, d] for d, sp, tx in others + fallback]
