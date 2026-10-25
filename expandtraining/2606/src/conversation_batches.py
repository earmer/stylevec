"""Conversation batches: put several speakers' lines from one conversation in the same micro-batch.

In-batch InfoNCE scores every anchor against every other anchor in its micro-batch. With rows drawn at
random those anchors come from unrelated scenes, so topic alone tells them apart. `train.conversation_group:
k` (0 = off) rewrites each micro-batch of an already ordered list so its Genshin rows come in groups of up
to k rows whose anchors share a conversation (genshin v2's talk_id) but not a speaker: the in-batch
negatives then share the anchor's topic, and only voice separates them.

The rewrite is a swap, never an insert: a row is replaced by a row of the same rule from the leader's
conversation, so each micro-batch keeps its rule counts and the curriculum's timing is unchanged. Rows
without a talk_id (v1 data, other sources) are left where they are. Replacement rows may repeat rows used
elsewhere; a conversation with fewer than two eligible speakers leaves its row a singleton.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def conversation_group(config: dict[str, Any]) -> int:
    group = int(config["train"].get("conversation_group", 0))
    if group < 0 or group == 1:
        raise ValueError("train.conversation_group must be 0 (off) or >= 2")
    return group


def _talk(row: Any) -> int | None:
    talk = row.meta.get("talk_id")
    return None if talk is None else int(talk)


def group_by_conversation(rows: list[Any], pool: list[Any], batch_size: int, group: int, seed: int) -> list[Any]:
    """`rows` reordered micro-batch by micro-batch (see the module docstring). `pool` is every train row
    that may be swapped in (normally the whole train split)."""
    if group < 2:
        return rows
    index: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for row in pool:
        talk = _talk(row)
        if talk is not None:
            index[(row.rule, talk)].append(row)
    rng = random.Random(seed)
    out: list[Any] = []
    for start in range(0, len(rows), batch_size):
        batch = list(rows[start : start + batch_size])
        taken: set[int] = set()  # positions already a leader or a mate
        for leader_at, leader in enumerate(batch):
            talk = _talk(leader)
            if talk is None or leader_at in taken:
                continue
            taken.add(leader_at)
            speakers = {leader.meta.get("speaker")}
            # Later rows of the batch that carry a talk_id and are not yet grouped: the slots to swap.
            slots = [i for i in range(leader_at + 1, len(batch)) if i not in taken and _talk(batch[i]) is not None]
            filled = 1
            for slot in slots:
                if filled >= group:
                    break
                rule = batch[slot].rule
                mates = [r for r in index.get((rule, talk), []) if r.meta.get("speaker") not in speakers]
                if not mates:
                    continue
                mate = rng.choice(mates)
                batch[slot] = mate
                taken.add(slot)
                speakers.add(mate.meta.get("speaker"))
                filled += 1
        out.extend(batch)
    return out


def conversation_stats(rows: list[Any], batch_size: int) -> dict[str, float]:
    """Share of talk-carrying rows that have a same-conversation, other-speaker row in their micro-batch."""
    with_talk = grouped = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        speakers_by_talk: dict[int, set[Any]] = defaultdict(set)
        for row in batch:
            if _talk(row) is not None:
                speakers_by_talk[_talk(row)].add(row.meta.get("speaker"))
        for row in batch:
            if _talk(row) is None:
                continue
            with_talk += 1
            grouped += len(speakers_by_talk[_talk(row)] - {row.meta.get("speaker")}) > 0
    return {"rows_with_talk": float(with_talk), "grouped_share": grouped / with_talk if with_talk else 0.0}
