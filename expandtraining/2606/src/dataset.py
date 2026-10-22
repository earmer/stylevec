from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from records import TripletRecord, normalize_text


class TripletTextDataset(Dataset[TripletRecord]):
    def __init__(self, triplets: list[TripletRecord], split: str) -> None:
        self.split = split
        self.triplets = [row for row in triplets if row.split == split]
        if not self.triplets:
            raise ValueError(f"no triplets for split={split}")

    def set_triplets(self, triplets: list[TripletRecord]) -> None:
        rows = [row for row in triplets if row.split == self.split]
        if not rows:
            raise ValueError(f"no triplets for split={self.split}")
        self.triplets = rows

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, index: int) -> TripletRecord:
        return self.triplets[index]


class TripletCollate:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            [normalize_text(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, rows: list[TripletRecord]) -> dict[str, Any]:
        if not rows:
            raise ValueError("empty batch")
        batch = {
            "anchor": self.encode([row.anchor_text for row in rows]),
            "positive": self.encode([row.positive_text for row in rows]),
            "negative": self.encode([row.negative_text for row in rows]),
            "margin": torch.tensor([row.margin for row in rows], dtype=torch.float32),
            "weight": torch.tensor([row.weight for row in rows], dtype=torch.float32),
            "triplet_id": [row.triplet_id for row in rows],
            "anchor_text": [normalize_text(row.anchor_text) for row in rows],
            "positive_text": [normalize_text(row.positive_text) for row in rows],
            "negative_text": [normalize_text(row.negative_text) for row in rows],
            "rule": [row.rule for row in rows],
            "meta": [row.meta for row in rows],
        }
        styles = [triplet_styles(row.rule, row.meta) for row in rows]
        batch["anchor_style"] = [anchor for anchor, _negative in styles]
        batch["negative_style"] = [negative for _anchor, negative in styles]
        return batch


def triplet_styles(rule: str, meta: dict[str, Any]) -> tuple[str | None, str | None]:
    """(anchor/positive style, negative style) as `domain:id`, for masking in-batch false negatives.

    The positive always shares the anchor's style. None is unknown and never masked: the neutral
    paraphrase (origin_vs_para), a feature's negative-polarity text, a missing meta key, or a rule not
    listed here.
    """
    if rule in ("same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"):
        return _style("genshin", meta.get("speaker")), _style("genshin", meta.get("negative_speaker"))
    if rule in ("generated_to_target", "paratarget_original_push"):
        return _style("genshin", meta.get("target_speaker")), _style("genshin", meta.get("source_speaker"))
    if rule == "target_hard_negative":
        return _style("genshin", meta.get("target_speaker")), _style("genshin", meta.get("negative_target"))
    if rule == "same_content_diff_style":
        # Anchor: the source line; negative: its rewrite in the target's style.
        return _style("genshin", meta.get("source_speaker")), _style("genshin", meta.get("target_speaker"))
    if rule == "trait_positive_vs_negative":
        return _style("msynthstel", meta.get("feature")), None
    if rule == "trait_cross_feature_hard":
        return _style("msynthstel", meta.get("feature")), _style("msynthstel", meta.get("negative_feature"))
    if rule.startswith("bookraw_"):
        return _style("bookraw", meta.get("author")), _style("bookraw", meta.get("negative_author"))
    if rule.startswith("length_course_"):
        # Genshin sources are speaker names, so these labels match the Genshin rules above.
        domain = meta.get("domain")
        if domain is None:
            return None, None
        return _style(domain, meta.get("source")), _style(domain, meta.get("negative_source"))
    return None, None


def _style(domain: str, value: Any) -> str | None:
    return None if value is None else f"{domain}:{value}"


def make_triplet_collate(tokenizer: Any, max_length: int) -> TripletCollate:
    return TripletCollate(tokenizer, max_length)
