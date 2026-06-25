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
        return batch


def make_triplet_collate(tokenizer: Any, max_length: int) -> TripletCollate:
    return TripletCollate(tokenizer, max_length)
