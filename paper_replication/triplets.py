"""Triplet construction exactly as described in the StyleDistance paper (Section 4.1).

For each style feature with N pairs:
  - Anchor (a): positive example from pair i
  - Positive (p): positive example from pair j (j != i) — same style, different content
  - Negative (n): paraphrase of a (50%) or paraphrase of p (50%) — always different style

Yields N * (N-1) triplets per feature.  With 90 train pairs per feature × 40 features = ~320K.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TripletDataset(Dataset):
    def __init__(
        self,
        pairs: list[dict],
        tokenizer: AutoTokenizer,
        max_len: int = 128,
        split: str = "train",
        cache_dir: Path | None = None,
    ):
        self.max_len = max_len
        self.split = split

        cache_path = cache_dir / f"triplets_{split}.pkl" if cache_dir else None
        if cache_path and cache_path.exists():
            with open(cache_path, "rb") as f:
                self.triplets = pickle.load(f)
            print(f"Loaded {len(self.triplets)} cached triplets from {cache_path}")
            return

        # Group pairs by feature
        by_feature: dict[str, list[dict]] = {}
        for row in pairs:
            feature = row["feature"]
            by_feature.setdefault(feature, []).append(row)

        self.triplets = []
        for feature, group in by_feature.items():
            n = len(group)
            for i in range(n):
                a_text = group[i]["positive"]
                for j in range(n):
                    if j == i:
                        continue
                    p_text = group[j]["positive"]
                    # 50%: negative = paraphrase of anchor (same content, diff style)
                    # 50%: negative = paraphrase of positive (diff content, diff style)
                    k = j if (i + j) % 2 == 0 else i
                    n_text = group[k]["negative"]
                    self.triplets.append((a_text, p_text, n_text))

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(self.triplets, f)
            print(f"Cached {len(self.triplets)} triplets to {cache_path}")

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        a, p, n = self.triplets[idx]
        return {"anchor": a, "positive": p, "negative": n}


def collate_triplets(
    batch: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_len: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of triplets.  Returns padded (3*batch, max_len) tensors."""
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    all_texts = anchors + positives + negatives
    encoded = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    bs = len(batch)
    return {
        "a_ids": encoded["input_ids"][:bs],
        "a_mask": encoded["attention_mask"][:bs],
        "p_ids": encoded["input_ids"][bs : 2 * bs],
        "p_mask": encoded["attention_mask"][bs : 2 * bs],
        "n_ids": encoded["input_ids"][2 * bs :],
        "n_mask": encoded["attention_mask"][2 * bs :],
    }
