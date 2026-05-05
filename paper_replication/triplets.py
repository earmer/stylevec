"""Triplet construction for multilingual StyleDistance training.

Three balanced triplet types:

  Type A — Traditional (within-language style discrimination):
    a, p, n all in same language X, same feature F.
    Teaches style discrimination within a language.

  Type B — Cross-lingual (style discrimination across languages):
    a, p in language X, n in language Y (Y != X), same feature F.
    Teaches that style features transcend language boundaries.
    Requires feature F exists in >= 2 languages.

  Type C — Language-as-feature (language identity as a style vector):
    a, p in language X, n in language Y (Y != X), any feature.
    Teaches language identity as a separable style dimension.
"""

from __future__ import annotations

import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TripletDataset(Dataset):
    """Original monolingual triplet dataset (kept for reference / fallback)."""

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


class MultilingualTripletDataset(Dataset):
    """Language-aware triplet dataset with three balanced triplet types.

    Types A+B are interleaved per (feature, lang, i, j): 50% traditional
    negative (same language), 50% cross-lingual negative (different language)
    when the feature exists in >= 2 languages.  Type C (language-as-feature)
    generates a comparable number of triplets so all three types are balanced.
    """

    def __init__(
        self,
        pairs: list[dict],
        tokenizer: AutoTokenizer,
        max_len: int = 128,
        split: str = "train",
        cache_dir: Path | None = None,
        max_pairs_per_feature: int = 0,
        seed: int | None = None,
    ):
        self.max_len = max_len
        self.split = split
        self.pairs = pairs
        self.max_pairs_per_feature = max_pairs_per_feature
        self.seed = seed if seed is not None else (42 if split != "train" else None)
        self._cache_path = cache_dir / f"triplets_multilingual_{split}.pkl" if cache_dir else None

        # Only cache when no per-epoch resampling (mpp == 0)
        if max_pairs_per_feature == 0 and self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self.triplets = pickle.load(f)
            print(f"Loaded {len(self.triplets)} cached triplets from {self._cache_path}")
            return

        self._build_triplets()

        if max_pairs_per_feature == 0 and self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self.triplets, f)
            print(f"Cached {len(self.triplets)} triplets to {self._cache_path}")

    def resample(self):
        """Re-sample pairs and rebuild all triplets (per-epoch rotation)."""
        if self.max_pairs_per_feature <= 0:
            return  # no-op when using all pairs
        self._build_triplets()

    def _build_triplets(self):
        seed = time.time_ns() if self.seed is None else self.seed
        self._last_seed = seed
        rng = random.Random(seed)

        # --- Indexing ---
        feature_lang_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in self.pairs:
            key = (row["feature"], row["lang"])
            feature_lang_groups[key].append(row)

        feature_langs: dict[str, set[str]] = defaultdict(set)
        for (feat, lang), group in feature_lang_groups.items():
            feature_langs[feat].add(lang)

        lang_groups: dict[str, list[dict]] = defaultdict(list)
        for row in self.pairs:
            lang_groups[row["lang"]].append(row)

        # --- Downsample per (feature, lang) group if requested ---
        if self.max_pairs_per_feature > 0:
            for key in list(feature_lang_groups.keys()):
                group = feature_lang_groups[key]
                if len(group) > self.max_pairs_per_feature:
                    feature_lang_groups[key] = rng.sample(group, self.max_pairs_per_feature)

        # --- Types A+B: style-feature triplets ---
        self.triplets = []
        count_a = 0
        count_b = 0

        for feat, langs in feature_langs.items():
            is_multilang = len(langs) >= 2
            for lang in sorted(langs):
                group = feature_lang_groups[(feat, lang)]
                n = len(group)
                other_langs = sorted(langs - {lang})
                for i in range(n):
                    for j in range(n):
                        if j == i:
                            continue
                        a = group[i]["positive"]
                        p = group[j]["positive"]

                        if is_multilang and rng.random() < 0.5:
                            # Type B: cross-lingual negative
                            target_lang = rng.choice(other_langs)
                            target_group = feature_lang_groups[(feat, target_lang)]
                            k = rng.randrange(len(target_group))
                            n_text = target_group[k]["negative"]
                            count_b += 1
                        else:
                            # Type A: traditional within-language negative
                            k_src = j if (i + j) % 2 == 0 else i
                            n_text = group[k_src]["negative"]
                            count_a += 1

                        self.triplets.append((a, p, n_text))

        total_ab = count_a + count_b
        print(f"  Type A (traditional):         {count_a:>8}")
        print(f"  Type B (cross-lingual):       {count_b:>8}")

        # --- Type C: language-as-feature ---
        all_langs = sorted(lang_groups.keys())
        target_c = total_ab  # balance Type C with A+B total
        per_lang_target = max(1, target_c // len(all_langs))

        count_c = 0
        for lang, group in lang_groups.items():
            other_langs = [l for l in all_langs if l != lang]
            n_pairs = len(group)
            for _ in range(per_lang_target):
                i = rng.randrange(n_pairs)
                j = rng.randrange(n_pairs)
                while j == i:
                    j = rng.randrange(n_pairs)
                a = group[i]["positive"]
                p = group[j]["positive"]
                target_lang = rng.choice(other_langs)
                target_group = lang_groups[target_lang]
                k = rng.randrange(len(target_group))
                neg_text = target_group[k]["positive"]
                self.triplets.append((a, p, neg_text))
                count_c += 1

        print(f"  Type C (language-as-feature): {count_c:>8}")

        rng.shuffle(self.triplets)

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
