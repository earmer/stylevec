"""Triplet construction for multilingual StyleDistance training.

Weighted curriculum triplet types:

  Type A — Traditional (within-language style discrimination):
    a, p, n all in same language X, same feature F.
    Teaches style discrimination within a language.

  Type B — Cross-lingual (style discrimination across languages):
    a, p in language X, n in language Y (Y != X), same feature F.
    Teaches that style features transcend language boundaries.
    Requires feature F exists in >= 2 languages.

  Type C — Language-as-feature (language identity as a style vector):
    a, p in language X, n in language Y (Y != X), any feature.
    Teaches language identity as a weak separable style dimension.

  Type D/E — Hard trait structure:
    cross-language same-trait pull and same-language different-trait negatives.
    Teaches trait independence inside language groups.
"""

from __future__ import annotations

import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def make_triplet(
    anchor: str,
    positive: str,
    negative: str,
    kind: str,
    weight: float,
    margin: float,
) -> dict[str, str | float]:
    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negative,
        "kind": kind,
        "weight": float(weight),
        "margin": float(margin),
    }


def _ratio_label(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


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
        return make_triplet(a, p, n, "monolingual", 1.0, 0.1)


class MultilingualTripletDataset(Dataset):
    """Language-aware triplet dataset with weighted curriculum triplet types."""

    def __init__(
        self,
        pairs: list[dict],
        tokenizer: AutoTokenizer,
        max_len: int = 128,
        split: str = "train",
        cache_dir: Path | None = None,
        max_pairs_per_feature: int = 0,
        seed: int | None = None,
        type_c_ratio: float = 0.2,
        cross_lang_trait_ratio: float = 0.25,
        same_lang_hard_ratio: float = 0.25,
    ):
        self.max_len = max_len
        self.split = split
        self.pairs = pairs
        self.max_pairs_per_feature = max_pairs_per_feature
        self.seed = seed if seed is not None else (42 if split != "train" else None)
        self.type_c_ratio = type_c_ratio
        self.cross_lang_trait_ratio = cross_lang_trait_ratio
        self.same_lang_hard_ratio = same_lang_hard_ratio
        ratios = (
            _ratio_label(type_c_ratio),
            _ratio_label(cross_lang_trait_ratio),
            _ratio_label(same_lang_hard_ratio),
        )
        cache_name = f"triplets_multilingual_weighted_{split}_{'_'.join(ratios)}.pkl"
        self._cache_path = cache_dir / cache_name if cache_dir else None
        self.triplet_counts: Counter[str] = Counter()

        # Only cache when no per-epoch resampling (mpp == 0)
        if max_pairs_per_feature == 0 and self._cache_path and self._cache_path.exists():
            with open(self._cache_path, "rb") as f:
                self.triplets = pickle.load(f)
            self._print_triplet_counts("Loaded")
            print(f"Loaded {len(self.triplets)} cached triplets from {self._cache_path}")
            return

        self._build_triplets()

        if max_pairs_per_feature == 0 and self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "wb") as f:
                pickle.dump(self.triplets, f)
            print(f"Cached {len(self.triplets)} triplets to {self._cache_path}")

    def resample(self):
        """Rebuild all triplets with fresh randomness (per-epoch rotation)."""
        self._build_triplets()

    def _add_triplet(
        self,
        anchor: str,
        positive: str,
        negative: str,
        kind: str,
        weight: float,
        margin: float,
    ):
        self.triplets.append(make_triplet(anchor, positive, negative, kind, weight, margin))
        self.triplet_counts[kind] += 1

    def _print_triplet_counts(self, prefix: str = "Built"):
        counts = Counter(item.get("kind", "legacy") for item in self.triplets)
        self.triplet_counts = counts
        print(f"  {prefix} multilingual triplets:")
        for kind in sorted(counts):
            weight = next(
                (item.get("weight", 1.0) for item in self.triplets if item.get("kind") == kind),
                1.0,
            )
            print(f"    {kind:<32} {counts[kind]:>8}  effective={counts[kind] * float(weight):>10.1f}")

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

        lang_feature_groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for (feat, lang), group in feature_lang_groups.items():
            if group:
                lang_feature_groups[lang][feat].extend(group)

        sampled_lang_groups: dict[str, list[dict]] = defaultdict(list)
        for (_feat, lang), group in feature_lang_groups.items():
            sampled_lang_groups[lang].extend(group)

        # --- Types A+B: style-feature triplets ---
        self.triplets = []
        self.triplet_counts = Counter()

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

                        if is_multilang and rng.random() < 0.1:
                            # Type B: cross-lingual negative
                            target_lang = rng.choice(other_langs)
                            target_group = feature_lang_groups[(feat, target_lang)]
                            k = rng.randrange(len(target_group))
                            n_text = target_group[k]["negative"]
                            self._add_triplet(
                                a, p, n_text,
                                "trait_cross_lang_negative",
                                0.8,
                                0.08,
                            )
                        else:
                            # Type A: traditional within-language negative
                            k_src = j if (i + j) % 2 == 0 else i
                            n_text = group[k_src]["negative"]
                            self._add_triplet(
                                a, p, n_text,
                                "trait_same_lang",
                                1.0,
                                0.10,
                            )

        total_ab = (
            self.triplet_counts["trait_same_lang"]
            + self.triplet_counts["trait_cross_lang_negative"]
        )

        # --- Type C: language-as-feature ---
        all_langs = sorted(lang_groups.keys())
        target_c = int(total_ab * self.type_c_ratio)
        if len(all_langs) >= 2 and target_c > 0:
            for _ in range(target_c):
                lang = rng.choice(all_langs)
                group = sampled_lang_groups[lang] or lang_groups[lang]
                if len(group) < 2:
                    continue
                other_langs = [l for l in all_langs if l != lang and (sampled_lang_groups[l] or lang_groups[l])]
                if not other_langs:
                    continue
                i = rng.randrange(len(group))
                j = rng.randrange(len(group) - 1)
                if j >= i:
                    j += 1
                target_lang = rng.choice(other_langs)
                target_group = sampled_lang_groups[target_lang] or lang_groups[target_lang]
                k = rng.randrange(len(target_group))
                self._add_triplet(
                    group[i]["positive"],
                    group[j]["positive"],
                    target_group[k]["positive"],
                    "language_feature",
                    0.2,
                    0.05,
                )

        # --- Cross-language same-trait pull with same-language hard negatives ---
        cross_candidates = []
        for feat, langs in feature_langs.items():
            if len(langs) < 2:
                continue
            for lang in sorted(langs):
                group = feature_lang_groups[(feat, lang)]
                if not group:
                    continue
                other_langs = [l for l in sorted(langs) if l != lang and feature_lang_groups[(feat, l)]]
                other_feats = [f for f in lang_feature_groups[lang] if f != feat and lang_feature_groups[lang][f]]
                if other_langs and other_feats:
                    cross_candidates.append((feat, lang, group, other_langs, other_feats))

        target_cross = int(total_ab * self.cross_lang_trait_ratio)
        for _ in range(target_cross):
            if not cross_candidates:
                break
            feat, lang, group, other_langs, other_feats = rng.choice(cross_candidates)
            anchor = rng.choice(group)
            pos_lang = rng.choice(other_langs)
            positive = rng.choice(feature_lang_groups[(feat, pos_lang)])
            neg_feat = rng.choice(other_feats)
            negative = rng.choice(lang_feature_groups[lang][neg_feat])
            self._add_triplet(
                anchor["positive"],
                positive["positive"],
                negative["positive"],
                "cross_lang_trait_pull",
                1.2,
                0.10,
            )

        # --- Same-language different-trait hard negatives ---
        hard_candidates = []
        for (feat, lang), group in feature_lang_groups.items():
            other_feats = [f for f in lang_feature_groups[lang] if f != feat and lang_feature_groups[lang][f]]
            if len(group) >= 2 and other_feats:
                hard_candidates.append((feat, lang, group, other_feats))

        target_hard = int(total_ab * self.same_lang_hard_ratio)
        for _ in range(target_hard):
            if not hard_candidates:
                break
            feat, lang, group, other_feats = rng.choice(hard_candidates)
            i = rng.randrange(len(group))
            j = rng.randrange(len(group) - 1)
            if j >= i:
                j += 1
            neg_feat = rng.choice(other_feats)
            negative = rng.choice(lang_feature_groups[lang][neg_feat])
            self._add_triplet(
                group[i]["positive"],
                group[j]["positive"],
                negative["positive"],
                "same_lang_trait_hard_negative",
                1.5,
                0.12,
            )

        rng.shuffle(self.triplets)
        self._print_triplet_counts()

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        if isinstance(triplet, dict):
            return triplet
        a, p, n = triplet
        return make_triplet(a, p, n, "legacy", 1.0, 0.1)


def collate_triplets(
    batch: list[dict[str, str]],
    tokenizer: AutoTokenizer,
    max_len: int = 128,
) -> dict[str, torch.Tensor]:
    """Tokenize a batch of triplets.  Returns padded (3*batch, max_len) tensors."""
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]
    weights = [float(item.get("weight", 1.0)) for item in batch]
    margins = [float(item.get("margin", 0.1)) for item in batch]
    kinds = [str(item.get("kind", "legacy")) for item in batch]

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
        "weights": torch.tensor(weights, dtype=torch.float32),
        "margins": torch.tensor(margins, dtype=torch.float32),
        "kinds": kinds,
    }
