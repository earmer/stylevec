from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import pandas as pd

from config import resolve_repo_path
from records import SampleRecord, normalize_text, stable_id


FEATURE_STOPWORDS = {"usage", "use", "of", "or", "and", "the", "a", "in", "words", "word", "frequent", "indicating"}


def feature_terms(feature: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[a-z]+", feature.lower()):
        if token in FEATURE_STOPWORDS:
            continue
        if token.endswith("es") and len(token) > 4 and token[-3] == "s":
            token = token[:-2]  # processes -> process
        elif token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        terms.add(token)
    return terms


def near_duplicate_feature(feature: str, kept: list[str], threshold: float = 0.5) -> bool:
    """The source lists some features twice under near-identical names ("Affective Processes" /
    "Affective Process", "Present Tense and Present-Focused Words" / "Present-Focused Tense and Words").
    Kept both, one served as the other's cross-feature negative. Word-set Jaccard after stopwords."""
    terms = feature_terms(feature)
    for other in kept:
        other_terms = feature_terms(other)
        union = terms | other_terms
        if union and len(terms & other_terms) / len(union) >= threshold:
            return True
    return False


MSYNTHSTEL_MODES = ("triplets", "pairs")


def msynthstel_mode(config: dict[str, Any]) -> str:
    """triplets: the trait rules as built so far. pairs: trait_positive_vs_negative only, each labelled with
    its pole pair (meta `pole`) for trait directions; trait_cross_feature_hard is dropped, as its negative
    often shares the anchor's feature (prepositions vs determiners) and never its content."""
    mode = str(config.get("data", {}).get("msynthstel_mode", "triplets"))
    if mode not in MSYNTHSTEL_MODES:
        raise ValueError(f"data.msynthstel_mode must be one of {MSYNTHSTEL_MODES}, got {mode!r}")
    return mode


def load_msynthstel_samples(config: dict[str, Any]) -> tuple[list[SampleRecord], dict[str, int]]:
    train_path = resolve_repo_path(str(config["paths"]["msynthstel_train"]))
    test_path = resolve_repo_path(str(config["paths"]["msynthstel_test"]))
    feature_cap = int(config["data"]["msynthstel_features"])
    rows_per_feature = int(config["data"].get("msynthstel_rows_per_feature", 4))
    # pairs keeps near-duplicate feature names: each is its own pole pair ("Affective / Perceptual" vs
    # "Affective / Cognitive"), which trait directions tell apart by `raw_feature`.
    keep_near_duplicates = msynthstel_mode(config) == "pairs"
    samples: list[SampleRecord] = []
    filter_counts: defaultdict[str, int] = defaultdict(int)

    for split, path in (("train", train_path), ("val", test_path)):
        frame = pd.read_parquet(path)
        required = {"positive", "negative", "feature", "feature_clean"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"msynthstel parquet missing columns {sorted(missing)}: {path}")
        feature_counts: defaultdict[str, int] = defaultdict(int)
        features_seen: list[str] = []
        skipped_features: set[str] = set()
        for row_index, row in frame.iterrows():
            feature = normalize_text(str(row["feature_clean"]))
            raw_feature = normalize_text(str(row["feature"]))
            positive = normalize_text(str(row["positive"]))
            negative = normalize_text(str(row["negative"]))
            if not feature or not raw_feature or not positive or not negative:
                filter_counts["empty_field"] += 1
                continue
            if feature in skipped_features:
                filter_counts["near_duplicate_feature"] += 1
                continue
            if feature not in features_seen:
                if feature_cap > 0 and len(features_seen) >= feature_cap:
                    filter_counts["feature_cap"] += 1
                    continue
                if not keep_near_duplicates and near_duplicate_feature(feature, features_seen):
                    skipped_features.add(feature)
                    filter_counts["near_duplicate_feature"] += 1
                    continue
                features_seen.append(feature)
            # pairs caps each pole pair: one feature name can hold several ("Self-focused / You-focused", ...).
            count_key = raw_feature if keep_near_duplicates else feature
            if rows_per_feature > 0 and feature_counts[count_key] >= rows_per_feature:
                filter_counts["rows_per_feature_cap"] += 1
                continue
            for polarity, text in (("positive", positive), ("negative", negative)):
                samples.append(
                    SampleRecord(
                        sample_id=stable_id("msynthstel", split, row_index, polarity),
                        text=text,
                        style_id=feature,
                        style_type="trait",
                        domain="msynthstel",
                        content_id=stable_id("msynthstel_content", split, row_index),
                        source_id=f"{path.name}:{row_index}:{polarity}",
                        language="en",
                        split=split,
                        quality=0.7,
                        meta={
                            "feature": feature,
                            "raw_feature": raw_feature,
                            "polarity": polarity,
                            "length": len(text),
                            "length_bucket": len(text) // 40,
                        },
                    )
                )
            feature_counts[count_key] += 1
    if not samples:
        raise ValueError("msynthstel importer produced no samples")
    return samples, dict(filter_counts)
