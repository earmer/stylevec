from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from config import resolve_repo_path
from records import SampleRecord, normalize_text, stable_id


def load_msynthstel_samples(config: dict[str, Any]) -> tuple[list[SampleRecord], dict[str, int]]:
    train_path = resolve_repo_path(str(config["paths"]["msynthstel_train"]))
    test_path = resolve_repo_path(str(config["paths"]["msynthstel_test"]))
    feature_cap = int(config["data"]["msynthstel_features"])
    rows_per_feature = int(config["data"].get("msynthstel_rows_per_feature", 4))
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
        for row_index, row in frame.iterrows():
            feature = normalize_text(str(row["feature_clean"]))
            raw_feature = normalize_text(str(row["feature"]))
            positive = normalize_text(str(row["positive"]))
            negative = normalize_text(str(row["negative"]))
            if not feature or not raw_feature or not positive or not negative:
                filter_counts["empty_field"] += 1
                continue
            if feature not in features_seen:
                if feature_cap > 0 and len(features_seen) >= feature_cap:
                    filter_counts["feature_cap"] += 1
                    continue
                features_seen.append(feature)
            if rows_per_feature > 0 and feature_counts[feature] >= rows_per_feature:
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
                        },
                    )
                )
            feature_counts[feature] += 1
    if not samples:
        raise ValueError("msynthstel importer produced no samples")
    return samples, dict(filter_counts)
