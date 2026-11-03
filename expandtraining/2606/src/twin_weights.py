"""Per-pair weights for the content-twin repulsion (train.infonce_twin_temperature).

A paraphrase twin is pushed away from its origin line as "same content, another voice". When the paraphrase
barely changed the style (a near-copy), that push teaches the model to separate two lines of one voice by
their content. train.twin_weight_file (jsonl rows {"text", "twin_text", "weight"}, written by
scripts/twin_style_delta.py from judged trait scores) scales each pair's push by how much the judge says the
style changed; pairs it lacks get train.twin_weight_default.
"""
from __future__ import annotations

import json
from typing import Any

from config import resolve_repo_path
from records import normalize_text


def twin_weight_settings(config: dict[str, Any]) -> dict[str, Any]:
    train = config["train"]
    return {"file": str(train.get("twin_weight_file", "") or ""),
            "default": float(train.get("twin_weight_default", 1.0))}


def load_twin_weights(path: str) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for line in resolve_repo_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            weight = float(row["weight"])
            if not 0.0 <= weight <= 10.0:
                raise ValueError(f"twin weight must be in [0, 10], got {weight} in {path}")
            weights[(normalize_text(row["text"]), normalize_text(row["twin_text"]))] = weight
    return weights


def apply_twin_weights(config: dict[str, Any], triplets: list[Any]) -> dict[str, int]:
    """Sets meta twin_weight on every row with a twin_text; a no-op without train.twin_weight_file."""
    settings = twin_weight_settings(config)
    if not settings["file"]:
        return {}
    weights = load_twin_weights(settings["file"])
    counts = {"labelled": 0, "default": 0}
    for row in triplets:
        twin = row.meta.get("twin_text")
        if not twin:
            continue
        key = (normalize_text(row.anchor_text), normalize_text(str(twin)))
        if key in weights:
            row.meta["twin_weight"] = weights[key]
            counts["labelled"] += 1
        else:
            row.meta["twin_weight"] = settings["default"]
            counts["default"] += 1
    print(f"twin weights from {settings['file']}: {counts}", flush=True)
    return counts
