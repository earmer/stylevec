from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from build_triplets import build_training_triplets
from config import load_config


def test_dryrun_config_builds_real_triplets() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "dryrun.yaml"
    config = load_config(config_path)
    triplets, manifest = build_training_triplets(config)
    assert any(row.split == "train" for row in triplets)
    assert any(row.split == "val" for row in triplets)
    assert manifest["counts"]["triplets/all/train"] <= config["data"]["max_triplets"]
    assert manifest["counts"]["samples/msynthstel/train"] > 0
    assert manifest["counts"]["samples/bookraw/train"] > 0
    assert manifest["counts"]["triplets/bookraw_same_work/train"] > 0
    assert manifest["counts"]["triplets/bookraw_same_author_cross_work/train"] > 0
    assert manifest["counts"]["triplets/bookraw_same_language_author_negative/train"] > 0
