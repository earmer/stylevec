from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prebuilt
from build_triplets import build_training_triplets
from config import load_config
from prebuilt import data_key, load_dataset, load_or_build, write_dataset

DRYRUN = Path(__file__).resolve().parents[1] / "configs" / "dryrun.yaml"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    config = load_config(DRYRUN)
    triplets, manifest = build_training_triplets(config)
    out = tmp_path_factory.mktemp("datasets") / "d"
    write_dataset(config, out, triplets, manifest)
    return config, triplets, manifest, out


def test_loaded_dataset_equals_a_fresh_build(built) -> None:
    config, triplets, manifest, out = built
    loaded, loaded_manifest = load_or_build(config, str(out))
    assert loaded == triplets
    assert loaded_manifest["manifest_hash"] == manifest["manifest_hash"]


def test_non_data_settings_reuse_the_dataset_and_keep_resume_hashes(built) -> None:
    config, _, _, out = built
    other = {**config, "train": {**config["train"], "lr": 1.0}, "curriculum": {"enabled": True, "mode": "anti"}}
    assert data_key(other) == data_key(config)
    _, manifest = load_dataset(other, str(out))
    fresh_counts = json.loads((out / "manifest.json").read_text())["counts"]
    assert manifest["manifest_hash"] == prebuilt.manifest_hash(other, fresh_counts)


@pytest.mark.parametrize(
    "change",
    [
        lambda c: {**c, "data": {**c["data"], "max_triplets": c["data"]["max_triplets"] + 1}},
        lambda c: {**c, "weights": {**c["weights"], "genshin_real": 0.5}},
        lambda c: {**c, "validation": {**c["validation"], "max_triplets": 7}},
    ],
)
def test_data_settings_refuse_the_dataset(built, change) -> None:
    config, _, _, out = built
    with pytest.raises(ValueError, match="data key"):
        load_dataset(change(config), str(out))


def test_builder_code_change_refuses_the_dataset(built, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _, _, out = built
    monkeypatch.setattr(prebuilt, "builder_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="data key"):
        load_dataset(config, str(out))


def test_truncated_or_unfinished_dataset_is_refused(built, tmp_path: Path) -> None:
    config, triplets, manifest, _ = built
    out = tmp_path / "d"
    write_dataset(config, out, triplets, manifest)
    lines = (out / "triplets.jsonl").read_text().splitlines()
    (out / "triplets.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="counts differ"):
        load_dataset(config, str(out))
    (out / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_dataset(config, str(out))
