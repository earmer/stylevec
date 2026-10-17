"""Prebuilt triplet datasets, named by what produced them.

A dataset is outputs/datasets/<name>/{triplets.jsonl,manifest.json}. Its data key hashes the config
sections the builder reads plus the builder's own source, so a dataset is reused only for a config
and code that would rebuild it byte for byte. Default names are `d-<key[:12]>`, which lets every
arm with the same data share one directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import config_hash, workspace_dir
from records import TripletRecord, stable_id, write_jsonl

# Sections build_training_triplets reads. base_model is a path the builder never opens.
DATA_SECTIONS = ("paths", "data", "weights", "margins", "validation")
BUILDER_SOURCES = (
    "build_triplets.py",
    "import_genshin.py",
    "import_paratarget.py",
    "import_msynthstel.py",
    "import_bookraw.py",
    "records.py",
    "similarity.py",
)


def builder_fingerprint() -> str:
    src = Path(__file__).resolve().parent
    return stable_id(*[(name, (src / name).read_text(encoding="utf-8")) for name in BUILDER_SOURCES], size=16)


def data_key(config: dict[str, Any]) -> str:
    sections = {name: config.get(name, {}) for name in DATA_SECTIONS}
    sections["paths"] = {k: v for k, v in sections["paths"].items() if k != "base_model"}
    return stable_id(sections, builder_fingerprint(), size=24)


def default_name(config: dict[str, Any]) -> str:
    return f"d-{data_key(config)[:12]}"


def dataset_dir(dataset: str) -> Path:
    path = Path(dataset).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        return path
    return workspace_dir() / "outputs" / "datasets" / dataset


def manifest_hash(config: dict[str, Any], counts: dict[str, int]) -> str:
    # Same formula as build_training_triplets, so a run resumes across fresh and prebuilt data.
    return stable_id(config_hash(config), counts, size=24)


def write_dataset(config: dict[str, Any], out_dir: Path, triplets: list[TripletRecord], manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "data_key": data_key(config), "builder": builder_fingerprint(), "rows": len(triplets)}
    # manifest last: a dataset without manifest.json is incomplete and never loaded.
    (out_dir / "manifest.json").unlink(missing_ok=True)
    write_jsonl(str(out_dir / "triplets.jsonl"), [row.to_json_dict() for row in triplets])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(dataset: str) -> dict[str, Any]:
    path = dataset_dir(dataset) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset has no manifest.json (missing or incomplete): {path.parent}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(config: dict[str, Any], dataset: str) -> tuple[list[TripletRecord], dict[str, Any]]:
    """Load a prebuilt dataset, refusing one built from different data settings or builder code."""
    directory = dataset_dir(dataset)
    manifest = read_manifest(dataset)
    expected = data_key(config)
    if manifest.get("data_key") != expected:
        found = manifest.get("data_key", "none (built before data keys)")
        raise ValueError(
            f"dataset {directory} has data key {found}, this config needs {expected}; "
            f"rebuild with scripts/build_dataset.py --config <config>"
        )
    triplets = []
    with open(directory / "triplets.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stored_id = row.pop("triplet_id")
            triplet = TripletRecord(**row)
            if triplet.triplet_id != stored_id:
                raise ValueError(f"dataset {directory}: triplet_id mismatch at row {len(triplets)}")
            triplets.append(triplet)
    counts: dict[str, int] = {}
    for row in triplets:
        for key in (f"triplets/{row.rule}/{row.split}", f"triplets/all/{row.split}"):
            counts[key] = counts.get(key, 0) + 1
    stored = {k: v for k, v in manifest["counts"].items() if k.startswith("triplets/")}
    if counts != stored:
        raise ValueError(f"dataset {directory}: triplet counts differ from manifest.json")
    return triplets, {**manifest, "manifest_hash": manifest_hash(config, manifest["counts"]), "dataset": str(directory)}


def load_or_build(config: dict[str, Any], dataset: str | None) -> tuple[list[TripletRecord], dict[str, Any]]:
    if dataset:
        return load_dataset(config, dataset)
    from build_triplets import build_training_triplets

    return build_training_triplets(config)
