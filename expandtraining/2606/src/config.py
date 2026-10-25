from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from leak import validate_leak_config
from trait_directions import validate_trait_direction_config


LOSS_TYPES = ("triplet", "infonce")


def workspace_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else repo_root() / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    data["_workspace_dir"] = str(workspace_dir())
    data["_repo_root"] = str(repo_root())
    validate_config(data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    for section in ("runtime", "paths", "data", "model", "train", "validation"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing config section: {section}")
    required_paths = (
        "genshin_db",
        "top_speakers",
        "paratarget_jsonl",
        "msynthstel_train",
        "msynthstel_test",
        "base_model",
    )
    for key in required_paths:
        if key not in config["paths"]:
            raise ValueError(f"missing paths.{key}")
        path = resolve_repo_path(str(config["paths"][key]))
        if not path.exists():
            raise FileNotFoundError(f"configured path does not exist: paths.{key}={path}")
    if bool(config["data"].get("bookraw_enabled", False)):
        if "bookraw_cleaned" not in config["paths"]:
            raise ValueError("missing paths.bookraw_cleaned")
        path = resolve_repo_path(str(config["paths"]["bookraw_cleaned"]))
        if not path.exists():
            raise FileNotFoundError(f"configured path does not exist: paths.bookraw_cleaned={path}")
        for key in (
            "bookraw_authors",
            "bookraw_segments_per_author",
            "bookraw_segments_per_work",
        ):
            if int(config["data"][key]) < 0:
                raise ValueError(f"data.{key} must be non-negative")
        for key in (
            "bookraw_min_segments_per_author",
            "bookraw_min_segments_per_work",
        ):
            if int(config["data"][key]) <= 0:
                raise ValueError(f"data.{key} must be positive")
    if str(config["model"]["base_model_path"]) != str(config["paths"]["base_model"]):
        raise ValueError("model.base_model_path must match paths.base_model")
    if int(config["train"]["batch_size"]) <= 0:
        raise ValueError("train.batch_size must be positive")
    if int(config["train"]["max_steps"]) <= 0:
        raise ValueError("train.max_steps must be positive")
    if int(config["data"]["max_triplets"]) <= 0:
        raise ValueError("data.max_triplets must be positive")
    validate_loss_config(config)


def validate_loss_config(config: dict[str, Any]) -> None:
    loss_type = str(config["train"].get("loss", "triplet"))
    if loss_type not in LOSS_TYPES:
        raise ValueError(f"train.loss must be one of {LOSS_TYPES}, got {loss_type!r}")
    if float(config["train"].get("infonce_temperature", 0.05)) <= 0.0:
        raise ValueError("train.infonce_temperature must be positive")
    if float(config["train"].get("infonce_triplet_weight", 0.0)) < 0.0:
        raise ValueError("train.infonce_triplet_weight must be non-negative")
    if float(config["train"].get("infonce_twin_temperature", 0.0)) < 0.0:
        raise ValueError("train.infonce_twin_temperature must be non-negative (0 = off)")
    max_steps = int(config["train"].get("max_steps", 0))
    for key in ("infonce_until_step", "stop_at_step"):
        value = int(config["train"].get(key, 0))
        if value < 0 or (value > 0 and value > max_steps):
            raise ValueError(f"train.{key} must be in [0, train.max_steps] (0 = off)")
    if int(config["model"].get("content_dim", 0)) < 0:
        raise ValueError("model.content_dim must be non-negative")
    regularization = config.get("regularization", {})
    if float(regularization.get("content_temperature", 0.1)) <= 0.0:
        raise ValueError("regularization.content_temperature must be positive")
    for key in ("content_weight", "content_leak_weight", "content_orthogonal_weight"):
        if float(regularization.get(key, 0.0)) < 0.0:
            raise ValueError(f"regularization.{key} must be non-negative")
    validate_leak_config(config)
    validate_trait_direction_config(config)


def config_hash(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def config_snapshot(config: dict[str, Any]) -> str:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    return yaml.safe_dump(clean, allow_unicode=True, sort_keys=True)


def configured_matryoshka_dims(config: dict[str, Any]) -> list[int]:
    """Prefixes the training loss covers (`regularization.matryoshka_dims`); the full dimension is implied."""
    projection_dim = int(config["model"]["projection_dim"])
    configured = config.get("regularization", {}).get("matryoshka_dims", [])
    return sorted({int(value) for value in configured if 0 < int(value) < projection_dim})


def validation_matryoshka_dims(config: dict[str, Any]) -> list[int]:
    """Prefixes validation and the probes report: `validation.matryoshka_dims`, else the training ones.

    Separate so an arm can train fewer prefixes (e.g. [64]) and still be scored at 32/64/128 like the rest.
    """
    configured = config.get("validation", {}).get("matryoshka_dims")
    if configured is None:
        return configured_matryoshka_dims(config)
    projection_dim = int(config["model"]["projection_dim"])
    return sorted({int(value) for value in configured if 0 < int(value) < projection_dim})

