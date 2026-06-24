from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    return _SPACE_RE.sub(" ", text).strip()


def stable_id(*parts: Any, size: int = 16) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:size]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    text: str
    style_id: str
    style_type: str
    domain: str
    content_id: str
    source_id: str
    language: str
    split: str
    quality: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not normalize_text(self.text):
            raise ValueError("SampleRecord.text must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {self.split}")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError(f"quality must be in [0, 1], got {self.quality}")

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["text"] = normalize_text(data["text"])
        return data


@dataclass(frozen=True)
class TripletRecord:
    anchor_text: str
    positive_text: str
    negative_text: str
    rule: str
    weight: float
    margin: float
    split: str = "train"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("anchor_text", "positive_text", "negative_text"):
            value = getattr(self, field_name)
            if not normalize_text(value):
                raise ValueError(f"TripletRecord.{field_name} must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"invalid split: {self.split}")
        if float(self.weight) <= 0.0:
            raise ValueError(f"weight must be positive, got {self.weight}")
        if float(self.margin) <= 0.0:
            raise ValueError(f"margin must be positive, got {self.margin}")

    @property
    def triplet_id(self) -> str:
        return stable_id(
            self.anchor_text,
            self.positive_text,
            self.negative_text,
            self.rule,
            self.split,
        )

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["triplet_id"] = self.triplet_id
        data["anchor_text"] = normalize_text(data["anchor_text"])
        data["positive_text"] = normalize_text(data["positive_text"])
        data["negative_text"] = normalize_text(data["negative_text"])
        return data


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty jsonl: {path}")
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


