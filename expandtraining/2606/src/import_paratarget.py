from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from config import resolve_repo_path
from import_genshin import heldout_speakers
from records import SampleRecord, normalize_text, stable_id


def load_paratarget_samples(config: dict[str, Any], speakers: list[str]) -> tuple[list[SampleRecord], dict[str, int]]:
    path = resolve_repo_path(str(config["paths"]["paratarget_jsonl"]))
    speaker_set = set(speakers)
    heldout = heldout_speakers(config, speakers)
    max_jobs = int(config["data"]["paratarget_jobs"])
    per_target_cap = int(config["data"].get("paratarget_per_target_cap", max_jobs))
    min_ratio = float(config["data"].get("paratarget_min_length_ratio", 0.5))
    max_ratio = float(config["data"].get("paratarget_max_length_ratio", 1.8))
    samples: list[SampleRecord] = []
    filter_counts: defaultdict[str, int] = defaultdict(int)
    per_target: defaultdict[str, int] = defaultdict(int)
    accepted_jobs = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            row = json.loads(line)
            validation = row.get("validation")
            if not isinstance(validation, dict) or validation.get("valid") is not True:
                filter_counts["invalid_validation"] += 1
                continue
            source_speaker = row.get("source_speaker")
            target_speaker = row.get("target_speaker")
            if source_speaker not in speaker_set or target_speaker not in speaker_set:
                filter_counts["speaker_not_top"] += 1
                continue
            if source_speaker == target_speaker:
                filter_counts["same_speaker"] += 1
                continue
            if per_target_cap > 0 and per_target[target_speaker] >= per_target_cap:
                filter_counts["target_cap"] += 1
                continue
            source_lines = row.get("source_lines")
            generated_lines = row.get("generated_lines")
            if not isinstance(source_lines, list) or not isinstance(generated_lines, list):
                filter_counts["bad_line_fields"] += 1
                continue
            if len(source_lines) != len(generated_lines):
                filter_counts["line_count_mismatch"] += 1
                continue
            kept_for_job = 0
            job_id = str(row.get("job_id", f"line-{line_no}"))
            job_split = paratarget_job_split(job_id, str(source_speaker), str(target_speaker), heldout)
            for index, (source_raw, generated_raw) in enumerate(zip(source_lines, generated_lines)):
                source_text = normalize_text(str(source_raw))
                generated_text = normalize_text(str(generated_raw))
                if not source_text or not generated_text:
                    filter_counts["empty_line"] += 1
                    continue
                if source_text.casefold() == generated_text.casefold():
                    filter_counts["copied_line"] += 1
                    continue
                ratio = len(generated_text) / max(1, len(source_text))
                if ratio < min_ratio or ratio > max_ratio:
                    filter_counts["length_ratio"] += 1
                    continue
                split = job_split
                samples.append(
                    SampleRecord(
                        sample_id=stable_id("paratarget", job_id, index),
                        text=generated_text,
                        style_id=str(target_speaker),
                        style_type="generated_target",
                        domain="paratarget",
                        content_id=stable_id("paratarget_content", job_id, index),
                        source_id=f"{job_id}:{index}",
                        language="zh",
                        split=split,
                        quality=0.75,
                        meta={
                            "source_speaker": str(source_speaker),
                            "target_speaker": str(target_speaker),
                            "source_text": source_text,
                            "source_length": len(source_text),
                            "length": len(generated_text),
                            "length_bucket": len(generated_text) // 20,
                            "job_id": job_id,
                        },
                    )
                )
                kept_for_job += 1
                per_target[target_speaker] += 1
                if per_target_cap > 0 and per_target[target_speaker] >= per_target_cap:
                    break
            if kept_for_job:
                accepted_jobs += 1
            if max_jobs > 0 and accepted_jobs >= max_jobs:
                break
    if not samples:
        raise ValueError("paratarget importer produced no samples")
    return samples, dict(filter_counts)


def paratarget_job_split(job_id: str, source: str, target: str, heldout: set[str]) -> str:
    """Whole job to one split; jobs touching a held-out speaker are always val."""
    if source in heldout or target in heldout:
        return "val"
    return "val" if stable_id(job_id, size=2) < "20" else "train"
