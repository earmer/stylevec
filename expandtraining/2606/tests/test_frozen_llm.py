from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from frozen_llm import content_residual, persona_coordinates, reference_lines, validation_rows
from records import TripletRecord


def triplet(anchor: str, speaker: str, split: str = "train", heldout: bool = False) -> TripletRecord:
    return TripletRecord(anchor, anchor + "。", "别人的话" + anchor, "same_speaker_real", 1.0, 0.2, split,
                         {"speaker": speaker, "negative_speaker": "乙", "heldout": heldout, "anchor_db_id": 7})


def test_persona_coordinates_point_each_speaker_at_its_own_voice():
    rng = np.random.default_rng(0)
    voices = rng.normal(size=(3, 32))
    reference = {f"s{k}": voices[k] + 0.1 * rng.normal(size=(20, 32)) for k in range(3)}
    rows = voices + 0.1 * rng.normal(size=(3, 32))
    coords = persona_coordinates(rows, reference)
    assert coords.shape == (3, 3)
    assert (coords.argmax(1) == np.arange(3)).all()


def test_validation_rows_carry_full_text_and_reference_lines_skip_heldout():
    triplets = [triplet(f"第{i}句话，说得很长很长" * 20, "甲", split="val") for i in range(3)]
    triplets += [triplet(f"训练台词{i}", "甲") for i in range(5)] + [triplet("留出的台词", "丙", heldout=True)]
    rows, texts = validation_rows(triplets, per_rule=8)
    assert len(rows) == len(texts) == 9
    assert list(rows["role"][:3]) == ["anchor", "positive", "negative"]
    assert len(texts[0]) > 120 and rows["text_preview"][0] == texts[0][:120]
    assert rows["db_id"][0] == 7 and rows["speaker"][2] == "乙"
    refs = reference_lines(triplets, speakers=4, lines=3)
    assert list(refs) == ["甲"] and len(refs["甲"]) == 3


def test_content_residual_removes_what_the_rewrite_predicts():
    rng = np.random.default_rng(0)
    flat = rng.normal(size=(400, 16)).astype(np.float32)
    style = rng.normal(size=(400, 16)).astype(np.float32)
    plain = flat @ rng.normal(size=(16, 16)).astype(np.float32) + style
    resid = content_residual(plain, flat, np.arange(400), ridge=1e-6)
    assert np.corrcoef(resid.ravel(), style.ravel())[0, 1] > 0.95


def test_aggregate_top1_pools_noisy_lines():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from speaker_aggregate import aggregate_top1
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(10, 32))
    lines = {f"s{i}": np.arange(i * 40, i * 40 + 40) for i in range(10)}
    embeddings = np.repeat(centres, 40, axis=0) + 4.0 * rng.normal(size=(400, 32))
    one, _ = aggregate_top1(embeddings, lines, 1)
    many, used = aggregate_top1(embeddings, lines, 16)
    assert used == 10 and many > 0.9 > one
