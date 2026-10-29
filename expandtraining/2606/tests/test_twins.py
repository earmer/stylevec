from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_triplets import build_genshin_triplets, combine_same_source_texts, twin_meta
from config import validate_loss_config
from dataset import TripletCollate, twin_columns
from losses import info_nce_loss
from model import encode_triplet_outputs
from records import TripletRecord
from report import paraphrase_rank, within_speaker_nullspace
from test_content_head import encoder, token_batch
from test_curriculum_and_validation import genshin
from train import objective_at_step, objective_loss, objective_settings_from_config


def unit(*rows: list[float]) -> torch.Tensor:
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def nce(twin=None, has_twin=None, twin_temperature=0.0):
    anchor = unit([1, 0, 0], [0, 1, 0])
    positive = unit([0.9, 0.1, 0], [0.1, 0.9, 0])
    negative = unit([0, 0, 1], [0, 0.2, 1])
    return info_nce_loss(
        anchor, positive, negative, torch.full((2,), 0.1), torch.ones(2), torch.tensor([0, 1]),
        torch.tensor([-1, -2]), 0.05, twin=twin, has_twin=has_twin, twin_temperature=twin_temperature,
    )


def test_twin_adds_one_candidate_at_its_own_temperature_and_masked_rows_are_unchanged() -> None:
    base, _ = nce()
    twin = unit([0.95, 0.05, 0], [0, 0, 1])
    off, _ = nce(twin, torch.tensor([False, False]), 0.03)
    assert torch.allclose(off, base)
    _, stats = nce(twin, torch.tensor([True, False]), 0.03)
    _, base_stats = nce()
    assert torch.allclose(stats["raw_loss"][1], base_stats["raw_loss"][1])
    anchor0 = unit([1, 0, 0])[0]
    cands = torch.cat([unit([0.9, 0.1, 0], [0.1, 0.9, 0]), unit([0, 1, 0]), unit([0, 0, 1], [0, 0.2, 1])])
    logits = torch.cat([cands @ anchor0 / 0.05, (twin[0] @ anchor0 / 0.03)[None]])
    assert torch.allclose(stats["raw_loss"][0], -torch.log_softmax(logits, dim=0)[0], atol=1e-5)
    assert stats["twin_violation"] == 1.0 and stats["twin_fraction"] == 0.5
    with pytest.raises(ValueError):
        nce(twin, torch.tensor([True]), 0.03)


def row(rule: str, negative: str, twin: str | None) -> TripletRecord:
    return TripletRecord("锚点", "正例", negative, rule, 1.0, 0.1, meta={"twin_text": twin} if twin else {})


def test_twin_columns_mask_missing_twins_and_twins_that_are_the_negative() -> None:
    rows = [row("same_speaker_real", "别人", "改写"), row("origin_vs_para", "改写", "改写"), row("trait_x", "别人", None)]
    out = twin_columns(rows, lambda texts: texts)
    assert out["twin"] == ["改写", "锚点", "锚点"]
    assert out["has_twin"].tolist() == [True, False, False]
    assert twin_meta("一样", " 一样 ") == {} and twin_meta("原句", None) == {}


def test_genshin_triplets_carry_the_paraphrase_as_twin() -> None:
    samples = [genshin(s, order, f"{s}说的第{order}句话") for s in "AB" for order in (0, 200, 400)]
    config = {"weights": {"genshin_real": 1.0}, "margins": {"medium": 0.12, "hard": 0.18},
              "data": {"genshin_min_id_gap": 150}}
    triplets = build_genshin_triplets(config, samples, "train")
    assert triplets and all(t.meta["twin_text"] == "neutral " + t.anchor_text for t in triplets)


def test_combine_same_source_texts_can_leave_the_anchor_out() -> None:
    rows = [genshin("A", i, text) for i, text in enumerate(["短", "第二句话", "第三句话长一些"])]
    assert combine_same_source_texts(rows, 0, 2, 100).startswith("短")
    skipped = combine_same_source_texts(rows, 0, 2, 100, skip_anchor=True)
    assert skipped == "第二句话 第三句话长一些"
    assert combine_same_source_texts(rows[:2], 0, 2, 100, skip_anchor=True) is None


def test_encode_outputs_return_twin_embeddings_separately() -> None:
    model = encoder(0)
    batch = {**token_batch(), "twin": token_batch()["positive"]}
    for combined in (False, True):
        out = encode_triplet_outputs(model, batch, combine_forward=combined)
        assert len(out["embeddings"]) == 3 and out["pre_norm"].shape == (6,)
        assert torch.allclose(out["twin"], out["embeddings"][1], atol=1e-5)


def train_config(**train) -> dict:
    return {"train": {"loss": "infonce", "max_steps": 3000, **train}, "model": {}}


def test_infonce_until_step_switches_to_the_hinge_and_keys_are_checked() -> None:
    settings = objective_settings_from_config(train_config(infonce_until_step=1050, infonce_twin_temperature=0.03))
    assert objective_at_step(settings, 1049)["loss"] == "infonce"
    assert objective_at_step(settings, 1050)["loss"] == "triplet"
    always = objective_settings_from_config(train_config())
    assert objective_at_step(always, 2999)["loss"] == "infonce"
    validate_loss_config(train_config(stop_at_step=1050, infonce_until_step=1050))
    for bad in ({"stop_at_step": 3001}, {"infonce_until_step": -1}, {"infonce_twin_temperature": -0.1}):
        with pytest.raises(ValueError):
            validate_loss_config(train_config(**bad))


def test_infonce_from_step_and_sqrt_ramp() -> None:
    late = objective_settings_from_config(train_config(infonce_from_step=500))
    assert objective_at_step(late, 499)["loss"] == "triplet"
    assert objective_at_step(late, 500)["loss"] == "infonce" and "infonce_weight" not in objective_at_step(late, 500)
    ramp = objective_settings_from_config(train_config(stop_at_step=2100, infonce_weight_schedule="sqrt"))
    weights = [objective_at_step(ramp, step)["infonce_weight"] for step in (0, 525, 1050, 2100, 2500)]
    assert weights == pytest.approx([0.0, 0.5, 0.5**0.5, 1.0, 1.0])
    # The weight scales the InfoNCE term only; the hinge on top keeps its weight.
    a, p, n = unit([1.0, 0.0], [0.8, 0.6]), unit([0.6, 0.8], [0.0, 1.0]), unit([0.9, 0.1], [0.5, 0.5])
    batch = {"margin": torch.tensor([0.2, 0.2]), "weight": torch.ones(2), "anchor_style": ["A", "B"],
             "negative_style": ["C", "D"]}
    settings = {**objective_settings_from_config(train_config(infonce_triplet_weight=2.0))}
    full, stats = objective_loss(a, p, n, batch, settings)
    half, _ = objective_loss(a, p, n, batch, {**settings, "infonce_weight": 0.5})
    assert half.item() == pytest.approx(full.item() - 0.5 * stats["infonce_loss"].item())
    for bad in ({"infonce_weight_schedule": "cosine"}, {"infonce_from_step": 1050, "infonce_until_step": 1050},
                {"infonce_from_step": 3001}):
        with pytest.raises(ValueError):
            validate_loss_config(train_config(**bad))


def test_infonce_windows_run_infonce_in_bursts() -> None:
    bursts = objective_settings_from_config(train_config(infonce_windows=[[0, 1050], [1500, 1600]]))
    losses = [objective_at_step(bursts, step)["loss"] for step in (0, 1049, 1050, 1499, 1500, 1599, 1600)]
    assert losses == ["infonce", "infonce", "triplet", "triplet", "infonce", "infonce", "triplet"]
    validate_loss_config(train_config(infonce_windows=[[0, 100], [100, 200]]))
    for bad in ({"infonce_windows": [[500, 600], [100, 200]]}, {"infonce_windows": [[0, 3001]]},
                {"infonce_windows": [[200, 200]]}, {"infonce_windows": [[0, 100]], "infonce_from_step": 50}):
        with pytest.raises(ValueError):
            validate_loss_config(train_config(**bad))


def test_infonce_tiers_limit_the_queries_to_clean_rows() -> None:
    a, p, n = unit([1.0, 0.0], [0.8, 0.6]), unit([0.6, 0.8], [0.0, 1.0]), unit([0.9, 0.1], [0.5, 0.5])
    batch = {"margin": torch.tensor([0.2, 0.2]), "weight": torch.ones(2), "anchor_style": ["A", "B"],
             "negative_style": ["C", "D"], "rule": ["origin_vs_para", "same_speaker_real"]}
    every = objective_settings_from_config(train_config())
    clean = objective_settings_from_config(train_config(infonce_tiers=["clean"]))
    _all, stats = objective_loss(a, p, n, batch, every)
    only, clean_stats = objective_loss(a, p, n, batch, clean)
    # Row 0 (clean) is the only query; row 1 (confounded) stays a candidate, so row 0's loss is unchanged.
    assert only.item() == pytest.approx(stats["raw_loss"][0].item())
    assert clean_stats["query_fraction"].item() == pytest.approx(0.5) and "query_fraction" not in stats
    none, _ = objective_loss(a, p, n, {**batch, "rule": ["same_speaker_real"] * 2}, clean)
    assert none.item() == 0.0
    # The hinge on top still sees every row.
    hinged, _ = objective_loss(a, p, n, batch, {**clean, "triplet_weight": 1.0})
    from losses import weighted_triplet_margin_loss

    triplet, _ = weighted_triplet_margin_loss(a, p, n, batch["margin"], batch["weight"])
    assert hinged.item() == pytest.approx(only.item() + triplet.item())
    with pytest.raises(ValueError):
        validate_loss_config(train_config(infonce_tiers=["easy"]))


def probe_rows(para_offset: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Held-out speakers A/B: two origin_vs_para triplets each plus extra real lines; the paraphrase sits
    para_offset away from its anchor along a content axis shared by every speaker."""
    rng = np.random.default_rng(0)
    style = {s: rng.normal(size=6) * 3 for s in "ABST"}
    recs, vecs = [], []

    def add(tid, role, rule, speaker, heldout, vec):
        recs.append({"triplet_id": tid, "role": role, "rule": rule, "heldout": heldout, "speaker": speaker,
                     "text_preview": f"{tid}{role}", "db_id": len(recs)})
        vecs.append(vec)

    for speaker in "AB":
        for i in range(2):
            content = rng.normal(size=6)
            anchor = style[speaker] + content
            add(f"{speaker}{i}", "anchor", "origin_vs_para", speaker, True, anchor)
            add(f"{speaker}{i}", "positive", "origin_vs_para", speaker, True, style[speaker] + rng.normal(size=6))
            add(f"{speaker}{i}", "negative", "origin_vs_para", "", True, content + para_offset * anchor)
        for i in range(4):
            add(f"{speaker}r{i}", "anchor", "same_speaker_real", speaker, True, style[speaker] + rng.normal(size=6))
    for speaker in "ST":
        for i in range(6):
            add(f"{speaker}{i}", "anchor", "same_speaker_real", speaker, False, style[speaker] + rng.normal(size=6))
    return pd.DataFrame(recs), np.stack(vecs)


def test_paraphrase_rank_separates_style_from_content_and_nullspace_keeps_shape() -> None:
    rows, emb = probe_rows(0.0)
    far = paraphrase_rank(rows, emb)
    assert far["n_para_rank_heldout"] == 4 and far["para_beats_own_heldout"] < 0.5
    near = paraphrase_rank(*probe_rows(0.99))
    assert near["para_beats_own_heldout"] > far["para_beats_own_heldout"] and near["para_top1_heldout"] > 0
    projected = within_speaker_nullspace(rows, emb, 2)
    assert projected.shape == emb.shape and np.allclose(np.linalg.norm(projected, axis=1), 1.0)


def test_line_leak_is_one_half_without_line_content_and_rises_with_it() -> None:
    from report import line_leak

    rng = np.random.default_rng(1)
    style, recs, vecs = rng.normal(size=16), [], []
    for i in range(40):
        content, para_style = rng.normal(size=16), rng.normal(size=16) * 0.3
        for role, vec in (("anchor", style + content), ("positive", style + rng.normal(size=16)),
                          ("negative", para_style + content)):
            recs.append({"triplet_id": f"t{i}", "role": role, "rule": "origin_vs_para", "heldout": True,
                         "speaker": "A", "text_preview": f"t{i}{role}"})
            vecs.append(vec)
    rows, emb = pd.DataFrame(recs), np.stack(vecs)
    leaky = line_leak(rows, emb)
    assert leaky["n_line_leak_heldout"] == 40 and leaky["line_leak_heldout"] > 0.9
    # Drop the content from the anchors: its own paraphrase is no closer than another line's.
    is_anchor = (rows["role"] == "anchor").to_numpy()
    emb[is_anchor] = style + rng.normal(size=(40, 16)) * 0.01
    assert abs(line_leak(rows, emb)["line_leak_heldout"] - 0.5) < 0.1
    assert line_leak(rows.assign(heldout=False), emb) == {}


def test_paraphrase_nullspace_removes_the_shared_content_axis() -> None:
    from report import paraphrase_nullspace

    rows, emb = probe_rows(0.0)
    assert paraphrase_nullspace(rows, emb, 2) is None  # only A/B (held out) have paraphrases here
    rows = rows.copy()
    rows["heldout"] = False
    projected = paraphrase_nullspace(rows, emb, 2)
    assert projected.shape == emb.shape and np.allclose(np.linalg.norm(projected, axis=1), 1.0)
