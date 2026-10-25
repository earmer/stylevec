from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_triplets import build_msynthstel_triplets
from config import load_config
from import_msynthstel import load_msynthstel_samples
from trait_directions import FAMILIES, TraitDirections, trait_direction_loss, validate_trait_direction_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dryrun.yaml"


def pairs_config(**data) -> dict:
    config = copy.deepcopy(load_config(CONFIG))
    config["data"].update({"msynthstel_mode": "pairs", "msynthstel_features": 0, "msynthstel_rows_per_feature": 90, **data})
    return config


def test_pairs_mode_keeps_every_pole_pair_with_its_own_negative() -> None:
    config = pairs_config()
    samples, _ = load_msynthstel_samples(config)
    rows = build_msynthstel_triplets(config, samples, "train")
    assert {row.rule for row in rows} == {"trait_positive_vs_negative"}
    poles = Counter(row.meta["pole"] for row in rows)
    assert len(poles) == 40 and set(poles.values()) == {90}
    assert set(FAMILIES) <= set(poles), sorted(set(FAMILIES) - set(poles))
    by_text = {sample.text: sample.content_id for sample in samples}
    assert all(by_text[row.anchor_text] == by_text[row.negative_text] for row in rows)
    assert all(by_text[row.anchor_text] != by_text[row.positive_text] for row in rows)


def test_triplets_mode_is_unchanged() -> None:
    config = load_config(CONFIG)
    samples, _ = load_msynthstel_samples(config)
    rows = build_msynthstel_triplets(config, samples, "train")
    assert "trait_cross_feature_hard" in {row.rule for row in rows}
    assert not any("pole" in row.meta for row in rows)


def outputs_for(poles: list[str], directions: dict[str, torch.Tensor], dim: int = 8, seed: int = 0):
    """Pairs whose difference is the pole's direction plus noise, on random (shared) content."""
    generator = torch.Generator().manual_seed(seed)
    content = torch.randn(len(poles), dim, generator=generator)
    shift = torch.stack([directions[p] for p in poles]) + 0.1 * torch.randn(len(poles), dim, generator=generator)
    anchor = (content + shift).requires_grad_(True)
    negative = content.clone().requires_grad_(True)
    batch = {"rule": ["trait_positive_vs_negative"] * len(poles), "meta": [{"pole": p} for p in poles]}
    return {"embeddings": (anchor, anchor.detach(), negative)}, batch


def settings(weight: float = 1.0) -> dict:
    return {"weight": weight, "temperature": 0.1, "momentum": 0.5}


def test_consistent_directions_are_classified_and_the_loss_reaches_both_poles() -> None:
    torch.manual_seed(0)
    poles = ["Polite / Impolite", "Active / Passive", "Complex / Simple", "With Humor / Without Humor"]
    directions = {p: F.normalize(torch.randn(8), dim=-1) for p in poles}
    state = TraitDirections(settings())
    for step in range(5):
        outputs, batch = outputs_for(poles * 2, directions, seed=step)
        loss, stats = trait_direction_loss(state, outputs, batch)
    assert stats["accuracy"] == 1.0 and stats["features_seen"] == 4 and stats["own_cos"] > 0.9
    loss.backward()
    anchor, _, negative = outputs["embeddings"]
    assert anchor.grad.abs().sum() > 0 and negative.grad.abs().sum() > 0


def test_family_members_leave_each_others_denominator() -> None:
    case = ["All Lower Case / Proper Capitalization", "All Upper Case / Proper Capitalization"]
    state = TraitDirections(settings())
    ids = state.ids([*case, "Polite / Impolite"], torch.device("cpu"))
    mask = state.family_mask(ids)
    assert mask[0, 1] and mask[1, 0] and not mask[0, 0] and not mask[0, 2] and not mask[2].any()


def test_off_and_unlabelled_rows_add_nothing() -> None:
    directions = {"Polite / Impolite": torch.ones(8)}
    outputs, batch = outputs_for(["Polite / Impolite"], directions)
    loss, stats = trait_direction_loss(TraitDirections(settings(0.0)), outputs, batch)
    assert float(loss) == 0.0 and stats == {}
    batch["meta"] = [{}]
    loss, stats = trait_direction_loss(TraitDirections(settings()), outputs, batch)
    assert float(loss) == 0.0 and stats == {}
    first, _ = trait_direction_loss(state := TraitDirections(settings()), *outputs_for(["Polite / Impolite"], directions))
    assert float(first) == 0.0  # one feature seen: nothing to contrast with yet
    assert state.seen.sum() == 1


def test_config_needs_pairs_mode() -> None:
    config = load_config(CONFIG)
    config.setdefault("regularization", {})["trait_direction_weight"] = 1.0
    with pytest.raises(ValueError, match="msynthstel_mode: pairs"):
        validate_trait_direction_config(config)
    config["data"]["msynthstel_mode"] = "pairs"
    validate_trait_direction_config(config)
    config["regularization"]["trait_direction_temperature"] = 0.0
    with pytest.raises(ValueError):
        validate_trait_direction_config(config)
