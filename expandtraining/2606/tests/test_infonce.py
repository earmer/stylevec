from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from losses import info_nce_loss


def unit(*rows: list[float]) -> torch.Tensor:
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def toy_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = unit([1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0])
    positive = unit([0.8, 0.2, 0, 0], [0.7, 0.3, 0, 0], [0, 0, 0.9, 0.1])
    negative = unit([0, 1, 0, 0], [0, 0.8, 0.2, 0], [0.6, 0.4, 0, 0])
    return anchor, positive, negative


def nce(anchor, positive, negative, anchor_groups, negative_groups, temperature=0.05):
    batch = anchor.shape[0]
    return info_nce_loss(
        anchor, positive, negative, torch.full((batch,), 0.1), torch.ones(batch),
        torch.tensor(anchor_groups), torch.tensor(negative_groups), temperature,
    )


def test_info_nce_matches_manual_masked_cross_entropy() -> None:
    anchor, positive, negative = toy_batch()
    # Rows 0 and 1 are one speaker; row 2's negative is also that speaker.
    loss, stats = nce(anchor, positive, negative, [0, 0, 1], [-1, -2, 0])
    expected = []
    for i in range(3):
        keep = [positive[i]]
        keep += [positive[j] for j in range(3) if j != i and [0, 0, 1][j] != [0, 0, 1][i]]
        keep += [anchor[j] for j in range(3) if j != i and [0, 0, 1][j] != [0, 0, 1][i]]
        keep += [negative[j] for j in range(3) if j == i or [-1, -2, 0][j] != [0, 0, 1][i]]
        logits = torch.stack(keep) @ anchor[i] / 0.05
        expected.append(-torch.log_softmax(logits, dim=0)[0])
    assert torch.allclose(stats["raw_loss"], torch.stack(expected), atol=1e-5)
    assert torch.allclose(loss, torch.stack(expected).mean(), atol=1e-5)
    # Masked out of 3 * (B - 1) = 6 candidates: rows 0/1 lose each other's P and A plus row 2's N; row 2 none.
    assert stats["masked_fraction"].item() == pytest.approx((3 / 6 + 3 / 6 + 0) / 3)


def test_info_nce_ignores_same_speaker_candidates() -> None:
    anchor, positive, negative = toy_batch()
    _, base = nce(anchor, positive, negative, [0, 0, 1], [-1, -2, -3])
    # Row 1's positive moved onto row 0's anchor: a false negative for row 0 if it were counted.
    moved = positive.clone()
    moved[1] = anchor[0]
    _, masked = nce(anchor, moved, negative, [0, 0, 1], [-1, -2, -3])
    assert masked["raw_loss"][0].item() == pytest.approx(base["raw_loss"][0].item(), abs=1e-6)
    _, unmasked = nce(anchor, moved, negative, [0, 1, 2], [-1, -2, -3])
    assert unmasked["raw_loss"][0].item() > base["raw_loss"][0].item() + 1.0
    assert unmasked["masked_fraction"].item() == 0.0


def test_info_nce_keeps_own_mined_negative() -> None:
    anchor, positive, negative = toy_batch()
    # Even labelled with the anchor's own style, the row's mined negative stays in its denominator.
    _, far = nce(anchor, positive, negative, [0, 1, 2], [0, 1, 2])
    closer = negative.clone()
    closer[0] = positive[0]
    _, near = nce(anchor, positive, closer, [0, 1, 2], [0, 1, 2])
    assert near["raw_loss"][0].item() > far["raw_loss"][0].item()


def test_info_nce_decreases_when_positive_is_closer() -> None:
    anchor, positive, negative = toy_batch()
    groups = ([0, 1, 2], [-1, -2, -3])
    loose, _ = nce(anchor, positive, negative, *groups)
    tighter = positive.clone()
    tighter[0] = unit([0.98, 0.02, 0, 0])[0]
    tight, stats = nce(anchor, tighter, negative, *groups)
    assert tight.item() < loose.item()
    assert stats["pos_cos"][0].item() > 0.99


def test_info_nce_gradient_flows_and_weights_scale() -> None:
    anchor, positive, negative = (x.clone().requires_grad_(True) for x in toy_batch())
    groups = (torch.tensor([0, 1, 2]), torch.tensor([-1, -2, -3]))
    loss, _ = info_nce_loss(anchor, positive, negative, torch.full((3,), 0.1), torch.ones(3), *groups, 0.05)
    loss.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    doubled, _ = info_nce_loss(anchor, positive, negative, torch.full((3,), 0.1), torch.full((3,), 2.0), *groups, 0.05)
    assert torch.allclose(doubled, loss * 2.0)


def test_collate_labels_styles_and_unknowns_never_match() -> None:
    from dataset import TripletCollate
    from records import TripletRecord
    from train import style_group_ids

    class Tok:
        def __call__(self, texts, **_kwargs):
            return {"input_ids": torch.zeros(len(texts), 2, dtype=torch.long),
                    "attention_mask": torch.ones(len(texts), 2, dtype=torch.long)}

    rows = [
        TripletRecord("a", "p", "n", "same_speaker_real", 1.0, 0.1, meta={"speaker": "X", "negative_speaker": "Y"}),
        TripletRecord("a2", "p2", "n2", "origin_vs_para", 1.0, 0.1, meta={"speaker": "X"}),
        TripletRecord("a3", "p3", "n3", "generated_to_target", 1.0, 0.1,
                      meta={"target_speaker": "Y", "source_speaker": "X"}),
        TripletRecord("a4", "p4", "n4", "length_course_genshin", 1.0, 0.1,
                      meta={"domain": "genshin", "source": "X", "negative_source": "Z"}),
        TripletRecord("a5", "p5", "n5", "unknown_rule", 1.0, 0.1),
    ]
    batch = TripletCollate(Tok(), 8)(rows)
    assert batch["anchor_style"] == ["genshin:X", "genshin:X", "genshin:Y", "genshin:X", None]
    assert batch["negative_style"] == ["genshin:Y", None, "genshin:X", "genshin:Z", None]
    anchor_groups, negative_groups = style_group_ids(batch, torch.device("cpu"))
    assert anchor_groups[0] == anchor_groups[1] == anchor_groups[3] == negative_groups[2]
    assert anchor_groups[2] == negative_groups[0]
    unknown = [int(anchor_groups[4]), int(negative_groups[1]), int(negative_groups[4])]
    assert len(set(unknown)) == 3 and all(value < 0 for value in unknown)


def test_matryoshka_trains_only_listed_prefixes_and_validation_keeps_its_own() -> None:
    from config import configured_matryoshka_dims, validation_matryoshka_dims

    config = {"model": {"projection_dim": 256}, "regularization": {"matryoshka_dims": [32, 64, 128]},
              "validation": {}}
    # Default: one key drives both, as before.
    assert configured_matryoshka_dims(config) == validation_matryoshka_dims(config) == [32, 64, 128]
    config["regularization"]["matryoshka_dims"] = [64]
    config["validation"]["matryoshka_dims"] = [32, 64, 128, 256]
    assert configured_matryoshka_dims(config) == [64]
    assert validation_matryoshka_dims(config) == [32, 64, 128]  # 256 is the full embedding, always reported


@pytest.mark.parametrize("loss_type", ["triplet", "infonce"])
def test_matryoshka_step_loss_renormalises_each_training_prefix(loss_type: str) -> None:
    from losses import weighted_triplet_margin_loss
    from train import matryoshka_step_loss, objective_loss, style_group_ids

    torch.manual_seed(0)
    anchor, positive, negative = (F.normalize(torch.randn(6, 256), dim=-1) for _ in range(3))
    batch = {"margin": torch.full((6,), 0.1), "weight": torch.ones(6),
             "anchor_style": ["a", "a", "b", "c", None, "d"], "negative_style": [None, "b", "a", None, "c", None]}
    settings = {"loss": loss_type, "temperature": 0.05, "triplet_weight": 0.0}
    groups = style_group_ids(batch, torch.device("cpu"))
    loss, stats = matryoshka_step_loss(anchor, positive, negative, batch, [64], 0.2, settings, groups)
    assert {key for key in stats if key.startswith("loss_dim_")} == {"loss_dim_64"}
    prefix = [F.normalize(x[:, :64], dim=-1) for x in (anchor, positive, negative)]
    expected, _ = objective_loss(*prefix, batch, settings, groups)
    assert torch.allclose(loss, expected * 0.2, atol=1e-6)
    if loss_type == "triplet":
        same, _ = weighted_triplet_margin_loss(*prefix, batch["margin"], batch["weight"])
        assert torch.allclose(expected, same)
    # The prefixes are unit vectors again before the loss sees them.
    from losses import matryoshka_prefix_loss

    seen: list[torch.Tensor] = []

    def spy(a, p, n):
        seen.extend([a, p, n])
        return objective_loss(a, p, n, batch, settings, groups)

    matryoshka_prefix_loss(anchor, positive, negative, [64], spy)
    assert all(x.shape[-1] == 64 and torch.allclose(x.norm(dim=-1), torch.ones(6)) for x in seen)


def test_objective_defaults_to_the_triplet_hinge_and_mixes_on_request() -> None:
    from losses import weighted_triplet_margin_loss
    from train import objective_loss, objective_settings_from_config

    assert objective_settings_from_config({"train": {}}) == {"loss": "triplet", "temperature": 0.05, "triplet_weight": 0.0}
    anchor, positive, negative = toy_batch()
    batch = {"margin": torch.full((3,), 0.1), "weight": torch.ones(3),
             "anchor_style": ["a", "b", "c"], "negative_style": [None, None, None]}
    triplet, _ = weighted_triplet_margin_loss(anchor, positive, negative, batch["margin"], batch["weight"])
    default, _ = objective_loss(anchor, positive, negative, batch, objective_settings_from_config({"train": {}}))
    assert torch.equal(default, triplet)
    pure, stats = objective_loss(anchor, positive, negative, batch, {"loss": "infonce", "temperature": 0.05,
                                                                       "triplet_weight": 0.0})
    assert "triplet_loss" not in stats and {"masked_fraction", "accuracy", "violations"} <= set(stats)
    mixed, stats = objective_loss(anchor, positive, negative, batch, {"loss": "infonce", "temperature": 0.05,
                                                                        "triplet_weight": 0.5})
    assert torch.allclose(mixed, pure + triplet * 0.5)
    assert torch.allclose(stats["triplet_loss"], triplet)


def test_infonce_spec_keys_round_trip_through_the_run_config(tmp_path: Path) -> None:
    from ablation import load_base, load_spec, materialize
    from config import (
        config_snapshot,
        configured_matryoshka_dims,
        validate_loss_config,
        validation_matryoshka_dims,
    )
    from prebuilt import data_key
    from train import objective_settings_from_config

    spec = load_spec(Path(__file__).resolve().parents[1] / "configs" / "ablations" / "infonce_a_1gpu.yaml")
    base = load_base(spec)
    _shared, arms = materialize(spec, base)
    loaded = {arm: yaml.safe_load(config_snapshot(config)) for arm, config in arms.items()}
    for arm, config in loaded.items():
        assert config == {k: v for k, v in arms[arm].items() if not k.startswith("_")}
        validate_loss_config(config)
        assert validation_matryoshka_dims(config) == [32, 64, 128]
    assert objective_settings_from_config(loaded["center"])["loss"] == "triplet"
    assert objective_settings_from_config(loaded["center_infonce"]) == {
        "loss": "infonce", "temperature": 0.05, "triplet_weight": 0.0}
    assert configured_matryoshka_dims(loaded["center_infonce"]) == [32, 64, 128]
    assert configured_matryoshka_dims(loaded["center_infonce_m64"]) == [64]
    # The reported prefixes are not part of the data key, so every arm reuses the base's dataset.
    assert data_key(loaded["center_infonce_m64"]) == data_key(base)
    with pytest.raises(ValueError, match="train.loss"):
        validate_loss_config({"train": {"loss": "hinge"}})
    with pytest.raises(ValueError, match="temperature"):
        validate_loss_config({"train": {"loss": "infonce", "infonce_temperature": 0.0}})
