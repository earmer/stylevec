"""Leak regularisers: penalties that keep a line's own content out of its style embedding, judged against its
content twin (the anchor's paraphrase, or a rewrite's source; dataset.twin_columns).

The leak InfoNCE taught is pairwise: z(line) close to z(its paraphrase). A linear decorrelation from a small
content code (infonce_b chead) sat at its noise floor while the leak grew, so these methods all see the
instance-level pair. The origin_vs_para hinge already targets the pair but fires on ~3% of rows; a twin
reaches ~63% of rows, which is the coverage these methods add.

One method per run, `regularization.leak_method` (`none` is off), at `leak_weight` until `leak_until_step`
(0 = every step). The twin is the target, encoded without gradient (encode_triplet_outputs detach_twin), so a
method costs one extra forward and no gradient checkpointing. A row whose twin is its own negative reads the
twin from the (detached) negative. `leak_queue` keeps that many past (style, twin) pairs, detached, so batch
statistics see more than one micro-batch's ~20 pairs.
"""

from __future__ import annotations

from typing import Any

import torch

LEAK_METHODS = ("none", "subspace")


def leak_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    regularization = config.get("regularization", {})
    return {
        "method": str(regularization.get("leak_method", "none")),
        "weight": float(regularization.get("leak_weight", 0.0)),
        "until_step": int(regularization.get("leak_until_step", 0)),
        "queue": int(regularization.get("leak_queue", 0)),
    }


def leak_enabled(settings: dict[str, Any]) -> bool:
    """subspace is a projection inside the model (model.leak_null_*), not a weighted loss."""
    if settings["method"] == "subspace":
        return True
    return settings["method"] != "none" and settings["weight"] > 0.0


def validate_leak_config(config: dict[str, Any]) -> None:
    settings = leak_settings_from_config(config)
    if settings["method"] not in LEAK_METHODS:
        raise ValueError(f"regularization.leak_method must be one of {LEAK_METHODS}, got {settings['method']!r}")
    null_dims = int(config["model"].get("leak_null_dims", 0))
    if (settings["method"] == "subspace") != (null_dims > 0):
        raise ValueError("regularization.leak_method subspace and model.leak_null_dims > 0 go together")
    if settings["weight"] < 0.0:
        raise ValueError("regularization.leak_weight must be non-negative")
    if settings["queue"] < 0:
        raise ValueError("regularization.leak_queue must be non-negative (0 = this micro-batch only)")
    max_steps = int(config["train"].get("max_steps", 0))
    if settings["until_step"] < 0 or (settings["until_step"] > 0 and settings["until_step"] > max_steps):
        raise ValueError("regularization.leak_until_step must be in [0, train.max_steps] (0 = off)")


class PairQueue:
    """The last `size` (style, twin) pairs, detached, oldest dropped first."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.style: torch.Tensor | None = None
        self.twin: torch.Tensor | None = None

    def extend(self, style: torch.Tensor, twin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This batch's pairs followed by the queued ones (gradient only through `style`'s rows); then
        queues this batch's pairs."""
        if self.size <= 0 or self.style is None:
            all_style, all_twin = style, twin
        else:
            all_style, all_twin = torch.cat([style, self.style]), torch.cat([twin, self.twin])
        if self.size > 0:
            self.style = all_style.detach()[: self.size]
            self.twin = all_twin.detach()[: self.size]
        return all_style, all_twin


class LeakState:
    """Per-run state a method keeps across steps: the pair queue, and the encoder's content_null (subspace),
    whose covariance this method feeds."""

    def __init__(self, settings: dict[str, Any], null: torch.nn.Module | None = None) -> None:
        self.settings = settings
        self.queue = PairQueue(settings["queue"])
        self.null = null
        if settings["method"] == "subspace" and null is None:
            raise ValueError("leak_method subspace needs the encoder's content_null (model.leak_null_dims)")


def subspace_update(state: LeakState, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Feeds this batch's unprojected twins to the encoder's content_null; returns how much of the anchors'
    and twins' (unprojected) embeddings sits in the content subspace."""
    anchor, _positive, negative = outputs["unprojected"]
    has_twin, is_negative = batch["has_twin"], batch["twin_is_negative"]
    twin = torch.where(is_negative[:, None], negative.detach(), outputs["twin_unprojected"])
    keep = has_twin | is_negative
    if int(keep.sum()) < 2:
        return {}
    state.null.update(twin[keep])
    return {"null_share_anchor": state.null.share(anchor[keep].detach()).detach(),
            "null_share_twin": state.null.share(twin[keep]).detach(),
            "null_updates": state.null.updates.float()}


def twin_pairs(outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """(anchor style embeddings, detached twin embeddings) for every row with a twin."""
    anchor, _positive, negative = outputs["embeddings"]
    has_twin, is_negative = batch["has_twin"], batch["twin_is_negative"]
    twin = torch.where(is_negative[:, None], negative.detach(), outputs["twin"].detach())
    keep = has_twin | is_negative
    return anchor[keep], twin[keep]


def leak_step_loss(
    state: LeakState,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    global_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = state.settings
    zero = outputs["embeddings"][0].new_tensor(0.0)
    if not leak_enabled(settings) or (0 < settings["until_step"] <= global_step):
        return zero, {}
    if settings["method"] == "subspace":
        return zero, subspace_update(state, outputs, batch)
    style, twin = twin_pairs(outputs, batch)
    if style.shape[0] < 2:
        return zero, {}
    style, twin = state.queue.extend(style, twin)
    pair_cos = (style * twin).sum(dim=-1)
    stats = {"pairs": style.new_tensor(float(style.shape[0])), "pair_cos": pair_cos.detach().mean()}
    raise ValueError(f"unknown leak method {settings['method']!r}")
