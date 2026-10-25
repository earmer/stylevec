"""Trait directions: learn one direction per msynthstel feature from its minimal pairs.

Each msynthstel row is one sentence written at both poles of a feature ("Polite / Impolite"); the trait
triplet encodes the first pole as its anchor and the second as its negative. As a triplet the pair asks one
feature to outweigh a whole shared sentence, which a voice-shaped space never grants (violations stayed
~0.9). The difference d = z(pole A) - z(pole B) instead cancels the shared content and keeps the change, so
the loss asks only that a feature's differences point one way and other features' another:

    loss = CE(cos(d, prototypes) / temperature, own feature)

Prototypes are per-feature running means of the (detached, normalised) differences, so a micro-batch with
one trait row still has every feature to contrast with. Features of one family (same pole kind, e.g. the
three capitalisation features) are masked out of each other's denominators: telling them apart is not the
point, and their data overlaps. Prototypes are not checkpointed; on resume they rebuild within a few steps.

`regularization.trait_direction_weight` (0 = off) at `trait_direction_temperature`, prototypes kept with
`trait_direction_momentum`. Rows are the `trait_positive_vs_negative` triplets of `data.msynthstel_mode:
pairs`, which labels each with its pole pair (meta `pole`); a small weights.msynthstel (rows need a positive
weight; 0.01) all but drops their hinge and InfoNCE terms.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

TRAIT_RULE = "trait_positive_vs_negative"
MAX_FEATURES = 64

# Pole pairs (msynthstel `feature`) whose data overlaps; unlisted pairs are a family of their own.
FAMILIES: dict[str, str] = {
    "All Lower Case / Proper Capitalization": "case",
    "All Upper Case / Proper Capitalization": "case",
    "With uppercase letters / Without uppercase letters": "case",
    "With Emojis / No Emojis": "emoji",
    "Text Emojis / No Emojis": "emoji",
    "With digits / Less frequent digits": "digits",
    "With number substitution / Without number substitution": "digits",
    "Self-focused / Inclusive-focused": "person",
    "Self-focused / Third-person singular": "person",
    "Self-focused / You-focused": "person",
    "Self-focused / Audience-focused": "person",
    "With personal pronouns / Less frequent pronouns": "person",
    "With pronouns / Less frequent pronouns": "person",
    "With function words / Less frequent function words": "function",
    "With articles / Less frequent articles": "function",
    "With determiners / Less frequent determiners": "function",
    "With prepositions / Less frequent prepositions": "function",
    "With conjunctions / Less frequent conjunctions": "function",
    "With common verbs / Less frequent common verbs": "function",
    "Affective process / Perceptual process": "process",
    "Affective processes / Cognitive processes": "process",
    "Cognitive process / Perceptual process": "process",
    "Present-focused / Future-focused": "tense",
    "Present-focused / Past-focused": "tense",
    "Polite / Impolite": "register",
    "Formal / Informal": "register",
    "Offensive / Non-Offensive": "register",
    "With contractions / Without contractions": "register",
}


def trait_direction_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    regularization = config.get("regularization", {})
    return {
        "weight": float(regularization.get("trait_direction_weight", 0.0)),
        "temperature": float(regularization.get("trait_direction_temperature", 0.1)),
        "momentum": float(regularization.get("trait_direction_momentum", 0.9)),
    }


def validate_trait_direction_config(config: dict[str, Any]) -> None:
    settings = trait_direction_settings_from_config(config)
    if settings["weight"] < 0.0 or settings["temperature"] <= 0.0 or not 0.0 <= settings["momentum"] < 1.0:
        raise ValueError("regularization.trait_direction_weight >= 0, _temperature > 0, _momentum in [0, 1)")
    mode = str(config.get("data", {}).get("msynthstel_mode", "triplets"))
    if settings["weight"] > 0.0 and mode != "pairs":
        raise ValueError("regularization.trait_direction_weight needs data.msynthstel_mode: pairs (pole labels)")


class TraitDirections:
    """Per-feature prototypes (unit vectors) and which features they belong to."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.index: dict[str, int] = {}
        self.families: list[str] = []
        self.prototypes: torch.Tensor | None = None
        self.seen: torch.Tensor | None = None

    def ids(self, poles: list[str], device: torch.device) -> torch.Tensor:
        for pole in poles:
            if pole not in self.index:
                if len(self.index) >= MAX_FEATURES:
                    raise ValueError(f"more than {MAX_FEATURES} trait features")
                self.index[pole] = len(self.index)
                self.families.append(FAMILIES.get(pole, pole))
        return torch.tensor([self.index[pole] for pole in poles], dtype=torch.long, device=device)

    def update(self, directions: torch.Tensor, ids: torch.Tensor) -> None:
        if self.prototypes is None:
            self.prototypes = directions.new_zeros(MAX_FEATURES, directions.shape[1])
            self.seen = torch.zeros(MAX_FEATURES, dtype=torch.bool, device=directions.device)
        momentum = self.settings["momentum"]
        for row, feature in zip(directions, ids.tolist()):
            old = self.prototypes[feature]
            new = row if not self.seen[feature] else momentum * old + (1.0 - momentum) * row
            self.prototypes[feature] = F.normalize(new, dim=-1)
            self.seen[feature] = True

    def family_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where prototype j is another feature of row i's family (excluded from its denominator)."""
        families = self.families + [""] * (MAX_FEATURES - len(self.families))
        rows = [[families[j] == self.families[i] and j != i for j in range(MAX_FEATURES)] for i in ids.tolist()]
        return torch.tensor(rows, dtype=torch.bool, device=ids.device)


def trait_pairs(outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
    """(z(pole A) - z(pole B), pole name) for the batch's labelled trait rows."""
    anchor, _positive, negative = outputs["embeddings"]
    keep = [i for i, (rule, meta) in enumerate(zip(batch["rule"], batch["meta"])) if rule == TRAIT_RULE and meta.get("pole")]
    if not keep:
        return anchor.new_zeros(0, anchor.shape[1]), []
    rows = torch.tensor(keep, dtype=torch.long, device=anchor.device)
    return anchor[rows] - negative[rows], [str(batch["meta"][i]["pole"]) for i in keep]


def trait_direction_loss(
    state: TraitDirections,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = state.settings
    zero = outputs["embeddings"][0].new_tensor(0.0)
    if settings["weight"] <= 0.0:
        return zero, {}
    differences, poles = trait_pairs(outputs, batch)
    if not poles:
        return zero, {}
    with torch.autocast(device_type=differences.device.type, enabled=False):
        differences = differences.float()
        ids = state.ids(poles, differences.device)
        directions = F.normalize(differences, dim=-1, eps=1e-6)
        stats = {"rows": zero.new_tensor(float(len(poles))), "delta_norm": differences.norm(dim=-1).mean().detach()}
        loss = zero.float()
        if state.prototypes is not None:
            known = state.seen[ids]
            if known.any() and int(state.seen.sum()) > 1:
                # A copy: update() below rewrites the prototypes before backward needs them.
                prototypes = state.prototypes.clone()
                logits = directions[known] @ prototypes.T / settings["temperature"]
                excluded = state.family_mask(ids[known]) | ~state.seen[None, :]
                logits = logits.masked_fill(excluded, float("-inf"))
                targets = ids[known]
                loss = F.cross_entropy(logits, targets)
                stats["accuracy"] = (logits.argmax(dim=-1) == targets).float().mean().detach()
                stats["own_cos"] = (directions[known] * prototypes[targets]).sum(dim=-1).mean().detach()
                stats["ce"] = loss.detach()
        state.update(directions.detach(), ids)
        stats["features_seen"] = zero.new_tensor(float(state.seen.sum()))
    return loss * settings["weight"], stats
