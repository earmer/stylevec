"""Staged sampling over "how much variation must be ignored" (automating.md, section three).

Rules are grouped into tiers; each stage mixes tiers by fixed probabilities.
Later stages raise the hard share gradually (~20% -> 50-60%), never all-hard.
`mode: mixed` samples the same total budget from the time-averaged mixture at every position,
which is the control the curriculum must beat.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

# easy: build basic identity; mid: reduce content shortcut; hard: fine source differences.
RULE_TIERS: dict[str, str] = {
    "same_speaker_real": "easy",
    "bookraw_same_work": "easy",
    "trait_positive_vs_negative": "easy",
    "generated_to_target": "easy",
    "bookraw_same_language_author_negative": "mid",
    "bookraw_same_author_cross_work": "mid",
    "origin_vs_para": "mid",
    "target_hard_negative": "mid",
    "trait_cross_feature_hard": "mid",
    "same_speaker_topic_matched": "hard",
    "same_content_diff_style": "hard",
    "paratarget_original_push": "mid",
}

DEFAULT_STAGES: list[dict[str, Any]] = [
    {"until": 0.35, "tiers": {"easy": 0.7, "mid": 0.25, "hard": 0.05}},
    {"until": 0.70, "tiers": {"easy": 0.35, "mid": 0.40, "hard": 0.25}},
    {"until": 1.00, "tiers": {"easy": 0.20, "mid": 0.30, "hard": 0.50}},
]


def rule_tier(rule: str) -> str:
    if rule.startswith("length_course_"):
        return "easy"
    return RULE_TIERS.get(rule, "mid")


def curriculum_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    stages = config.get("curriculum", {}).get("stages") or DEFAULT_STAGES
    previous = 0.0
    for stage in stages:
        if not previous < float(stage["until"]) <= 1.0:
            raise ValueError(f"curriculum stage boundaries must increase within (0, 1]: {stages}")
        if sum(float(v) for v in stage["tiers"].values()) <= 0:
            raise ValueError(f"curriculum stage has no mass: {stage}")
        previous = float(stage["until"])
    if previous != 1.0:
        raise ValueError("last curriculum stage must end at 1.0")
    return stages


def stage_at(stages: list[dict[str, Any]], progress: float) -> int:
    for index, stage in enumerate(stages):
        if progress < float(stage["until"]):
            return index
    return len(stages) - 1


def mean_tier_weights(stages: list[dict[str, Any]]) -> dict[str, float]:
    total: defaultdict[str, float] = defaultdict(float)
    previous = 0.0
    for stage in stages:
        share = float(stage["until"]) - previous
        mass = sum(float(v) for v in stage["tiers"].values())
        for tier, value in stage["tiers"].items():
            total[tier] += share * float(value) / mass
        previous = float(stage["until"])
    return dict(total)


def curriculum_order(
    triplets: list[Any],
    config: dict[str, Any],
    seed: int,
    total: int,
) -> list[Any]:
    """Return `total` triplets ordered so position/total maps to a curriculum stage.

    Triplets are drawn without replacement within a rule until the rule is exhausted, then reshuffled.
    """
    cfg = config.get("curriculum", {})
    mode = str(cfg.get("mode", "staged"))
    if mode not in {"staged", "mixed"}:
        raise ValueError(f"curriculum.mode must be staged or mixed, got {mode}")
    stages = curriculum_stages(config)
    rng = random.Random(seed)
    by_rule: dict[str, list[Any]] = defaultdict(list)
    for row in triplets:
        by_rule[row.rule].append(row)
    tier_rules: dict[str, list[str]] = defaultdict(list)
    for rule in sorted(by_rule):
        tier_rules[rule_tier(rule)].append(rule)
    queues: dict[str, list[Any]] = {rule: [] for rule in by_rule}

    def draw(rule: str) -> Any:
        if not queues[rule]:
            queues[rule] = list(by_rule[rule])
            rng.shuffle(queues[rule])
        return queues[rule].pop()

    fixed_mixed = mean_tier_weights(stages)
    ordered: list[Any] = []
    for position in range(total):
        weights = fixed_mixed if mode == "mixed" else stages[stage_at(stages, position / total)]["tiers"]
        tiers = [tier for tier in sorted(tier_rules) if float(weights.get(tier, 0.0)) > 0.0]
        if not tiers:  # stage names only tiers with no data: fall back to whatever exists
            tiers = sorted(tier_rules)
            weights = {tier: 1.0 for tier in tiers}
        tier = rng.choices(tiers, weights=[float(weights.get(t, 0.0)) for t in tiers])[0]
        ordered.append(draw(rng.choice(tier_rules[tier])))
    return ordered
