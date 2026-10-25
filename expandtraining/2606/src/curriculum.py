"""Sampling schedules over rule groups ("tiers") as a function of training progress.

`curriculum.grouping` picks how rules are grouped:

  difficulty   easy / mid / hard (automating.md section three; the legacy default, kept so old runs replay)
  cleanliness  clean / topic / confounded: can the anchor's content alone pick the positive?
                 clean       content is held fixed (the negative is the same text in another voice) or the
                             data is synthetic with one style feature varied: content cannot solve it
                 topic       real text, but the positive is off-scene and the negative shares the scene:
                             content points at the negative
                 confounded  shared topic, scene or overlap can pick the positive (same_speaker_real,
                             length_course_*, same-work bookraw); unknown rules land here too
               Easy-first put confounded rules first, so the content shortcut was learned from step 0; the
               cleanliness default starts clean-only and adds confounded rows last and thin (#5).

A schedule gives tier probabilities as a function of progress. Every mode spends the same row budget, so
arms differ only in what is drawn when:

  staged   stage mixes, easy-heavy -> hard-heavy (~20% -> 50-60% hard, never all-hard)
  anti     the staged schedule played backwards; same exposure as staged, opposite order
  mixed    staged's time-averaged mix at every position; same exposure, no order (the control)
  uniform  every rule equally likely, tiers ignored (~ the legacy rule-balanced order)

`pacing: step` switches mixes at stage boundaries; `pacing: linear` interpolates between stage midpoints.
`curriculum.tiers` moves a rule to another tier of the grouping, or out of training with `drop`.
`curriculum.rule_shares` weights rules within their tier (default 1 each; in uniform mode, across all rules).
CURRICULUM.md says which pair of modes isolates which effect.
"""
from __future__ import annotations

import random
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import accumulate
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

# Whether the anchor's content alone can pick the positive; see the module docstring.
RULE_CLEANLINESS: dict[str, str] = {
    "origin_vs_para": "clean",
    "paratarget_original_push": "clean",
    "same_content_diff_style": "clean",
    "generated_to_target": "clean",
    "target_hard_negative": "clean",
    "trait_positive_vs_negative": "clean",
    "trait_cross_feature_hard": "clean",
    "same_speaker_topic_matched": "topic",
    "bookraw_same_author_cross_work": "topic",
    "same_speaker_real": "confounded",
    "bookraw_same_work": "confounded",
    "bookraw_same_language_author_negative": "confounded",
}

MODES = ("staged", "anti", "mixed", "uniform")
PACINGS = ("step", "linear")
TIERS = ("easy", "mid", "hard")
CLEANLINESS = ("clean", "topic", "confounded")
DROP = "drop"
# Schedules are tabulated over this many progress buckets; mixed averages the same buckets staged uses,
# so the two arms' expected exposures match exactly.
RESOLUTION = 1000

DEFAULT_STAGES: list[dict[str, Any]] = [
    {"until": 0.35, "tiers": {"easy": 0.7, "mid": 0.25, "hard": 0.05}},
    {"until": 0.70, "tiers": {"easy": 0.35, "mid": 0.40, "hard": 0.25}},
    {"until": 1.00, "tiers": {"easy": 0.20, "mid": 0.30, "hard": 0.50}},
]

CLEAN_STAGES: list[dict[str, Any]] = [
    {"until": 0.20, "tiers": {"clean": 1.0}},
    {"until": 0.60, "tiers": {"clean": 0.6, "topic": 0.4}},
    {"until": 1.00, "tiers": {"clean": 0.45, "topic": 0.45, "confounded": 0.10}},
]

# grouping -> (tier names, default stages, rule table, tier of length_course_*, tier of an unlisted rule)
GROUPINGS: dict[str, tuple[tuple[str, ...], list[dict[str, Any]], dict[str, str], str, str]] = {
    "difficulty": (TIERS, DEFAULT_STAGES, RULE_TIERS, "easy", "mid"),
    "cleanliness": (CLEANLINESS, CLEAN_STAGES, RULE_CLEANLINESS, "confounded", "confounded"),
}


def curriculum_grouping(config: dict[str, Any]) -> str:
    grouping = str((config.get("curriculum", {}) or {}).get("grouping", "difficulty"))
    if grouping not in GROUPINGS:
        raise ValueError(f"curriculum.grouping must be one of {tuple(GROUPINGS)}, got {grouping}")
    return grouping


def default_stages(grouping: str = "difficulty") -> list[dict[str, Any]]:
    return GROUPINGS[grouping][1]


def rule_tier(rule: str, overrides: dict[str, str] | None = None, grouping: str = "difficulty") -> str:
    if overrides and rule in overrides:
        return overrides[rule]
    _, _, table, length_course, unlisted = GROUPINGS[grouping]
    if rule.startswith("length_course_"):
        return length_course
    return table.get(rule, unlisted)


def curriculum_stages(config: dict[str, Any]) -> list[dict[str, Any]]:
    grouping = curriculum_grouping(config)
    names = GROUPINGS[grouping][0]
    stages = config.get("curriculum", {}).get("stages") or default_stages(grouping)
    previous = 0.0
    for stage in stages:
        if not previous < float(stage["until"]) <= 1.0:
            raise ValueError(f"curriculum stage boundaries must increase within (0, 1]: {stages}")
        unknown = set(stage["tiers"]) - set(names)
        if unknown:
            raise ValueError(f"curriculum stage names unknown tiers {sorted(unknown)}; {grouping} uses {names} "
                             "(a base config's stages are kept when only curriculum.grouping changes)")
        if any(float(v) < 0 for v in stage["tiers"].values()):
            raise ValueError(f"curriculum stage has a negative weight: {stage}")
        if sum(float(v) for v in stage["tiers"].values()) <= 0:
            raise ValueError(f"curriculum stage has no mass: {stage}")
        previous = float(stage["until"])
    if previous != 1.0:
        raise ValueError("last curriculum stage must end at 1.0")
    return stages


def curriculum_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("curriculum", {}) or {}
    mode = str(cfg.get("mode", "staged"))
    if mode not in MODES:
        raise ValueError(f"curriculum.mode must be one of {MODES}, got {mode}")
    pacing = str(cfg.get("pacing", "step"))
    if pacing not in PACINGS:
        raise ValueError(f"curriculum.pacing must be one of {PACINGS}, got {pacing}")
    grouping = curriculum_grouping(config)
    names = GROUPINGS[grouping][0]
    tiers = {str(rule): str(tier) for rule, tier in (cfg.get("tiers") or {}).items()}
    bad = {rule: tier for rule, tier in tiers.items() if tier not in (*names, DROP)}
    if bad:
        raise ValueError(f"curriculum.tiers values must be one of {(*names, DROP)} under {grouping}: {bad}")
    shares = {str(rule): float(value) for rule, value in (cfg.get("rule_shares") or {}).items()}
    if any(value <= 0 for value in shares.values()):
        raise ValueError(f"curriculum.rule_shares must be positive (drop a rule with curriculum.tiers): {shares}")
    return {"mode": mode, "pacing": pacing, "grouping": grouping, "stages": curriculum_stages(config),
            "tiers": tiers, "shares": shares}


def stage_at(stages: list[dict[str, Any]], progress: float) -> int:
    for index, stage in enumerate(stages):
        if progress < float(stage["until"]):
            return index
    return len(stages) - 1


def _proportions(weights: dict[str, Any]) -> dict[str, float]:
    mass = sum(float(v) for v in weights.values())
    return {tier: float(v) / mass for tier, v in weights.items()}


def stage_weights(stages: list[dict[str, Any]], progress: float, pacing: str = "step") -> dict[str, float]:
    """Tier proportions of the staged schedule at `progress` in [0, 1)."""
    if pacing == "step":
        return _proportions(stages[stage_at(stages, progress)]["tiers"])
    # linear: each stage's mix is pinned at its midpoint and interpolated in between.
    mids: list[float] = []
    previous = 0.0
    for stage in stages:
        mids.append((previous + float(stage["until"])) / 2)
        previous = float(stage["until"])
    mixes = [_proportions(stage["tiers"]) for stage in stages]
    if progress <= mids[0]:
        return mixes[0]
    if progress >= mids[-1]:
        return mixes[-1]
    i = bisect_right(mids, progress) - 1
    t = (progress - mids[i]) / (mids[i + 1] - mids[i])
    names = sorted(set(mixes[i]) | set(mixes[i + 1]))
    return {n: (1 - t) * mixes[i].get(n, 0.0) + t * mixes[i + 1].get(n, 0.0) for n in names}


def _restrict(weights: dict[str, float], present: list[str]) -> dict[str, float]:
    """Renormalize onto tiers that have data. A stage naming only empty tiers falls back to uniform."""
    kept = {tier: weights.get(tier, 0.0) for tier in present if weights.get(tier, 0.0) > 0.0}
    mass = sum(kept.values())
    if mass <= 0.0:
        return {tier: 1.0 / len(present) for tier in present}
    return {tier: value / mass for tier, value in kept.items()}


def schedule_table(settings: dict[str, Any], present: list[str]) -> list[dict[str, float]]:
    """Tier proportions over the `present` tiers for each of RESOLUTION progress buckets."""
    stages, pacing = settings["stages"], settings["pacing"]
    staged = [_restrict(stage_weights(stages, (b + 0.5) / RESOLUTION, pacing), present) for b in range(RESOLUTION)]
    if settings["mode"] == "anti":
        return staged[::-1]
    if settings["mode"] == "mixed":
        # Average after renormalizing, so an empty tier redistributes identically in staged and mixed.
        mean = {tier: sum(row.get(tier, 0.0) for row in staged) / RESOLUTION for tier in present}
        return [mean] * RESOLUTION
    return staged


def _tier_rules(rules: list[str], overrides: dict[str, str], grouping: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for rule in sorted(rules):
        tier = rule_tier(rule, overrides, grouping)
        if tier != DROP:
            grouped[tier].append(rule)
    return dict(grouped)


def curriculum_order(
    triplets: list[Any],
    config: dict[str, Any],
    seed: int,
    total: int,
) -> list[Any]:
    """Return `total` triplets ordered so position/total maps to the schedule.

    Triplets are drawn without replacement within a rule until the rule is exhausted, then reshuffled.
    """
    settings = curriculum_settings(config)
    rng = random.Random(seed)
    by_rule: dict[str, list[Any]] = defaultdict(list)
    for row in triplets:
        if rule_tier(row.rule, settings["tiers"], settings["grouping"]) != DROP:
            by_rule[row.rule].append(row)
    if not by_rule:
        raise ValueError("curriculum has no rules left to sample (all dropped or no triplets)")
    queues: dict[str, list[Any]] = {rule: [] for rule in by_rule}

    def draw(rule: str) -> Any:
        if not queues[rule]:
            queues[rule] = list(by_rule[rule])
            rng.shuffle(queues[rule])
        return queues[rule].pop()

    shares = settings["shares"]

    def rule_picker(rules: list[str]) -> tuple[list[str], list[float]]:
        return rules, list(accumulate(shares.get(rule, 1.0) for rule in rules))

    if settings["mode"] == "uniform":
        rules, cumulative = rule_picker(sorted(by_rule))
        return [draw(rng.choices(rules, cum_weights=cumulative)[0]) for _ in range(total)]
    tier_rules = _tier_rules(list(by_rule), settings["tiers"], settings["grouping"])
    pickers = {tier: rule_picker(rules) for tier, rules in tier_rules.items()}
    table = schedule_table(settings, sorted(tier_rules))
    choices = [(list(row), list(accumulate(row.values()))) for row in table]
    ordered: list[Any] = []
    for position in range(total):
        tiers, cumulative = choices[position * RESOLUTION // total]
        tier = rng.choices(tiers, cum_weights=cumulative)[0]
        rules, rule_cumulative = pickers[tier]
        ordered.append(draw(rng.choices(rules, cum_weights=rule_cumulative)[0]))
    return ordered


def _rule_rows(
    draws: dict[str, float], available: dict[str, int], overrides: dict[str, str], grouping: str
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for rule in sorted(set(draws) | set(available)):
        count = int(available.get(rule, 0))
        drawn = float(draws.get(rule, 0.0))
        rows[rule] = {
            "tier": rule_tier(rule, overrides, grouping),
            "available": count,
            "draws": round(drawn, 1),
            "repeats": round(drawn / count, 2) if count else None,
        }
    return rows


def expected_exposure(
    available: dict[str, int],
    config: dict[str, Any],
    total: int,
    phases: int = 3,
) -> dict[str, Any]:
    """Expected draws per rule and tier share per phase for a `total`-row budget, without sampling.

    `repeats` = draws / available triplets: how often each distinct triplet is reused.
    """
    settings = curriculum_settings(config)
    present = {rule: n for rule, n in available.items() if n > 0}
    tier_rules = _tier_rules(list(present), settings["tiers"], settings["grouping"])
    if not tier_rules:
        raise ValueError("curriculum has no rules left to sample (all dropped or no triplets)")
    shares = settings["shares"]
    tier_mass = {tier: sum(shares.get(rule, 1.0) for rule in rules) for tier, rules in tier_rules.items()}
    if settings["mode"] == "uniform":
        flat = {tier: mass / sum(tier_mass.values()) for tier, mass in tier_mass.items()}
        table = [flat] * RESOLUTION
    else:
        table = schedule_table(settings, sorted(tier_rules))
    share_by_phase = []
    for phase in range(phases):
        rows = table[phase * RESOLUTION // phases : (phase + 1) * RESOLUTION // phases]
        share_by_phase.append({tier: round(sum(r.get(tier, 0.0) for r in rows) / len(rows), 4) for tier in sorted(tier_rules)})
    mean = {tier: sum(r.get(tier, 0.0) for r in table) / RESOLUTION for tier in tier_rules}
    draws = {
        rule: total * mean[tier] * shares.get(rule, 1.0) / tier_mass[tier]
        for tier, rules in tier_rules.items() for rule in rules
    }
    return {
        "mode": settings["mode"],
        "pacing": settings["pacing"],
        "grouping": settings["grouping"],
        "rows": total,
        "tier_share_by_phase": share_by_phase,
        "rules": _rule_rows(draws, available, settings["tiers"], settings["grouping"]),
    }


def schedule_summary(
    ordered: list[Any],
    config: dict[str, Any],
    available: dict[str, int],
    phases: int = 10,
) -> dict[str, Any]:
    """What a run actually sampled: tier share per phase and draws/repeats per rule."""
    settings = curriculum_settings(config)
    total = len(ordered)
    by_phase = [Counter() for _ in range(phases)]
    draws: Counter[str] = Counter()
    for position, row in enumerate(ordered):
        by_phase[position * phases // max(1, total)][rule_tier(row.rule, settings["tiers"], settings["grouping"])] += 1
        draws[row.rule] += 1
    share_by_phase = [
        {tier: round(count / sum(counter.values()), 4) for tier, count in sorted(counter.items())}
        for counter in by_phase if counter
    ]
    return {
        "mode": settings["mode"],
        "pacing": settings["pacing"],
        "grouping": settings["grouping"],
        "rows": total,
        "tier_share_by_phase": share_by_phase,
        "rules": _rule_rows(dict(draws), available, settings["tiers"], settings["grouping"]),
    }
