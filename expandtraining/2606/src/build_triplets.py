from __future__ import annotations

import random
from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Any

from config import config_hash
from import_bookraw import load_bookraw_samples
from import_genshin import hub_speakers, load_genshin_samples, load_top_speakers
from import_msynthstel import load_msynthstel_samples, msynthstel_mode
from import_paratarget import load_paratarget_samples
from similarity import char_ngrams, jaccard
from records import SampleRecord, TripletRecord, normalize_text, stable_id


def build_training_triplets(config: dict[str, Any]) -> tuple[list[TripletRecord], dict[str, Any]]:
    speakers = load_top_speakers(config)
    genshin_samples, genshin_filters = load_genshin_samples(config, speakers)
    paratarget_samples, paratarget_filters = load_paratarget_samples(config, speakers)
    msynthstel_samples, msynthstel_filters = load_msynthstel_samples(config)
    bookraw_samples, bookraw_filters = load_bookraw_samples(config)

    triplets: list[TripletRecord] = []
    triplets.extend(build_genshin_triplets(config, genshin_samples, "train"))
    triplets.extend(build_genshin_triplets(config, genshin_samples, "val"))
    triplets.extend(build_paratarget_triplets(config, genshin_samples, paratarget_samples, "train"))
    triplets.extend(build_paratarget_triplets(config, genshin_samples, paratarget_samples, "val"))
    triplets.extend(build_msynthstel_triplets(config, msynthstel_samples, "train"))
    triplets.extend(build_msynthstel_triplets(config, msynthstel_samples, "val"))
    triplets.extend(build_bookraw_triplets(config, bookraw_samples, "train"))
    triplets.extend(build_bookraw_triplets(config, bookraw_samples, "val"))
    samples = genshin_samples + paratarget_samples + msynthstel_samples + bookraw_samples
    triplets.extend(build_length_course_triplets(config, samples, "train"))
    triplets.extend(build_length_course_triplets(config, samples, "val"))

    triplets = cap_triplets(config, triplets)
    require_rule_coverage(config, triplets)
    if not any(row.split == "train" for row in triplets):
        raise ValueError("triplet builder produced no train triplets")
    if not any(row.split == "val" for row in triplets):
        raise ValueError("triplet builder produced no val triplets")

    counts = Counter()
    for sample in samples:
        counts[f"samples/{sample.domain}/{sample.split}"] += 1
    for triplet in triplets:
        counts[f"triplets/{triplet.rule}/{triplet.split}"] += 1
        counts[f"triplets/all/{triplet.split}"] += 1
    manifest = {
        "manifest_hash": stable_id(config_hash(config), counts, size=24),
        "speakers": speakers,
        "counts": dict(counts),
        "filters": {
            "genshin": genshin_filters,
            "paratarget": paratarget_filters,
            "msynthstel": msynthstel_filters,
            "bookraw": bookraw_filters,
        },
    }
    return triplets, manifest


def require_rule_coverage(config: dict[str, Any], triplets: list[TripletRecord]) -> None:
    required = {
        "genshin": {"same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"},
        "paratarget": {"generated_to_target", "target_hard_negative"},
        "msynthstel": {"trait_positive_vs_negative", "trait_cross_feature_hard"},
    }
    if msynthstel_mode(config) == "pairs":
        required["msynthstel"].discard("trait_cross_feature_hard")
    if bool(config["data"].get("same_content_diff_style_enabled", True)):
        required["paratarget"].add("same_content_diff_style")
    if bool(config["data"].get("bookraw_enabled", False)):
        required["bookraw"] = {
            "bookraw_same_work",
            "bookraw_same_author_cross_work",
            "bookraw_same_language_author_negative",
        }
    present_rules = {row.rule for row in triplets}
    missing = {
        source: sorted(rules.difference(present_rules))
        for source, rules in required.items()
        if rules.difference(present_rules)
    }
    if missing:
        raise ValueError(f"missing required triplets after caps: {missing}")


def by_style(samples: list[SampleRecord], split: str, domain: str | None = None) -> dict[str, list[SampleRecord]]:
    grouped: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        if sample.split != split:
            continue
        if domain is not None and sample.domain != domain:
            continue
        grouped[sample.style_id].append(sample)
    return dict(grouped)


def add_triplet(
    rows: list[TripletRecord],
    anchor: str,
    positive: str,
    negative: str,
    rule: str,
    split: str,
    weight: float,
    margin: float,
    meta: dict[str, Any] | None = None,
) -> None:
    rows.append(
        TripletRecord(
            anchor_text=anchor,
            positive_text=positive,
            negative_text=negative,
            rule=rule,
            weight=weight,
            margin=margin,
            split=split,
            meta=meta or {},
        )
    )


def build_genshin_triplets(config: dict[str, Any], samples: list[SampleRecord], split: str) -> list[TripletRecord]:
    """Same-speaker triplets that cannot be solved by scene/topic.

    same_speaker_real           positive: same speaker, different scene. negative: length-matched random speaker.
    same_speaker_topic_matched  positive: same speaker, different scene, least lexical overlap.
                                negative: another speaker's line from the SAME scene as the anchor.
    origin_vs_para              positive: same speaker, different scene. negative: neutral paraphrase of the anchor.
    """
    weights = config["weights"]
    margins = config["margins"]
    min_gap = int(config["data"].get("genshin_min_id_gap", 150))
    candidates = int(config["data"].get("genshin_candidates", 24))
    # >1 widens the hard pool instead of replaying it: each train anchor gets this many topic-matched
    # triplets, with its k least-overlapping far positives and k distinct same-scene negatives. Validation
    # keeps one per anchor, so evaluation does not change with it. Raise data.max_triplets to keep them.
    variants = int(config["data"].get("genshin_topic_matched_variants", 1))
    if variants < 1:
        raise ValueError("data.genshin_topic_matched_variants must be >= 1")
    hubs = hub_speakers(config)
    grouped = by_style(samples, split, "genshin")
    speakers = [speaker for speaker, rows in grouped.items() if len(rows) >= 2]
    if len(speakers) < 2:
        return []
    grams = {row.sample_id: char_ngrams(row.text) for rows in grouped.values() for row in rows}
    pool = [(speaker, row) for speaker in speakers for row in grouped[speaker]]
    triplets: list[TripletRecord] = []
    for speaker in speakers:
        rows = grouped[speaker]
        for anchor in rows:
            rng = random.Random(f"genshin:{anchor.sample_id}")
            far = far_rows(anchor, rows, min_gap)
            if not far:
                continue
            anchor_grams = grams[anchor.sample_id]
            sampled_far = rng.sample(far, min(candidates, len(far)))
            # Length-matched, as the negatives are (a length-matched line; the paraphrase): with a random
            # positive the negative was closer in length to the anchor in 64/64 held-out same_speaker_real
            # and 92% of origin_vs_para triplets, so any length signal counted against the rule.
            real_positive = min(sampled_far, key=lambda row: abs(sample_length(row) - sample_length(anchor)))
            topic_positive = min(sampled_far, key=lambda row: jaccard(anchor_grams, grams[row.sample_id]))
            others = [
                row
                for other, row in rng.sample(pool, min(candidates * 4, len(pool)))
                if other != speaker and length_compatible(anchor, row)
            ]
            meta = {
                "speaker": speaker,
                "heldout": bool(anchor.meta.get("heldout", False)),
                "anchor_db_id": int(anchor.meta.get("db_id", -1)),
                **twin_meta(anchor.text, anchor.meta.get("para_text")),
                **({"talk_id": int(anchor.meta["talk_id"])} if "talk_id" in anchor.meta else {}),
            }
            if others:
                easy_negative = min(others[:candidates], key=lambda row: abs(sample_length(row) - sample_length(anchor)))
                add_triplet(
                    triplets, anchor.text, real_positive.text, easy_negative.text,
                    "same_speaker_real", split, float(weights["genshin_real"]), float(margins["medium"]),
                    {
                        **meta,
                        "negative_speaker": easy_negative.style_id,
                        "positive_db_id": int(real_positive.meta.get("db_id", -1)),
                        "negative_db_id": int(easy_negative.meta.get("db_id", -1)),
                    },
                )
            scene = [n for n in anchor.meta.get("scene_neighbors", []) if 0.4 <= len(n[1]) / max(1, len(anchor.text)) <= 2.5]
            # A non-hub speaker (not Paimon/Traveler) when the scene has one; hubs only as a fallback.
            scene = [n for n in scene if n[0] not in hubs] or scene
            if scene:
                first = scene[rng.randrange(len(scene))]
                pairs = [(topic_positive, first)]
                if variants > 1 and split == "train":
                    pairs = topic_matched_pairs(anchor, sampled_far, scene, first, grams, variants)
                for variant, (positive, neighbor) in enumerate(pairs):
                    add_triplet(
                        triplets, anchor.text, positive.text, neighbor[1],
                        "same_speaker_topic_matched", split,
                        float(weights.get("genshin_topic_matched", weights["genshin_real"])), float(margins["hard"]),
                        {
                            **meta,
                            "negative_speaker": neighbor[0],
                            "negative_id_distance": neighbor[2],
                            "positive_id_distance": abs(int(positive.meta["db_id"]) - int(anchor.meta["db_id"])),
                            "positive_db_id": int(positive.meta["db_id"]),
                            **({"variant": variant} if variant else {}),
                        },
                    )
            para_text = anchor.meta.get("para_text")
            if "twin_text" in meta:
                add_triplet(
                    triplets, anchor.text, real_positive.text, para_text,
                    "origin_vs_para", split, float(weights["genshin_real"]), float(margins["hard"]),
                    {**meta, "positive_db_id": int(real_positive.meta.get("db_id", -1))},
                )
    return triplets


def twin_meta(anchor_text: str, twin: Any) -> dict[str, str]:
    """{"twin_text": ...} when the anchor has a content twin (same content, another voice: a Genshin line's
    paraphrase, a rewrite's source line) that differs from it. train.infonce_twin_temperature adds each
    row's twin to its InfoNCE candidates; nothing else reads it."""
    if isinstance(twin, str) and twin.strip() and twin.strip() != anchor_text.strip():
        return {"twin_text": twin}
    return {}


# origin_vs_para needs a paraphrase that differs from the origin: an identical one cannot satisfy the
# margin (10.4% of d-350af1498f6a's train triplets; ~half of this rule's late violations in
# curriculum_a_1gpu). Small edits (呢/会 dropped, ！ -> …) are kept: they are the style signal.

def topic_matched_pairs(
    anchor: SampleRecord,
    sampled_far: list[SampleRecord],
    scene: list[Any],
    first: Any,
    grams: dict[str, Any],
    variants: int,
) -> list[tuple[SampleRecord, Any]]:
    """Up to `variants` (positive, scene neighbour) pairs: positives by rising word overlap with the anchor,
    neighbours drawn without replacement after `first`. Pair 0 is the single-variant triplet, unchanged."""
    anchor_grams = grams[anchor.sample_id]
    positives = sorted(sampled_far, key=lambda row: jaccard(anchor_grams, grams[row.sample_id]))
    rest = [neighbor for neighbor in scene if neighbor is not first]
    random.Random(f"genshin_variants:{anchor.sample_id}").shuffle(rest)
    neighbors = [first, *rest]
    count = min(variants, len(positives), len(neighbors))
    return [(positives[i], neighbors[i]) for i in range(count)]


def far_rows(anchor: SampleRecord, rows: list[SampleRecord], min_gap: int) -> list[SampleRecord]:
    """Same-speaker rows at least `min_gap` DB ids away (different scene); else the farthest available.
    Rows that know their conversation (genshin v2's talk_id) use it instead: any other conversation."""
    if "talk_id" in anchor.meta:
        others = [row for row in rows if row.sample_id != anchor.sample_id]
        far = [row for row in others if row.meta.get("talk_id") != anchor.meta["talk_id"]]
        return far
    key = "db_id" if "db_id" in anchor.meta else "order"
    here = int(anchor.meta.get(key, 0))
    others = [row for row in rows if row.sample_id != anchor.sample_id]
    far = [row for row in others if abs(int(row.meta.get(key, 0)) - here) >= min_gap]
    if far or not others:
        return far
    farthest = max(abs(int(row.meta.get(key, 0)) - here) for row in others)
    return [row for row in others if abs(int(row.meta.get(key, 0)) - here) == farthest]


def length_compatible(anchor: SampleRecord, other: SampleRecord, ratio: float = 2.0) -> bool:
    a, b = max(1, sample_length(anchor)), max(1, sample_length(other))
    return max(a, b) / min(a, b) <= ratio


def build_paratarget_triplets(
    config: dict[str, Any],
    genshin_samples: list[SampleRecord],
    paratarget_samples: list[SampleRecord],
    split: str,
) -> list[TripletRecord]:
    weights = config["weights"]
    margins = config["margins"]
    real_by_speaker = by_style(genshin_samples, split, "genshin")
    generated_by_target = by_style(paratarget_samples, split, "paratarget")
    samples = [sample for sample in paratarget_samples if sample.split == split]
    if not samples:
        return []
    triplets: list[TripletRecord] = []
    for index, generated in enumerate(samples):
        target = generated.style_id
        source = str(generated.meta["source_speaker"])
        target_refs = real_by_speaker.get(target, [])
        source_refs = real_by_speaker.get(source, [])
        if not target_refs or not source_refs:
            continue
        positive = target_refs[index % len(target_refs)]
        negative_source = source_refs[index % len(source_refs)]
        source_text = str(generated.meta["source_text"])
        # Jobs touching a held-out speaker are val-only; marking them lets validation report them apart.
        heldout = {"heldout": bool(generated.meta.get("heldout", False))}
        twin = twin_meta(generated.text, source_text)
        add_triplet(
            triplets,
            generated.text,
            positive.text,
            negative_source.text,
            "generated_to_target",
            split,
            float(weights["paratarget"]),
            float(margins["hard"]),
            {"target_speaker": target, "source_speaker": source, **heldout, **twin},
        )
        if bool(config["data"].get("paratarget_original_push_enabled", False)):
            add_triplet(
                triplets,
                generated.text,
                positive.text,
                source_text,
                "paratarget_original_push",
                split,
                float(weights.get("paratarget_original_push", float(weights["paratarget"]) * 0.12)),
                float(margins["easy"]),
                {
                    "target_speaker": target,
                    "source_speaker": source,
                    "source_length": len(source_text),
                    "generated_length": len(generated.text),
                    **heldout,
                    **twin,
                },
            )
        if bool(config["data"].get("same_content_diff_style_enabled", True)):
            source_positive = source_refs[(index + 1) % len(source_refs)]
            add_triplet(
                triplets,
                source_text,
                source_positive.text,
                generated.text,
                "same_content_diff_style",
                split,
                float(weights["paratarget"]),
                float(margins["adversarial"]),
                {"target_speaker": target, "source_speaker": source, **heldout,
                 **twin_meta(source_text, generated.text)},
            )
        targets = [name for name in generated_by_target if name != target and generated_by_target[name]]
        if targets:
            other_target = targets[index % len(targets)]
            other_negative = pick_other_target_negative(
                generated_by_target[other_target], index // len(targets), (generated.text, positive.text)
            )
            if other_negative is not None:
                add_triplet(
                    triplets,
                    generated.text,
                    positive.text,
                    other_negative.text,
                    "target_hard_negative",
                    split,
                    float(weights["paratarget"]),
                    float(margins["hard"]),
                    {"target_speaker": target, "negative_target": other_target, **heldout, **twin},
                )
    return triplets


def pick_other_target_negative(
    pool: list[SampleRecord],
    turn: int,
    avoid: tuple[str, ...],
) -> SampleRecord | None:
    """The `turn`-th visit to this other target takes its `turn`-th line (cycling), so negatives spread
    over the whole pool instead of reusing pool[0]. Deterministic, no RNG. Advances past a line whose
    text equals the anchor or positive; None only if every line does."""
    for step in range(len(pool)):
        candidate = pool[(turn + step) % len(pool)]
        if candidate.text not in avoid:
            return candidate
    return None


def build_msynthstel_triplets(
    config: dict[str, Any],
    samples: list[SampleRecord],
    split: str,
) -> list[TripletRecord]:
    weights = config["weights"]
    margins = config["margins"]
    positives: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    negatives: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        if sample.split != split or sample.domain != "msynthstel":
            continue
        if sample.meta.get("polarity") == "positive":
            positives[sample.style_id].append(sample)
        elif sample.meta.get("polarity") == "negative":
            negatives[sample.style_id].append(sample)
    features = [feature for feature, rows in positives.items() if len(rows) >= 2 and negatives.get(feature)]
    if not features:
        return []
    pairs = msynthstel_mode(config) == "pairs"
    triplets: list[TripletRecord] = []
    for feature_index, feature in enumerate(features):
        rows = positives[feature]
        bad_rows = negatives[feature]
        for index, anchor in enumerate(rows):
            positive = rows[(index + 1) % len(rows)]
            # The importer appends each row's two poles together, so index i is anchor i's own pair.
            negative = bad_rows[index % len(bad_rows)]
            meta: dict[str, Any] = {"feature": feature}
            if pairs:
                if negative.content_id != anchor.content_id:
                    raise ValueError(f"msynthstel pair {anchor.sample_id} lost its other pole")
                meta["pole"] = anchor.meta["raw_feature"]
            add_triplet(
                triplets,
                anchor.text,
                positive.text,
                negative.text,
                "trait_positive_vs_negative",
                split,
                float(weights["msynthstel"]),
                float(margins["medium"]),
                meta,
            )
            if pairs:
                continue
            other_feature = features[(feature_index + 1) % len(features)]
            if other_feature != feature:
                other_negative = positives[other_feature][index % len(positives[other_feature])]
                add_triplet(
                    triplets,
                    anchor.text,
                    positive.text,
                    other_negative.text,
                    "trait_cross_feature_hard",
                    split,
                    float(weights["msynthstel"]),
                    float(margins["hard"]),
                    {"feature": feature, "negative_feature": other_feature},
                )
    return triplets


def build_length_course_triplets(
    config: dict[str, Any],
    samples: list[SampleRecord],
    split: str,
) -> list[TripletRecord]:
    if not bool(config["data"].get("length_course_enabled", False)):
        return []
    weights = config["weights"]
    margins = config["margins"]
    max_parts = int(config["data"].get("length_course_parts", 2))
    max_chars = int(config["data"].get("length_course_max_chars", 6000))
    limit = length_course_domain_limit(config, split)
    # anchor (default): the negative is as short as the anchor while the positive is long, so encoding
    # length favours the wrong answer and only style can satisfy the margin. positive: the pre-2026-10-18
    # rule, where P and N are both long and length is neutral, so it never penalised encoding length.
    negative_length = str(config["data"].get("length_course_negative_length", "anchor"))
    if negative_length not in ("anchor", "positive"):
        raise ValueError(f"data.length_course_negative_length must be anchor or positive, got {negative_length!r}")
    match_anchor = negative_length == "anchor"
    # The default positive starts with the anchor itself, so it shares the anchor's text (char-bigram
    # Jaccard 0.48 Genshin / 0.80 msynthstel): InfoNCE keeps rewarding that overlap long after the hinge
    # saturates. True builds it from the source's next lines only.
    skip_anchor = bool(config["data"].get("length_course_positive_excludes_anchor", False))

    grouped: defaultdict[tuple[str, str], list[SampleRecord]] = defaultdict(list)
    domain_rows: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        key = length_course_key(sample, split)
        if key is None:
            continue
        grouped[key].append(sample)
        domain_rows[sample.domain].append(sample)

    negative_index = {
        domain: build_length_negative_index(rows)
        for domain, rows in domain_rows.items()
        if len({row.style_id for row in rows}) >= 2
    }
    # Anchors round-robin over sources (each source's lines in a fixed shuffled order): taking sources in
    # sorted order filled the domain limit from the first 5 Genshin speakers.
    sources: list[tuple[str, str, list[SampleRecord], list[int]]] = []
    for domain, source in sorted(grouped):
        rows = grouped[(domain, source)]
        if len(rows) < 2 or domain not in negative_index:
            continue
        rows = sorted(rows, key=lambda row: (sample_length(row), row.sample_id))
        visit = list(range(len(rows)))
        random.Random(f"length_course:{domain}:{source}").shuffle(visit)
        sources.append((domain, source, rows, visit))
    counts: defaultdict[str, int] = defaultdict(int)
    triplets: list[TripletRecord] = []
    for turn in range(max((len(rows) for _d, _s, rows, _v in sources), default=0)):
        for domain, source, rows, visit in sources:
            if turn >= len(rows) or counts[domain] >= limit:
                continue
            index = visit[turn]
            anchor = rows[index]
            positive_text = combine_same_source_texts(rows, index, max_parts, max_chars, skip_anchor)
            if positive_text is None:
                continue
            negative = pick_similar_length_negative(
                negative_index[domain],
                source,
                len(anchor.text) if match_anchor else len(positive_text),
                counts[domain],
            )
            if negative is None:
                continue
            add_triplet(
                triplets,
                anchor.text,
                positive_text,
                negative.text,
                f"length_course_{domain}",
                split,
                float(weights.get("length_course", 0.08)),
                float(margins["easy"]),
                {
                    "domain": domain,
                    "source": source,
                    "heldout": bool(anchor.meta.get("heldout", False)),
                    "negative_source": negative.style_id,
                    "anchor_length": len(anchor.text),
                    "positive_length": len(positive_text),
                    "negative_length": len(negative.text),
                    **twin_meta(anchor.text, anchor.meta.get("para_text")),
                },
            )
            counts[domain] += 1
    return triplets


def length_course_key(sample: SampleRecord, split: str) -> tuple[str, str] | None:
    if sample.split != split:
        return None
    if sample.domain == "genshin":
        return sample.domain, sample.style_id
    if sample.domain == "msynthstel" and sample.meta.get("polarity") == "positive":
        return sample.domain, sample.style_id
    if sample.domain == "bookraw":
        return sample.domain, sample.style_id
    return None


def length_course_domain_limit(config: dict[str, Any], split: str) -> int:
    key = "length_course_triplets_per_domain"
    if split == "train":
        default = max(1, int(int(config["data"]["max_triplets"]) * 0.02))
        return int(config["data"].get(key, default))
    default = max(1, int(int(config["validation"].get("max_triplets", 100)) * 0.02))
    return int(config["validation"].get(key, default))


def combine_same_source_texts(
    rows: list[SampleRecord],
    index: int,
    max_parts: int,
    max_chars: int,
    skip_anchor: bool = False,
) -> str | None:
    parts: list[str] = []
    first = 1 if skip_anchor else 0
    for offset in range(first, min(max_parts + first, len(rows))):
        candidate = rows[(index + offset) % len(rows)].text
        combined = normalize_text(" ".join(parts + [candidate]))
        if parts and len(combined) > max_chars:
            break
        parts.append(candidate)
    if len(parts) < 2:
        return None
    combined = normalize_text(" ".join(parts))
    if len(combined) <= len(rows[index].text):
        return None
    return combined


def build_length_negative_index(rows: list[SampleRecord]) -> dict[str, Any]:
    # Ties in length by a hash, not by source name: name order put the same few sources nearest every length.
    sorted_rows = sorted(rows, key=lambda row: (sample_length(row), stable_id("length_negative", row.sample_id)))
    return {
        "rows": sorted_rows,
        "lengths": [sample_length(row) for row in sorted_rows],
    }


LENGTH_NEGATIVE_POOL = 16


def length_tolerance(length: int) -> int:
    return max(2, length // 20)


def pick_similar_length_negative(
    index: dict[str, Any],
    source: str,
    target_length: int,
    turn: int = 0,
    pool: int = LENGTH_NEGATIVE_POOL,
) -> SampleRecord | None:
    """The `turn`-th (cycling) of up to `pool` other-source rows nearest in length, all within
    length_tolerance of the nearest one. Always taking the nearest gave the same row per length: rows
    tie-break by source name, so 77% of Genshin negatives were one speaker."""
    rows: list[SampleRecord] = index["rows"]
    lengths: list[int] = index["lengths"]
    center = bisect_left(lengths, target_length)
    tolerance = length_tolerance(target_length)
    nearest: list[SampleRecord] = []
    best: int | None = None
    for radius in range(len(rows) + 1):
        probes = (center,) if radius == 0 else (center - radius, center + radius)
        in_window = False  # lengths are sorted: once both probes leave the window, every later one has too
        for probe in probes:
            if not 0 <= probe < len(rows):
                continue
            distance = abs(lengths[probe] - target_length)
            if best is not None and distance > best + tolerance:
                continue
            in_window = True
            if rows[probe].style_id == source:
                continue
            if best is None:
                best = distance
            nearest.append(rows[probe])
        if len(nearest) >= pool or not in_window:
            break
    if not nearest:
        return None
    return nearest[turn % len(nearest)]


def sample_length(sample: SampleRecord) -> int:
    value = sample.meta.get("length")
    if isinstance(value, int):
        return value
    return len(sample.text)


def build_bookraw_triplets(
    config: dict[str, Any],
    samples: list[SampleRecord],
    split: str,
) -> list[TripletRecord]:
    if not samples:
        return []
    weights = config["weights"]
    margins = config["margins"]
    by_author = by_style(samples, split, "bookraw")
    authors = [author for author, rows in by_author.items() if len(rows) >= 2]
    if len(authors) < 2:
        return []

    rows_by_work: dict[str, dict[str, list[SampleRecord]]] = {}
    candidates = [sample for sample in samples if sample.split == split and sample.domain == "bookraw"]
    negative_index = build_bookraw_negative_index(candidates)
    for author in authors:
        works: defaultdict[str, list[SampleRecord]] = defaultdict(list)
        for sample in by_author[author]:
            works[str(sample.meta["work"])].append(sample)
        rows_by_work[author] = {work: rows for work, rows in works.items() if len(rows) >= 2}

    by_rule_rows: dict[str, list[TripletRecord]] = {
        "bookraw_same_work": [],
        "bookraw_same_author_cross_work": [],
        "bookraw_same_language_author_negative": [],
    }

    for author_index, author in enumerate(authors):
        works = rows_by_work.get(author, {})
        if not works:
            continue
        work_names = sorted(works)
        for work in work_names:
            rows = works[work]
            for index, anchor in enumerate(rows):
                positive = rows[(index + 1) % len(rows)]
                negative = bookraw_negative(anchor, negative_index, author, author_index + index)
                if negative is None:
                    continue
                add_triplet(
                    by_rule_rows["bookraw_same_work"],
                    anchor.text,
                    positive.text,
                    negative.text,
                    "bookraw_same_work",
                    split,
                    float(weights["bookraw_same_work"]),
                    float(margins["medium"]),
                    {"author": author, "work": work, "negative_author": negative.style_id},
                )

        if len(work_names) >= 2:
            for work_index, work in enumerate(work_names):
                rows = works[work]
                positive_rows = works[work_names[(work_index + 1) % len(work_names)]]
                for index, anchor in enumerate(rows):
                    positive = positive_rows[index % len(positive_rows)]
                    negative = bookraw_negative(anchor, negative_index, author, author_index + index)
                    if negative is None:
                        continue
                    add_triplet(
                        by_rule_rows["bookraw_same_author_cross_work"],
                        anchor.text,
                        positive.text,
                        negative.text,
                        "bookraw_same_author_cross_work",
                        split,
                        float(weights["bookraw_cross_work"]),
                        float(margins["hard"]),
                        {
                            "author": author,
                            "work": work,
                            "positive_work": positive.meta["work"],
                            "negative_author": negative.style_id,
                        },
                    )

        author_rows = by_author[author]
        for index, anchor in enumerate(author_rows):
            positive = bookraw_author_positive(anchor, rows_by_work[author], index)
            negative = bookraw_negative(anchor, negative_index, author, author_index + index)
            if positive is None or negative is None:
                continue
            add_triplet(
                by_rule_rows["bookraw_same_language_author_negative"],
                anchor.text,
                positive.text,
                negative.text,
                "bookraw_same_language_author_negative",
                split,
                float(weights["bookraw_author_negative"]),
                float(margins["hard"]),
                {
                    "author": author,
                    "work": anchor.meta["work"],
                    "positive_work": positive.meta["work"],
                    "negative_author": negative.style_id,
                },
            )

    limit = bookraw_rule_limit(config, split)
    triplets: list[TripletRecord] = []
    for rule in sorted(by_rule_rows):
        triplets.extend(by_rule_rows[rule][:limit])
    return triplets


def bookraw_author_positive(
    anchor: SampleRecord,
    works: dict[str, list[SampleRecord]],
    index: int,
) -> SampleRecord | None:
    work = str(anchor.meta["work"])
    other_work_names = [name for name in sorted(works) if name != work and works[name]]
    if other_work_names:
        rows = works[other_work_names[index % len(other_work_names)]]
        return rows[index % len(rows)]
    same_work_rows = works.get(work, [])
    if len(same_work_rows) < 2:
        return None
    for offset in range(1, len(same_work_rows) + 1):
        candidate = same_work_rows[(index + offset) % len(same_work_rows)]
        if candidate.sample_id != anchor.sample_id:
            return candidate
    return None


def build_bookraw_negative_index(
    candidates: list[SampleRecord],
) -> dict[str, Any]:
    by_lang_bucket: defaultdict[tuple[str, Any], list[SampleRecord]] = defaultdict(list)
    by_lang: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for row in candidates:
        lang = str(row.meta.get("lang", row.language))
        bucket = row.meta.get("length_bucket")
        by_lang_bucket[(lang, bucket)].append(row)
        by_lang[lang].append(row)
    return {
        "by_lang_bucket": dict(by_lang_bucket),
        "by_lang": dict(by_lang),
        "all": candidates,
    }


def pick_not_author(pool: list[SampleRecord], author: str, index: int) -> SampleRecord | None:
    if not pool:
        return None
    start = index % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate.style_id != author:
            return candidate
    return None


def bookraw_negative(
    anchor: SampleRecord,
    negative_index: dict[str, Any],
    author: str,
    index: int,
) -> SampleRecord | None:
    lang = str(anchor.meta.get("lang", anchor.language))
    bucket = anchor.meta.get("length_bucket")
    pools = [
        negative_index["by_lang_bucket"].get((lang, bucket), []),
        negative_index["by_lang"].get(lang, []),
        negative_index["all"],
    ]
    for pool in pools:
        negative = pick_not_author(pool, author, index)
        if negative is not None:
            return negative
    return None


def bookraw_rule_limit(config: dict[str, Any], split: str) -> int:
    if split == "train":
        default = max(1, int(int(config["data"]["max_triplets"]) * 0.05))
        return int(config["data"].get("bookraw_triplets_per_rule", default))
    default = max(1, int(int(config["validation"].get("max_triplets", 100)) * 0.05))
    return int(config["validation"].get("bookraw_triplets_per_rule", default))


def cap_triplets(config: dict[str, Any], triplets: list[TripletRecord]) -> list[TripletRecord]:
    max_train = int(config["data"]["max_triplets"])
    max_val = int(config["validation"].get("max_triplets", max(4, max_train // 2)))
    train = [row for row in triplets if row.split == "train"]
    val = [row for row in triplets if row.split == "val"]
    return cap_by_rule(train, max_train) + cap_by_rule(val, max_val)


def cap_by_rule(rows: list[TripletRecord], limit: int) -> list[TripletRecord]:
    by_rule_rows: defaultdict[str, list[TripletRecord]] = defaultdict(list)
    for row in rows:
        by_rule_rows[row.rule].append(row)
    rules = sorted(by_rule_rows)
    # Builders emit rows speaker/author by speaker/author; truncating without a shuffle drops the tail ones
    # (including every held-out speaker). Deterministic per-rule shuffle keeps the cap unbiased.
    for rule in rules:
        random.Random(f"cap:{rule}").shuffle(by_rule_rows[rule])
    capped: list[TripletRecord] = []
    index = 0
    while len(capped) < limit:
        added = False
        for rule in rules:
            rule_rows = by_rule_rows[rule]
            if index < len(rule_rows):
                capped.append(rule_rows[index])
                added = True
                if len(capped) >= limit:
                    break
        if not added:
            break
        index += 1
    return capped
