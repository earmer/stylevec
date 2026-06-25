from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from typing import Any

from config import config_hash
from import_bookraw import load_bookraw_samples
from import_genshin import load_genshin_samples, load_top_speakers
from import_msynthstel import load_msynthstel_samples
from import_paratarget import load_paratarget_samples
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
        "genshin": {"same_speaker_real", "origin_vs_para"},
        "paratarget": {"generated_to_target", "same_content_diff_style", "target_hard_negative"},
        "msynthstel": {"trait_positive_vs_negative", "trait_cross_feature_hard"},
    }
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
    weights = config["weights"]
    margins = config["margins"]
    grouped = by_style(samples, split, "genshin")
    speakers = [speaker for speaker, rows in grouped.items() if len(rows) >= 2]
    if len(speakers) < 2:
        return []
    triplets: list[TripletRecord] = []
    for speaker_index, speaker in enumerate(speakers):
        rows = grouped[speaker]
        negative_speaker = speakers[(speaker_index + 1) % len(speakers)]
        negative_rows = grouped[negative_speaker]
        for index, anchor in enumerate(rows):
            positive = rows[(index + 1) % len(rows)]
            negative = negative_rows[index % len(negative_rows)]
            add_triplet(
                triplets,
                anchor.text,
                positive.text,
                negative.text,
                "same_speaker_real",
                split,
                float(weights["genshin_real"]),
                float(margins["medium"]),
                {"speaker": speaker, "negative_speaker": negative_speaker},
            )
            para_text = anchor.meta.get("para_text")
            if isinstance(para_text, str) and para_text:
                add_triplet(
                    triplets,
                    anchor.text,
                    positive.text,
                    para_text,
                    "origin_vs_para",
                    split,
                    float(weights["genshin_real"]),
                    float(margins["hard"]),
                    {"speaker": speaker},
                )
    return triplets


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
        add_triplet(
            triplets,
            generated.text,
            positive.text,
            negative_source.text,
            "generated_to_target",
            split,
            float(weights["paratarget"]),
            float(margins["hard"]),
            {"target_speaker": target, "source_speaker": source},
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
                },
            )
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
            {"target_speaker": target, "source_speaker": source},
        )
        targets = [name for name in generated_by_target if name != target and generated_by_target[name]]
        if targets:
            other_target = targets[index % len(targets)]
            other_negative = generated_by_target[other_target][0]
            add_triplet(
                triplets,
                generated.text,
                positive.text,
                other_negative.text,
                "target_hard_negative",
                split,
                float(weights["paratarget"]),
                float(margins["hard"]),
                {"target_speaker": target, "negative_target": other_target},
            )
    return triplets


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
    triplets: list[TripletRecord] = []
    for feature_index, feature in enumerate(features):
        rows = positives[feature]
        bad_rows = negatives[feature]
        for index, anchor in enumerate(rows):
            positive = rows[(index + 1) % len(rows)]
            negative = bad_rows[index % len(bad_rows)]
            add_triplet(
                triplets,
                anchor.text,
                positive.text,
                negative.text,
                "trait_positive_vs_negative",
                split,
                float(weights["msynthstel"]),
                float(margins["medium"]),
                {"feature": feature},
            )
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
    counts: defaultdict[str, int] = defaultdict(int)
    triplets: list[TripletRecord] = []
    for domain, source in sorted(grouped):
        rows = grouped[(domain, source)]
        if len(rows) < 2 or domain not in negative_index:
            continue
        rows = sorted(rows, key=lambda row: (sample_length(row), row.sample_id))
        for index, anchor in enumerate(rows):
            if counts[domain] >= limit:
                break
            positive_text = combine_same_source_texts(rows, index, max_parts, max_chars)
            if positive_text is None:
                continue
            negative = pick_similar_length_negative(
                negative_index[domain],
                source,
                len(positive_text),
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
                    "negative_source": negative.style_id,
                    "anchor_length": len(anchor.text),
                    "positive_length": len(positive_text),
                    "negative_length": len(negative.text),
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
) -> str | None:
    parts: list[str] = []
    for offset in range(min(max_parts, len(rows))):
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
    sorted_rows = sorted(rows, key=lambda row: (sample_length(row), row.style_id, row.sample_id))
    return {
        "rows": sorted_rows,
        "lengths": [sample_length(row) for row in sorted_rows],
    }


def pick_similar_length_negative(
    index: dict[str, Any],
    source: str,
    target_length: int,
) -> SampleRecord | None:
    rows: list[SampleRecord] = index["rows"]
    lengths: list[int] = index["lengths"]
    center = bisect_left(lengths, target_length)
    for radius in range(len(rows) + 1):
        probes = (center,) if radius == 0 else (center - radius, center + radius)
        for probe in probes:
            if not 0 <= probe < len(rows):
                continue
            candidate = rows[probe]
            if candidate.style_id != source:
                return candidate
    return None


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
