# Trait Applicability Map v2

This note records the curated style-trait applicability map for the 75 `pipeline/datadelta/deduped` languages from `af` to `zh`.
The full matrix is in `trait_applicability.csv`.

## Scope

- Language universe: 75 languages from `data/datasets/msynthstel/pipeline/datadelta/deduped/*.jsonl`.
- Current cleaned corpus keeps 64 languages; `data_status=dropped_quality` marks languages removed by the PPL/boilerplate cleanup, not by linguistic applicability.
- Public trait set: 38 unique `feature_clean` labels from the English SynthSTEL parquet.
- Raw contrast rows: 40. The apparent 40 rows become 38 public traits because `Usage of Self-Focused Perspective or Words` has three raw contrast variants.

## Status Semantics

- `direct`: use the English trait definition directly.
- `adapt`: keep the trait but rewrite the operational definition for the language.
- `drop`: do not generate or evaluate this trait for the language.
- `recommended_status`: pragmatic setting for dataset generation.
- `strict_status`: conservative cross-lingual setting; adapted traits are dropped unless a later experiment explicitly validates the adapted form.

## Accepted Fixes

- `hu`: `Usage of Prepositions` is now `adapt` via Hungarian case suffixes and postpositions.
- `fi`: `Usage of Prepositions` is now `adapt` via Finnish case endings and adpositions.
- `et`: `Usage of Prepositions` is now `adapt` via Estonian case endings and adpositions.
- `vi`: `Usage of Long Words` is now `adapt` because Vietnamese spacing does not cleanly correspond to alphabetic word length.
- `el`: `Usage of Numerical Substitution` is now `adapt` via Greeklish-style digit substitutions.

## Adapt Rather Than Abandon

- `PREP_ADAPT`: covers languages where English-like prepositions should become postpositions, case markers, case suffixes, or particles.
- `NUMSUB_ADAPT`: keeps strong internet-writing cases such as Arabizi, Greeklish, CJK phonetic number substitution, Korean/Thai local forms, and Cyrillic visual substitutions.
- `ART_ADAPT_OPTIONAL`: keeps `id` and `ms` as optional demonstrative/definiteness adaptations with strict mode still dropping the article trait.
- `WORDLEN_ADAPT`: keeps `ja my th vi zh` only through localized length proxies rather than whitespace-delimited alphabetic word length.

## Partial Or Conservative Calls

- Cyrillic `NUMSUB` languages are recommended as `adapt`, but strict mode remains `drop` because usage is less universally stable than Greeklish or Arabizi.
- Indic/native-script `NUMSUB` cases remain `drop` for this pass unless a later web/register study establishes robust local digit substitution conventions.
- No-article languages remain `drop` for `Usage of Articles`, except optional `id/ms` adaptation. This avoids turning the trait into a broad determiner/demonstrative feature.
- Casing traits remain hard drops for uncased scripts.

## Summary Counts

- Recommended status counts: {'adapt': 51, 'direct': 2674, 'drop': 125}
- Strict status counts: {'direct': 2674, 'drop': 176}
- Flag counts: {'ART_ADAPT_OPTIONAL': 2, 'ART_DROP': 42, 'CASE_DROP': 66, 'NUMSUB_ADAPT': 16, 'NUMSUB_DROP': 17, 'PREP_ADAPT': 28, 'WORDLEN_ADAPT': 5}
- Quality-dropped languages still included in the map: af cy eo eu ga la mn mt so sw uz

## Source Pointers For Spot Checks

- Hungarian postpositions: https://www.hungarianreference.com/postpositions-prepositions-personal-pronomial-before-after-between-instead-without.aspx
- Finnish adpositions: https://jkorpela.fi/finnish/Prepositions_and_postpositions.html
- Estonian postpositions/adpositions: https://www.colanguage.com/estonian-prepositions-and-postpositions
- Vietnamese word segmentation: https://universaldependencies.org/vi/index.html
- Greeklish examples: https://academickids.com/encyclopedia/index.php/Greeklish
- Arabizi digit conventions: https://kaleela.com/en/blog/what-is-arabizi-a-guide-to-help-you-understand-the-arabic-chat-alphabet/
