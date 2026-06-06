"""
translate.py
------------
Translate a contrast dataset (positive/negative/feature) into multiple
target languages using liteLLM, following the HuggingFace Datasets
subset layout: one subdirectory per language, same parquet filename.

Output layout:
    output-dir/
        en/train-00000-of-00001.parquet   ← original, copied as-is
        zh/train-00000-of-00001.parquet
        ja/train-00000-of-00001.parquet
        fr/train-00000-of-00001.parquet
        ru/train-00000-of-00001.parquet
        zh/train-00000-of-00001.ckpt.jsonl  ← resume markers (per lang)

Each translated parquet has the same columns as the source
(positive, negative, feature, feature_clean), just in the target language.
Rows for skipped features keep None.

Usage:
    # single split
    python translate.py --input data/train.parquet --output-dir out/

    # whole splits directory
    python translate.py --splits-dir data/ --output-dir out/

Dependencies:
    pip install litellm pandas pyarrow
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import litellm
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────

litellm.api_base = "http://127.0.0.1:3000/v1"
MODEL = os.environ.get("LITELLM_MODEL", "openai/gemini-3-flash")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 20))
PARALLEL = int(os.environ.get("PARALLEL", 4))

LANGS = {
    "zh": ("中文", "请对以上内容进行意译，使其符合中文母语者的自然表达习惯："),
    "ja": (
        "日本語",
        "上記の内容を意訳し、日本語母語話者として自然な表現にしてください：",
    ),
    "fr": (
        "Français",
        "Traduisez librement ce qui précède en français naturel et idiomatique :",
    ),
    "ru": (
        "Русский",
        "Сделайте вольный перевод вышеизложенного на естественный русский язык:",
    ),
}

# Loaded once at startup from feature_lang.json (same directory as this file)
_FEATURE_LANG: dict = {}


def load_feature_lang() -> None:
    path = Path(__file__).with_name("feature_lang.json")
    global _FEATURE_LANG
    with path.open(encoding="utf-8") as f:
        _FEATURE_LANG = json.load(f)


def feature_lang_entry(feature: str, lang_code: str) -> dict:
    return _FEATURE_LANG.get(feature, {}).get(lang_code, {})


# ── JSON schema for structured output ────────────────────────────────────────

RESPONSE_SCHEMA = {
    "name": "translation",
    "strict": True,
    "schema": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "positive": {"type": "string"},
                "negative": {"type": "string"},
            },
            "required": ["id", "positive", "negative"],
            "additionalProperties": False,
        },
    },
}

SYSTEM_PROMPT = "Output ONLY valid JSON matching the provided schema. No explanation."

# ── Checkpoint ────────────────────────────────────────────────────────────────


def ckpt_path(lang_parquet: Path) -> Path:
    return lang_parquet.with_suffix(".ckpt.jsonl")


def load_checkpoint(lang_parquet: Path) -> set[str]:
    """Return set of feature names already fully translated for this lang file."""
    done: set[str] = set()
    p = ckpt_path(lang_parquet)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            done.add(entry["feature"])
    return done


def save_checkpoint(lang_parquet: Path, feature: str) -> None:
    with ckpt_path(lang_parquet).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"feature": feature}, ensure_ascii=False) + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────


def feature_description(feature: str) -> str:
    parts = [p.strip() for p in feature.split("/")]
    if len(parts) == 2:
        return (
            f'• "positive" is the {parts[0]} version\n'
            f'• "negative" is the {parts[1]} version'
        )
    return f"• The feature dimension is: {feature}"


def batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def build_prompt(rows: list[dict], feature: str, lang_code: str) -> str:
    lang_label, lang_suffix = LANGS[lang_code]
    desc = feature_description(feature)
    note = feature_lang_entry(feature, lang_code).get("text", "")
    data = json.dumps(
        [
            {"id": r["id"], "positive": r["positive"], "negative": r["negative"]}
            for r in rows
        ],
        ensure_ascii=False,
        indent=2,
    )
    note_block = f"\n{note}\n" if note else ""
    return (
        f'Feature: "{feature}"\n'
        f"{desc}\n\n"
        f"{data}\n\n"
        f"BAN DIRECT PROPER NOUND COPY-PASTE! The English sentences above are reference material only — do NOT DIRECT translate them. No direct English nouns in the output, replace them to localized alternatives.\n"
        f"Instead, re-create each sentence from scratch as a native {lang_label} speaker would\n"
        f"naturally write it, preserving only the content topic and the style feature contrast, avoid just copy-paste the nouns.\n"
        f"The output must feel fully native: idiomatic phrasing, natural register, no trace\n"
        f"of English sentence structure.\n"
        f"{note_block}\n"
        f"{lang_suffix}"
    )


def call_llm(prompt: str) -> list[dict] | None:
    try:
        resp = litellm.completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": RESPONSE_SCHEMA,
            },
            reasoning={"effort": "medium"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as exc:
        print(f"  [warn] LLM call failed: {exc}", file=sys.stderr)
        return None


# ── Per-language translation ──────────────────────────────────────────────────


def translate_lang(df_en: pd.DataFrame, lang_code: str, out_path: Path) -> None:
    """
    Translate df_en into lang_code and write to out_path.

    Resume behaviour:
      - If out_path already exists, read it back as the working result so
        previously translated rows are not lost.
      - A .ckpt.jsonl sidecar tracks which feature groups are fully done;
        those are skipped on resume.
      - The parquet is flushed after every completed feature group, so a
        crash loses at most one batch of BATCH_SIZE rows.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lang_label = LANGS[lang_code][0]

    # Restore partial result or start fresh
    if out_path.exists():
        result = pd.read_parquet(out_path)
        print(f"  Resuming from existing {out_path.name}")
    else:
        result = df_en[["positive", "negative", "feature", "feature_clean"]].copy()
        result["positive"] = None
        result["negative"] = None

    done = load_checkpoint(out_path)
    groups = df_en.groupby("feature", sort=False)
    total_groups = df_en["feature"].nunique()

    for g_idx, (feature, group) in enumerate(groups, 1):
        entry = feature_lang_entry(feature, lang_code)

        if entry.get("action") == "skip":
            print(
                f"  [{g_idx}/{total_groups}] '{feature}'"
                f"  → skipped (not applicable in {lang_label})"
            )
            continue

        if feature in done:
            print(f"  [{g_idx}/{total_groups}] '{feature}'  → already done")
            continue

        rows_as_dicts = (
            group[["positive", "negative"]]
            .rename_axis("id")
            .reset_index()
            .to_dict("records")
        )
        batches_list = list(batched(rows_as_dicts, BATCH_SIZE))
        n_batches = len(batches_list)
        failed_any = False

        with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
            futures = {}
            for b_idx, batch in enumerate(batches_list, 1):
                prompt = build_prompt(batch, feature, lang_code)
                future = executor.submit(call_llm, prompt)
                futures[future] = (b_idx, batch)

            for future in as_completed(futures):
                b_idx, batch = futures[future]
                print(
                    f"  [{g_idx}/{total_groups}] '{feature}'  batch={b_idx}/{n_batches}"
                )
                results = future.result()

                if results is None or len(results) != len(batch):
                    if results is not None:
                        print(
                            f"    [warn] id count mismatch: "
                            f"sent {len(batch)}, got {len(results)}",
                            file=sys.stderr,
                        )
                    failed_any = True
                    continue

                id_to_result = {r["id"]: r for r in results}
                for row in batch:
                    orig_idx = row["id"]
                    if orig_idx in id_to_result:
                        result.at[orig_idx, "positive"] = id_to_result[orig_idx][
                            "positive"
                        ]
                        result.at[orig_idx, "negative"] = id_to_result[orig_idx][
                            "negative"
                        ]

        if not failed_any:
            save_checkpoint(out_path, feature)
            result.to_parquet(out_path, index=False)  # flush

    result.to_parquet(out_path, index=False)  # final write


# ── Per-file orchestration ────────────────────────────────────────────────────


def process_file(input_path: Path, output_dir: Path) -> None:
    df_en = pd.read_parquet(input_path)
    assert {"positive", "negative", "feature"} <= set(df_en.columns), (
        "Input must have columns: positive, negative, feature"
    )

    # English subset — copy as-is
    en_out = output_dir / "en" / input_path.name
    en_out.parent.mkdir(parents=True, exist_ok=True)
    df_en.to_parquet(en_out, index=False)
    print(f"  Copied English → {en_out}")

    for lang_code in LANGS:
        lang_out = output_dir / lang_code / input_path.name
        print(f"\n{'─' * 60}")
        print(f"  Language : {LANGS[lang_code][0]} ({lang_code})")
        print(f"  Output   : {lang_out}")
        print(f"{'─' * 60}")
        translate_lang(df_en, lang_code, lang_out)
        print(f"  Saved → {lang_out}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single input parquet file")
    group.add_argument("--splits-dir", help="Directory containing split parquet files")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    args = parser.parse_args()

    load_feature_lang()
    output_dir = Path(args.output_dir)

    if args.input:
        print(f"\nProcessing: {args.input}")
        process_file(Path(args.input), output_dir)
    else:
        parquets = sorted(Path(args.splits_dir).glob("*.parquet"))
        if not parquets:
            print(f"No parquet files found in {args.splits_dir}", file=sys.stderr)
            sys.exit(1)
        for p in parquets:
            print(f"\n{'═' * 60}")
            print(f"  Split: {p.name}")
            print(f"{'═' * 60}")
            process_file(p, output_dir)


if __name__ == "__main__":
    main()
