#!/usr/bin/env python3
"""Post-hoc repair of the labeled bookraw Parquet exports.

The source EPUBs and `work_catalog_*.jsonl` are no longer present, so
`extract_labeled_bookraw.py` cannot be re-run to fix its own defects. This
script repairs the existing exports from the evidence they already carry, and
regenerates every derived view from the repaired master so the five files stay
consistent with each other.

Repairs, in order:

1. Whitespace injection. `dom_blocks()` extracted text with
   `tag.get_text(" ")`, which inserts a space at every inline-tag boundary.
   That splits Chinese words (`有蕡 其实 〔1〕` for `有蕡其实〔1〕`) and strands a
   space before fullwidth punctuation (`〔1〕 。`). Only typographically
   impossible spacing is removed: a space after fullwidth punctuation, or a
   space abutting a fullwidth bracket. Spaces between two CJK characters are
   left alone, because those are genuine (`二十七日 晴。` in a Lu Xun diary
   entry). English rows are never touched.

2. The 后记 role-state leak. `classify_blocks()` sets `role_state` when an
   inline block matches a role cue and never resets it, so every later block in
   the work inherits that cue. Jin Yong's novels are 99%+ novel prose that a
   stray `后记` marker relabelled as `editorial_matter`; those are restored to
   `primary_text`. Works where the cue landed on a real, short, terminal
   afterword (Ba Jin's `寒夜`, `憩园`) are left alone, because there the
   inherited label is correct.

3. Wrong-author prefaces. A section whose title credits a different writer
   (`OSCAR WILDE: AN IDLER'S IMPRESSION by Edgar Saltus`) is not the catalog
   author's primary text, so those rows are re-attributed to the writer the
   section names.

4. Language mismatches. `lang` is copied from the catalog, not measured from
   the text. Rows marked `zh` that carry Latin prose with no CJK (Hu Shi's
   English writings) are corrected to `en`.

5. Junk rows. Text that is only punctuation or only a page number carries no
   style signal and is dropped from the master, as is Project Gutenberg licence
   boilerplate, which ships inside the book body and otherwise survives the role
   and length gates as apparent `primary_text`.

Derived views are rebuilt with the masks from `extract_labeled_bookraw.main()`,
so `books_work_passages_labeled`, `books_work_primary_clean`,
`books_work_primary_strict` and `books_work_translations` never drift from the
master. `books_work_passages_labeled` is a byte-identical duplicate of the
master: its filter keeps all six emitted roles and no `unknown` row survives
extraction, so it is written as the filtered frame rather than copied.

Run from the repository root:

    python3 scripts/bookraw/repair_labeled_parquet.py --dry-run
    python3 scripts/bookraw/repair_labeled_parquet.py
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Tables live on the data volume; see reports/bookraw/README.md.
OUT = Path(os.environ.get("STYLEVEC_DATA", "/Volumes/EMC/stylevec_data")) / "processed/bookraw"

CJK = r"\u3400-\u9fff"
HAS_CJK = re.compile(f"[{CJK}]")
HAS_LATIN = re.compile(r"[A-Za-z]")

# Typographically impossible spacing in Chinese: a space after fullwidth
# punctuation, or a space abutting a fullwidth bracket. Both are artifacts of
# `get_text(" ")` splitting on an inline tag boundary. A space between two CJK
# characters may be genuine and is deliberately left alone.
PUNCT_AFTER = re.compile(r"(?<=[，。、；：？！）」』】〕《〉·]) +")
SPACE_BEFORE_OPEN = re.compile(r" +(?=[（「『【〔《〈])")
SPACE_BEFORE_CLOSE = re.compile(r"(?<=[（「『【〔《〈]) +(?=[）」』】〕《〉])")
# `get_text(" ")` also strands a space *after* a closing bracket, because the
# bracket and its sibling text land in separate inline elements.
SPACE_AFTER_CLOSE = re.compile(r"(?<=[）」』】〕《〉]) +(?=[\u3400-\u9fff（「『【〔《])")

ONLY_SYMBOLS = re.compile(r"^[\W_]+$", re.UNICODE)
PAGE_NUMBER = re.compile(r"^[\s0-9ivxlcdmIVXLCDM]{1,12}$")
# A letter of a prose script. Text with none is index furniture, a separator, or
# a stray numeral. `\w` is not usable here: it is Unicode-aware, so it treats
# `Ⅱ` (category Nl) and the kana pairs `わワ` that OCR emits for a lost glyph as
# word characters, and `\W` misses them for the same reason. Cyrillic and Greek
# count as prose: the notes index sorts Russian loanwords by initial
# (`А（565）Б（565）`) and real Russian prose is in the corpus, so neither is
# index furniture and neither should be deleted.
PROSE_LETTER = re.compile(rf"[{CJK}]|[A-Za-z]|[぀-ヿ]|[Ѐ-ӿ]")
# A leaked marker can land on a navigation stub (`返回总目录`) rather than on
# swallowed prose. Relabelling that to `primary_text` would only swap one wrong
# label for another, so the leak repair skips anything already caught as junk.
NAVIGATION = re.compile(r"^(?:返回|回到|back to)\s*(?:总)?目录$|^return to (?:table of )?contents$", re.I)

# Project Gutenberg ships its licence inside the book body, so it survives the
# length and role gates as apparent `primary_text`. Match the licence's own
# vocabulary rather than the bare word "Gutenberg", which also appears in real
# prose (Mark Twain on the Gutenberg Bible, a press-history essay).
GUTENBERG_BOILERPLATE = re.compile(
    r"project gutenberg|pg laf|gutenberg literary archive foundation|"
    r"gutenberg\.org|gutenberg\.org/license|redistribut(e|ing) project gutenberg",
    re.I,
)

# Chinese ebook-piracy stamps injected into the text layer of otherwise English
# novels (`公众号：古德猫宁李`, `网站： 沉金书屋 https://www.chenjin5.com`).
PIRACY_WATERMARK = re.compile(
    r"公众号[:：]|沉金书屋|chenjin5\.com|古德猫宁李|书伴|bookzz|wwxiao\.com",
    re.I,
)

# HTML entities the extractor failed to unescape. `&#9;` is a tab and appears in
# 12 rows of a Hu Shi logic text; every other entity class is already absent
# from the corpus, so these are an isolated miss rather than a general one.
HTML_ENTITIES = {"&#9;": " ", "&#160;": " ", "&nbsp;": " "}

# A leaked role is only safe to revert when it swallowed most of the work.
LEAKED_EVIDENCE = "inline_heading:后记"
# The first leaked block must appear within this fraction of the work, from the
# start. Jin Yong's leaked regions begin in the first ~0.5% (after a contents
# list); a genuine afterword begins at 0.97-0.999 (the tail of the book).
LEAK_MAX_ONSET = 0.05

# `… by <Name>` in a section title credits a different author for that text.
# The trailing name may carry typographic apostrophes and middle initials.
SECTION_AUTHOR = re.compile(
    r"\bby\s+([A-Z][\w.'’-]*(?:\s+[A-Z][\w.'’-]*){0,3})\s*$",
    re.UNICODE,
)
# Prefatory sections that are somebody else's prose by definition. Matching is
# anchor-free so that typographic apostrophes (U+2019) inside words such as
# `IDLER'S` cannot break the surrounding word boundaries.
FOREIGN_SECTION = re.compile(
    r"(impression|recollections?|reminiscences?|memories|confessions?|"
    r"biography|life and|his life)",
    re.I,
)


def fix_zh_spacing(text: str) -> str:
    text = PUNCT_AFTER.sub("", text)
    text = SPACE_BEFORE_OPEN.sub("", text)
    text = SPACE_BEFORE_CLOSE.sub("", text)
    text = SPACE_AFTER_CLOSE.sub("", text)
    return text


def is_junk(text: str) -> bool:
    return (
        not text
        or bool(ONLY_SYMBOLS.match(text))
        or bool(PAGE_NUMBER.match(text))
        or not PROSE_LETTER.search(text)
    )


def is_boilerplate(text: str) -> bool:
    return bool(GUTENBERG_BOILERPLATE.search(text))


def is_watermark(text: str) -> bool:
    return bool(PIRACY_WATERMARK.search(text))


def unescape_entities(text: str) -> str:
    for entity, replacement in HTML_ENTITIES.items():
        if entity in text:
            text = text.replace(entity, replacement)
    return re.sub(r"\s+", " ", text).strip()


def disambiguate_section_ids(master: pd.DataFrame) -> pd.Series:
    """Give each `(work_id, section_title)` pair a unique `section_id`.

    The extractor hashed `(work_id, spine_index, dom_position, section_title)`, so
    a title repeated inside one work -- a chapter name used twice, or a
    back-matter section re-entering a spine item -- produced a *different* id per
    occurrence, and 26 ids in the corpus end up mapping to more than one title.
    A section is identified by its title within a work, so hash that instead. The
    empty-string sentinel (no resolved section) is left alone.
    """
    pairs = master.loc[master.section_title.fillna("").ne(""), ["work_id", "section_title"]]
    fresh = {
        (work_id, title): hashlib.sha1(f"{work_id}\0{title}".encode("utf-8")).hexdigest()[:16]
        for work_id, title in pairs.drop_duplicates().itertuples(index=False, name=None)
    }
    return pd.Series(
        [fresh.get(key) for key in master[["work_id", "section_title"]].itertuples(index=False, name=None)],
        index=master.index,
    )


def is_navigation(text: str) -> bool:
    return bool(NAVIGATION.match(text.strip()))


# A block's own reference marks. Editions differ, and each style pairs with its
# own note syntax:
#   中华经典藏书  `举头望明月〔2〕`  glossed by `〔2〕 却顾：回头望。`
#   other volumes   `孝经 … [1]`       glossed by `[1] …`
#   老子 chapters   `大成若缺 (1)`    glossed by `(1) 成：善。`
# A single pattern is not enough, and `(1)` is far too common in classical prose
# to be read on its own -- it is only trusted when the section's notes define it.
FULLWIDTH_REF = re.compile(r"〔(\d+)〕")
HALFWIDTH_REF = re.compile(r"\[(\d{1,6})\]")
PAREN_REF = re.compile(r"\((\d{1,6})\)")
PAREN_NOTE = re.compile(r"^\s*\((\d{1,6})\)")
# The EPUB writes a poet's name as a pseudo-verse with its own biographical
# note: `白居易 (2)`, `程颢 (2)`, `陈抟 (2)`. That is a credit line, not the
# work, and its number resolves to a note about the poet rather than a gloss of
# the text, so it otherwise passes the gloss test.
AUTHOR_CREDIT_LINE = re.compile(r"^[^\s，。；：！？、（）()《》〔〕\[\]]{2,8}\s*[(（]\d{1,6}[)）]\s*$")

# Evidence for the Zhonghua-classics state machine, which labels everything after
# an unlabelled `<h>` heading as commentary.
CHAPTER_INTRO = "chapter_intro_before_body_marker"
# Apparatus that gloses a section's text. A block under one of these headings
# numbers the references of the work it belongs to.
NOTE_HEADINGS = {
    "inline_heading:【注释】",
    "heading:【注释】",
    "inline_heading:【译文】",
    "heading:【译文】",
}
# Numbered notes that the marker rule caught before any heading was seen, so they
# carry no `【注释】` evidence of their own. 孝经 numbers its notes this way.
NOTE_EVIDENCE = "epub_note_semantics_or_numbered_note"
# Commentary written by the volume's modern editors rather than by the classical
# author it discusses. The reviewed 中华经典藏书 volume credits 胡平生、王国轩、
# 陈晓芬、于天池、陈曦, so a `commentary` block in it is theirs even when the
# catalog names 王国维 or 王永彬 as the `author` of the work being analysed.
#
# The value written into `author` for those blocks. A named editor would be a
# guess -- the volume credits five of them for one file -- so record that the
# author is the modern editorial voice rather than guessing one.
EDITORIAL_AUTHOR = "(现代编者)"


def ref_numbers(text: str, styles: tuple[str, ...] = ("fullwidth", "halfwidth")) -> set[str]:
    """Reference numbers carried by a block, in the requested bracket styles."""
    found: set[str] = set()
    if "fullwidth" in styles:
        found.update(FULLWIDTH_REF.findall(text or ""))
    if "halfwidth" in styles:
        found.update(HALFWIDTH_REF.findall(text or ""))
    if "paren" in styles:
        found.update(PAREN_REF.findall(text or ""))
    return found


def glossed_numbers(master: pd.DataFrame) -> dict[tuple[str, str], dict[str, set[str]]]:
    """Reference numbers each section's apparatus actually defines.

    Returned per bracket style, because the evidence only counts when a block and
    its note use the *same* style. A section whose notes are `〔n〕` cannot vouch
    for a block that cites `(n)`, and `(n)` in isolation is just a citation.
    """
    table: dict[tuple[str, str], dict[str, set[str]]] = {}
    notes = master[master.role_evidence.isin(NOTE_HEADINGS) | master.role_evidence.eq(NOTE_EVIDENCE)]
    for (work_id, section_id), group in notes.groupby(["work_id", "section_id"], sort=False):
        entry = {"fullwidth": set(), "halfwidth": set(), "paren": set()}
        for text in group.text:
            entry["fullwidth"].update(FULLWIDTH_REF.findall(text))
            entry["halfwidth"].update(HALFWIDTH_REF.findall(text))
            # Only a note that *opens* with `(n)` defines it; a prose aside that
            # merely cites `(3)` does not. Every line-start match counts, so a
            # block carrying several notes defines all of them.
            for line in str(text).splitlines():
                match = PAREN_NOTE.match(line)
                if match:
                    entry["paren"].add(match.group(1))
        table[(work_id, section_id)] = entry
    return table


def restore_own_text(master: pd.DataFrame) -> pd.Series:
    """Recover work text mislabelled `commentary` by the classics heading rules.

    In the reviewed 中华经典藏书 edition, `classify_blocks` set
    `role_state = "commentary"` at every unlabelled heading, and separately
    labelled everything after the work title as introductory, so the poems and
    essays themselves were filed as commentary: 李白's 月下独酌, 孟郊's 游子吟
    and the 关雎 stanzas among them. That left 289 authors with no
    `primary_text` at all.

    A block is restored only when the reference marks it carries are glossed by
    the apparatus block of the *same* section, which commentary about a poem
    never does. Three bracket styles are recognised, and a block only counts when
    its own style is the style that section's notes use: `〔n〕` in 中华经典藏书,
    `[n]` in other volumes, and `(n)` where a note block opens with `(n)`.

    The same test is applied to every commentary evidence, not just the heading
    state, because `introductory_material_after_work_title` and
    `reviewed_publisher_commentary_style` also swallow poem bodies -- the first
    22 lines of 关雎 carry the second of those. Restricting the test to one
    evidence would leave those stranded as commentary *and* strip them of their
    real author, which is worse than either error alone.
    """
    candidate = master.text_role.eq("commentary") & ~master.author.eq(EDITORIAL_AUTHOR)
    if candidate.empty:
        return candidate
    table = glossed_numbers(master)
    resolved = {}
    for index, row in master.loc[candidate, ["text", "work_id", "section_id"]].iterrows():
        defined = table.get((row.work_id, row.section_id))
        if not defined:
            resolved[index] = False
            continue
        # A block counts only where its own bracket style is the one its notes
        # define, so `(1)` cannot be satisfied by a section that numbers with
        # `〔1〕` and vice versa.
        hit = any(
            ref_numbers(row.text, (style,)) & defined[style] for style in defined if defined[style]
        )
        if hit and AUTHOR_CREDIT_LINE.match(str(row.text).strip()):
            # `白居易 (2)` is the EPUB's author-credit line, not the work. Its
            # number resolves to a biographical note, which makes it look exactly
            # like a glossed reference, so exclude the shape outright.
            hit = False
        resolved[index] = bool(hit)
    return pd.Series(resolved).reindex(master.index, fill_value=False)


def editorial_rows(master: pd.DataFrame) -> pd.Series:
    """Commentary whose `author` is the subject, not the writer.

    Every `commentary` block in this corpus comes from one 中华经典藏书 volume
    whose modern editors (胡平生、王国轩、陈晓芬、于天池、陈曦) are named in the
    source filename. Those editors write the 评析, the 题解, the 小传 and the
    导读, so none of it belongs to the classical author in the `author` column --
    237 rows of 王国轩's prose on 王国维's 人间词话 were recorded as 王国维's own
    writing. Record the modern editorial voice instead, rather than guessing which
    of the five editors wrote any given block.

    Rows already carrying that value are excluded so the repair is idempotent.
    """
    commentary = master.text_role.eq("commentary")
    already = master.author.eq(EDITORIAL_AUTHOR)
    return commentary & ~already


def leaked_works(master: pd.DataFrame, max_onset: float) -> set[str]:
    """Works whose `editorial_matter` role came from the unbounded role-state leak.

    A work qualifies when the first leaked block sits so late in reading order
    that everything after it is the leak -- the signature of a cue marker near
    a book's start poisoning the whole body. A genuine afterword sits at the
    *end* of a work, so it fails this test and is left alone.
    """
    leaked = master[master.role_evidence.eq(LEAKED_EVIDENCE) & master.text_role.eq("editorial_matter")]
    if leaked.empty:
        return set()
    ordered = master.sort_values(["work_id", "source_spine_index", "source_dom_position"], kind="stable")
    within_work = pd.Series(np.arange(len(ordered)), index=ordered.index).groupby(ordered.work_id).cumcount()
    first_leak = leaked.groupby("work_id").apply(
        lambda g: within_work.loc[g.index].min(), include_groups=False
    )
    totals = master.groupby("work_id").size()
    fraction = first_leak / totals
    return set(fraction[fraction <= max_onset].index)


def section_authors(master: pd.DataFrame) -> dict[str, str]:
    """Map `work_id` + section title to the author that section actually credits."""
    mapping: dict[tuple[str, str], str] = {}
    titles = master.loc[master.section_title.notna(), ["work_id", "section_title"]].drop_duplicates()
    for work_id, title in titles.itertuples(index=False):
        if not FOREIGN_SECTION.search(title):
            continue
        match = SECTION_AUTHOR.search(title.strip())
        if match:
            mapping[(work_id, title)] = match.group(1)
    return mapping


def build_views(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    length_ok = (
        ((master.lang.eq("zh")) & (master.text.str.len() >= 30))
        | ((~master.lang.eq("zh")) & (master.text.str.len() >= 80))
    )
    primary = master.text_role.eq("primary_text")
    confident = master.role_confidence.isin(["high", "medium"])
    resolved = master.boundary_confidence.isin(["high", "medium"])
    clean_roles = {
        "primary_text",
        "translation",
        "annotation",
        "commentary",
        "editorial_matter",
        "paratext",
    }
    return {
        "books_work_blocks_labeled.parquet": master,
        "books_work_passages_labeled.parquet": master[master.text_role.isin(clean_roles)],
        "books_work_primary_clean.parquet": master[primary & confident & resolved & length_ok],
        "books_work_primary_strict.parquet": master[
            primary & master.role_confidence.eq("high") & master.boundary_confidence.eq("high") & length_ok
        ],
        "books_work_translations.parquet": master[master.text_role.eq("translation")],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--leak-max-onset", type=float, default=LEAK_MAX_ONSET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = args.input_dir

    master = pd.read_parquet(data / "books_work_blocks_labeled.parquet")
    original_rows = len(master)

    spacing = master.lang.eq("zh") & master.text.map(fix_zh_spacing).ne(master.text)
    master.loc[spacing, "text"] = master.loc[spacing, "text"].map(fix_zh_spacing)

    leaked = leaked_works(master, args.leak_max_onset)
    # Only the blocks carrying the leaked evidence are touched, so a work's
    # genuine prefatory blocks (other evidence) survive the repair.
    relabelled = (
        master.work_id.isin(leaked)
        & master.role_evidence.eq(LEAKED_EVIDENCE)
        & master.text_role.eq("editorial_matter")
        & ~master.text.map(is_navigation)
    )
    master.loc[relabelled, ["text_role", "role_confidence", "role_evidence", "role_source"]] = [
        "primary_text",
        "medium",
        "role_state_leak_reverted_to_primary_text",
        "post_hoc_repair",
    ]

    restored = restore_own_text(master)
    master.loc[restored, ["text_role", "role_confidence", "role_evidence", "role_source"]] = [
        "primary_text",
        "medium",
        "classics_heading_state_restored_to_primary_text",
        "post_hoc_repair",
    ]

    credits = section_authors(master)
    credited = pd.Series(False, index=master.index)
    if credits:
        intended = pd.Series(
            [credits.get(key) for key in pd.MultiIndex.from_frame(master[["work_id", "section_title"]])],
            index=master.index,
        )
        # Only rows whose author actually differs are touched, so re-running the
        # repair is a no-op.
        credited = intended.notna() & intended.ne(master.author)
        master.loc[credited, "author"] = intended[credited]

    # Runs after the re-attribution above, so a section credit is applied to the
    # classical author's own prose before the editorial remainder is disowned.
    editorial = editorial_rows(master)
    master.loc[editorial, "author"] = EDITORIAL_AUTHOR
    master.loc[editorial, "role_evidence"] = (
        master.loc[editorial, "role_evidence"].astype(str) + "|attribution_to_modern_editor"
    )

    latin_in_zh = (
        master.lang.eq("zh")
        & ~master.text.str.contains(HAS_CJK, regex=True, na=False)
        & master.text.str.contains(HAS_LATIN, regex=True, na=False)
    )
    master.loc[latin_in_zh, "lang"] = "en"

    junk = master.text.map(is_junk) | master.text.map(is_boilerplate)
    symbols = master.text.map(is_junk)
    dropped = int(junk.sum())
    # Report counts against the rows that actually survive the deletions, so the
    # summary describes the files on disk rather than the pre-deletion frame.
    master = master[~junk].reset_index(drop=True)
    kept = ~junk

    print(f"spacing repaired        : {int((spacing & kept).sum())}")
    print(f"leak-reverted works     : {len(leaked)}")
    print(f"leak-reverted rows      : {int((relabelled & kept).sum())}")
    print(f"classics text restored  : {int((restored & kept).sum())}")
    print(f"re-attributed sections  : {int((credited & kept).sum())}")
    print(f"editorial author disowned: {int((editorial & kept).sum())}")
    print(f"lang zh->en             : {int((latin_in_zh & kept).sum())}")
    print(f"symbol/page rows dropped: {int(symbols.sum())}")
    print(f"licence rows dropped    : {int((junk & ~symbols).sum())}")
    print(f"rows                    : {original_rows} -> {len(master)}")

    if args.dry_run:
        return

    for name, frame in build_views(master).items():
        frame.to_parquet(data / name, index=False)
        print(f"wrote {name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
