#!/usr/bin/env python3
"""
Temporary script: rough Chinese paraphrase similarity demos.

Implements:
1) Character n-gram Jaccard similarity (set-based by default).
2) Normalized Levenshtein similarity: 1 - dist / max_len.

Also supports:
- Splitting a single input into two blocks by a delimiter (e.g. "///").
- Line-by-line scoring for two multi-line blocks (useful for dialogue).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter


def _maybe_normalize(text: str, strip_space: bool) -> str:
    if not strip_space:
        return text
    # Keep it simple: drop all Unicode whitespace.
    return "".join(ch for ch in text if not ch.isspace())


def _drop_chars(text: str, drop: str) -> str:
    if not drop:
        return text
    table = {ord(ch): None for ch in drop}
    return text.translate(table)


def ngrams(text: str, n: int) -> list[str]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def jaccard_ngram_similarity(a: str, b: str, n: int, multiset: bool) -> float:
    a_grams = ngrams(a, n)
    b_grams = ngrams(b, n)

    if not a_grams and not b_grams:
        return 1.0

    if multiset:
        ca = Counter(a_grams)
        cb = Counter(b_grams)
        inter = sum((ca & cb).values())
        union = sum((ca | cb).values())
        return inter / union if union else 1.0

    sa = set(a_grams)
    sb = set(b_grams)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0


def levenshtein_distance(a: str, b: str) -> int:
    # O(len(a)*len(b)) time, O(min(len(a),len(b))) memory.
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        short, long_ = a, b
    else:
        short, long_ = b, a

    prev = list(range(len(short) + 1))
    for i, lc in enumerate(long_, start=1):
        cur = [i]
        for j, sc in enumerate(short, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if lc == sc else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def normalized_levenshtein_similarity(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (dist / max_len)


def _print_scores(a: str, b: str, n: int, multiset: bool) -> None:
    j = jaccard_ngram_similarity(a, b, n=n, multiset=multiset)
    l = normalized_levenshtein_similarity(a, b)
    print(f"n-gram Jaccard (n={n}, multiset={multiset}): {j:.6f}")
    print(f"normalized Levenshtein similarity:         {l:.6f}")


def _bench(a: str, b: str, n: int, multiset: bool, iters: int) -> None:
    # Very rough wall-clock timing.
    t0 = time.perf_counter()
    for _ in range(iters):
        jaccard_ngram_similarity(a, b, n=n, multiset=multiset)
    t1 = time.perf_counter()
    for _ in range(iters):
        normalized_levenshtein_similarity(a, b)
    t2 = time.perf_counter()

    print(f"bench iters: {iters}")
    print(f"n-gram Jaccard total:      {t1 - t0:.6f}s")
    print(f"normalized Levenshtein total: {t2 - t1:.6f}s")


def _split_pair(text: str, delim: str) -> tuple[str, str]:
    parts = text.split(delim)
    if len(parts) != 2:
        raise ValueError(f"expected exactly 2 blocks separated by {delim!r}, got {len(parts)} blocks")
    return parts[0], parts[1]


def _iter_nonempty_lines(text: str) -> list[str]:
    # Preserve order, drop empty/whitespace-only lines.
    return [line.strip() for line in text.splitlines() if line.strip()]


def _print_line_by_line(a: str, b: str, n: int, multiset: bool) -> None:
    a_lines = _iter_nonempty_lines(a)
    b_lines = _iter_nonempty_lines(b)

    m = min(len(a_lines), len(b_lines))
    print(f"lines A: {len(a_lines)}")
    print(f"lines B: {len(b_lines)}")
    if len(a_lines) != len(b_lines):
        print(f"note: line count differs; comparing first {m} line pairs")

    for i in range(m):
        la = a_lines[i]
        lb = b_lines[i]
        j = jaccard_ngram_similarity(la, lb, n=n, multiset=multiset)
        l = normalized_levenshtein_similarity(la, lb)
        print(f"[{i+1:02d}] jaccard={j:.6f} lev_norm={l:.6f}")
        print(f"A: {la}")
        print(f"B: {lb}")


def _bench_lines(a: str, b: str, n: int, multiset: bool, iters: int) -> None:
    a_lines = _iter_nonempty_lines(a)
    b_lines = _iter_nonempty_lines(b)
    pairs = list(zip(a_lines, b_lines))

    if not pairs:
        print("bench: no line pairs to compare")
        return

    t0 = time.perf_counter()
    for _ in range(iters):
        for la, lb in pairs:
            jaccard_ngram_similarity(la, lb, n=n, multiset=multiset)
    t1 = time.perf_counter()
    for _ in range(iters):
        for la, lb in pairs:
            normalized_levenshtein_similarity(la, lb)
    t2 = time.perf_counter()

    print(f"bench mode: line_by_line")
    print(f"bench iters: {iters}")
    print(f"line pairs: {len(pairs)}")
    print(f"n-gram Jaccard total:         {t1 - t0:.6f}s")
    print(f"normalized Levenshtein total: {t2 - t1:.6f}s")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("a", help="text A, or combined text containing both blocks (use '-' to read from stdin)")
    p.add_argument("b", nargs="?", default=None, help="text B (optional if --split-delim is used)")
    p.add_argument("-n", "--ngram", type=int, default=3, help="n for char n-grams (default: 3)")
    p.add_argument("--multiset", action="store_true", help="use multiset Jaccard (Counter-based)")
    p.add_argument("--strip-space", action="store_true", help="remove all whitespace before scoring")
    p.add_argument(
        "--drop-chars",
        default="",
        help="drop these characters before scoring (e.g. '：' to remove Chinese colon)",
    )
    p.add_argument(
        "--split-delim",
        default="",
        help="if set, split a single combined input into A/B blocks by this delimiter (e.g. '///')",
    )
    p.add_argument(
        "--line-by-line",
        action="store_true",
        help="score non-empty lines in order, printing per-line results",
    )
    p.add_argument("--bench", type=int, default=0, metavar="ITERS", help="run simple benchmark loop")
    args = p.parse_args(argv)

    if args.a == "-":
        a_raw = sys.stdin.read()
    else:
        a_raw = args.a

    if args.split_delim:
        a_raw, b_raw = _split_pair(a_raw if args.b is None else (a_raw + args.split_delim + args.b), args.split_delim)
    else:
        if args.b is None:
            raise SystemExit("missing argument: b (or pass --split-delim to split a single combined input)")
        b_raw = args.b

    a = _maybe_normalize(a_raw, strip_space=args.strip_space)
    b = _maybe_normalize(b_raw, strip_space=args.strip_space)
    a = _drop_chars(a, args.drop_chars)
    b = _drop_chars(b, args.drop_chars)

    if args.line_by_line:
        _print_line_by_line(a, b, n=args.ngram, multiset=args.multiset)
        if args.bench > 0:
            _bench_lines(a, b, n=args.ngram, multiset=args.multiset, iters=args.bench)
    else:
        _print_scores(a, b, n=args.ngram, multiset=args.multiset)
        if args.bench > 0:
            _bench(a, b, n=args.ngram, multiset=args.multiset, iters=args.bench)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
