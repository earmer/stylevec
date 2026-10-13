from __future__ import annotations


def char_ngrams(text: str, n: int = 2) -> frozenset[str]:
    compact = "".join(text.split())
    if len(compact) < n:
        return frozenset([compact]) if compact else frozenset()
    return frozenset(compact[i : i + n] for i in range(len(compact) - n + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
