"""Authorship-change probe for 红楼梦 (Hongloumeng).

Chapters 1-80: Cao Xueqin (Cheng editions via Project Gutenberg #24264).
Chapters 81-120: traditionally attributed to Gao E / Cheng Weiyuan.

Usage:
    # Full run with embedding cache (run on box):
    uv run scripts/hongloumeng_probe.py \
        --text outputs/hongloumeng/raw/pg24264.txt \
        --out  outputs/hongloumeng/results \
        --checkpoint outputs/runs/spread_a_1gpu__center__s42 \
        --matryoshka 64 256 \
        --batch-size 64 --device cuda \
        --save-embeddings

    # Local analysis from cache (no model needed):
    uv run scripts/hongloumeng_probe.py \
        --text outputs/hongloumeng/raw/pg24264.txt \
        --out  outputs/hongloumeng/results \
        --from-cache outputs/hongloumeng/results/cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# SECTION 1 – Text parsing
# ---------------------------------------------------------------------------
_CHAPTER_RE = re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+回)(?:[　\s]|$)")
_SENT_END = frozenset("。！？")
_PG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_PG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"

STYLE_CHARS = list(
    "之其或亦方即皆仍故尚呢了的着吗么罢吧便就道说且又也只不便把从"
    "将把对于而已乃与及更甚若因为所以则却"
)

# Main character names to mask (traditional + simplified; both may appear in the mixed encoding)
_CHAR_NAMES = [
    "賈寶玉", "宝玉", "林黛玉", "黛玉", "薛寶釵", "宝钗", "王熙鳳", "凤姐", "鳳姐",
    "賈母", "贾母", "王夫人", "襲人", "袭人", "晴雯", "探春", "湘雲", "湘云",
    "李紈", "李纨", "平兒", "平儿", "鴛鴦", "鸳鸯", "紫鵑", "紫鹃", "鶯兒", "莺儿",
    "麝月", "妙玉", "惜春", "迎春", "元春", "賈政", "贾政", "賈璉", "贾琏",
    "賈珍", "贾珍", "賈蓉", "贾蓉", "秦可卿", "尤二姐", "尤三姐",
    "薛姨媽", "薛姨妈", "薛蟠", "香菱", "劉姥姥", "刘姥姥", "趙姨娘", "赵姨娘",
    "賈環", "贾环", "彩雲", "彩云", "金釧", "金钏", "玉釧", "玉钏",
    "司棋", "芳官", "茗煙", "茗烟", "林之孝", "周瑞家", "寶蟾", "宝蟾",
    "夏金桂", "小紅", "小红", "齡官", "龄官",
]
# Sort longest first to avoid partial matches
_CHAR_NAMES_SORTED = sorted(set(_CHAR_NAMES), key=len, reverse=True)
_NAME_MASK_CHAR = "某"

# Dialogue markers for dialogue-share covariate
_DIALOGUE_RE = re.compile(r'道[：:「]|說[：:「]|说[：:「]|[「""\']')


def mask_names(text: str) -> str:
    for name in _CHAR_NAMES_SORTED:
        text = text.replace(name, _NAME_MASK_CHAR)
    return text


def _to_canonical_chapter_num(heading: str) -> int | None:
    _DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3,
              "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    _MULT = {"十": 10, "百": 100}
    m = re.match(r"第(.+?)回", heading)
    if not m:
        return None
    inner = m.group(1).strip()
    if re.match(r"^[0-9]+$", inner):
        return int(inner)
    has_mult = any(ch in _MULT for ch in inner)
    all_digits = all(ch in _DIGIT for ch in inner)
    if all_digits and not has_mult:
        s = "".join(str(_DIGIT[ch]) for ch in inner)
        return int(s) if s else None
    val = 0
    pending = 0
    for ch in inner:
        if ch in _MULT:
            mult = _MULT[ch]
            val += (pending if pending else 1) * mult
            pending = 0
        elif ch in _DIGIT:
            n = _DIGIT[ch]
            if n != 0:
                pending = n
        elif ch.isdigit():
            pending = pending * 10 + int(ch)
    val += pending
    return val if val else None


def parse_chapters(text_path: Path) -> dict[int, str]:
    raw = text_path.read_text(encoding="utf-8")
    start_idx = raw.find(_PG_START)
    end_idx = raw.find(_PG_END)
    if start_idx != -1:
        raw = raw[start_idx:]
    if end_idx != -1:
        raw = raw[:raw.find(_PG_END)]
    lines = raw.splitlines()
    seen: set[int] = set()
    chapter_starts: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        m = _CHAPTER_RE.match(line.strip())
        if not m:
            continue
        num = _to_canonical_chapter_num(m.group(1))
        if num is None or not (1 <= num <= 120):
            continue
        if num in seen:
            continue
        seen.add(num)
        chapter_starts.append((idx, num))
    if len(chapter_starts) != 120:
        raise ValueError(
            f"Expected 120 chapters, found {len(chapter_starts)}: "
            + str([n for _, n in chapter_starts])
        )
    chapter_starts.sort(key=lambda t: t[0])
    chapters: dict[int, str] = {}
    for i, (start_line, num) in enumerate(chapter_starts):
        end_line = chapter_starts[i + 1][0] if i + 1 < len(chapter_starts) else len(lines)
        body_lines = []
        for line in lines[start_line + 1 : end_line]:
            stripped = line.strip()
            if re.match(r"^[-=_＝—─━]{4,}", stripped):
                continue
            if _CHAPTER_RE.match(stripped):
                continue
            body_lines.append(line)
        chapters[num] = "\n".join(body_lines)
    return chapters


# ---------------------------------------------------------------------------
# SECTION 2 – Verse detection
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_VERSE_MARKERS = re.compile(r"^[　\s]*(詞|曲|賦|詩|歌|謠|令|調|牌)[　\s：:「]")
_PUNC_OR_SPACE = frozenset("，。！？；：、—…　 \t\r\n​")


def _cjk_len(s: str) -> int:
    return sum(1 for c in s if "一" <= c <= "鿿" or "㐀" <= c <= "䶿")


def is_verse_line(line: str) -> bool:
    """Detect verse lines: 5- or 7-char CJK lines, couplets, or marked by 詞/曲 etc."""
    s = line.strip()
    if not s:
        return False
    # Marker: starts with 詞/曲/賦/etc.
    if _VERSE_MARKERS.match(s):
        return True
    # Strip leading/trailing punctuation and spaces
    core = s.strip("，。！？；：、—…　 \t「」『』")
    n_cjk = _cjk_len(core)
    n_total = len(core.replace("　", "").replace(" ", ""))
    if n_total == 0:
        return False
    cjk_ratio = n_cjk / n_total
    if cjk_ratio < 0.6:
        return False
    # 5-char or 7-char lines (the most common classical verse lengths)
    if n_cjk in (5, 7) and n_total <= n_cjk + 2:
        return True
    # Couplet: two equal-length CJK clauses split by ，
    # e.g. "玉帶林中掛，金簪雪裡埋" → 5+5
    if "，" in core:
        halves = core.split("，")
        if len(halves) == 2:
            a, b = _cjk_len(halves[0]), _cjk_len(halves[1])
            if a == b and a in (4, 5, 6, 7, 8) and a > 0:
                return True
    # 4- or 6-char lines (some verse forms)
    if n_cjk in (4, 6) and n_total <= n_cjk + 1:
        return True
    return False


def has_dialogue(passage: str) -> bool:
    return bool(_DIALOGUE_RE.search(passage))


def chunk_chapter(
    text: str,
    min_chars: int = 60,
    max_chars: int = 150,
) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    buf: list[str] = []
    buf_len = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        verse = is_verse_line(stripped)
        buf.append(stripped)
        buf_len += len(stripped)
        if buf_len >= min_chars and any(c in _SENT_END for c in stripped):
            passage = "".join(buf)
            for segment in _split_at_sentence_ends(passage, max_chars):
                if len(segment.strip()) >= min_chars:
                    passages.append({
                        "text": segment.strip(),
                        "is_verse": verse,
                        "has_dialogue": has_dialogue(segment),
                    })
            buf = []
            buf_len = 0
    if buf:
        passage = "".join(buf).strip()
        if passage:
            passages.append({
                "text": passage[:max_chars],
                "is_verse": False,
                "has_dialogue": has_dialogue(passage),
            })
    return passages


def _split_at_sentence_ends(text: str, max_chars: int) -> list[str]:
    segments: list[str] = []
    start = 0
    last_end = 0
    for i, ch in enumerate(text):
        if ch in _SENT_END:
            last_end = i + 1
            if last_end - start >= max_chars:
                segments.append(text[start:last_end])
                start = last_end
    if start < len(text):
        tail = text[start:]
        while len(tail) > max_chars:
            segments.append(tail[:max_chars])
            tail = tail[max_chars:]
        if tail:
            segments.append(tail)
    return segments or [text[:max_chars]]


# ---------------------------------------------------------------------------
# SECTION 3 – Stylometry baseline
# ---------------------------------------------------------------------------

def stylometry_vectors(passages: list[str]) -> np.ndarray:
    mat = np.zeros((len(passages), len(STYLE_CHARS)), dtype=np.float32)
    for i, p in enumerate(passages):
        n = max(len(p), 1)
        for j, ch in enumerate(STYLE_CHARS):
            mat[i, j] = p.count(ch) / n
    mean = mat.mean(axis=0, keepdims=True)
    std = mat.std(axis=0, keepdims=True) + 1e-8
    return (mat - mean) / std


# ---------------------------------------------------------------------------
# SECTION 4 – Base-model embedding
# ---------------------------------------------------------------------------

def base_model_embed(
    passages: list[str],
    model_path: str,
    batch_size: int = 16,
    device_str: str = "auto",
) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if device_str == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=192, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=False)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = F.normalize(pooled.float(), p=2, dim=-1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


# ---------------------------------------------------------------------------
# SECTION 5 – Trained-checkpoint embedding
# ---------------------------------------------------------------------------

def trained_model_embed(
    passages: list[str],
    run_dir: Path,
    matryoshka_dims: list[int],
    batch_size: int = 16,
    device_str: str = "auto",
) -> dict[str, np.ndarray]:
    import torch
    from transformers import AutoTokenizer

    from config import load_config
    from model import build_qwen_style_encoder, load_trainable_state_dict
    from records import normalize_text

    config = load_config(run_dir / "config.yaml")

    if device_str == "auto":
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("mps") if torch.backends.mps.is_available()
                  else torch.device("cpu"))
    else:
        device = torch.device(device_str)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model_path"], local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_qwen_style_encoder(config).to(device)
    checkpoint_dir = run_dir / "checkpoints" / "latest"
    state_path = checkpoint_dir / "weights.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"weights.pt not found at {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "model_trainable" not in state:
        raise ValueError(f"checkpoint missing model_trainable: {state_path}")
    load_trainable_state_dict(model, state["model_trainable"])
    model.eval()

    proj_dim = int(config["model"]["projection_dim"])
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(passages), batch_size):
            batch = [normalize_text(p) for p in passages[i : i + batch_size]]
            enc = tokenizer(
                batch, padding=True, truncation=True,
                max_length=int(config["model"]["max_length"]), return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            emb = out["embeddings"].float().cpu().numpy()
            vectors.append(emb)

    full = np.concatenate(vectors, axis=0)
    result: dict[str, np.ndarray] = {"full": full}
    for dim in matryoshka_dims:
        if dim < proj_dim:
            prefix = full[:, :dim]
            norms = np.linalg.norm(prefix, axis=-1, keepdims=True)
            result[f"dim_{dim}"] = prefix / np.maximum(norms, 1e-8)
    return result


# ---------------------------------------------------------------------------
# SECTION 6 – Embedding cache I/O
# ---------------------------------------------------------------------------

def save_embeddings(
    cache_dir: Path,
    method_label: str,
    embeddings: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{method_label}.npy",
            embeddings.astype(np.float16))


def load_embeddings(cache_dir: Path, method_label: str) -> np.ndarray:
    path = cache_dir / f"{method_label}.npy"
    if not path.exists():
        raise FileNotFoundError(f"embedding cache not found: {path}")
    return np.load(path).astype(np.float32)


def save_passages_parquet(
    cache_dir: Path,
    passage_chapters: list[int],
    passage_lengths: list[int],
    passage_verse: list[bool],
    passage_dialogue: list[bool],
    passage_texts: list[str],
) -> None:
    import pandas as pd
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "chapter": passage_chapters,
        "length": passage_lengths,
        "is_verse": passage_verse,
        "has_dialogue": passage_dialogue,
        "text_preview": [t[:80] for t in passage_texts],
    })
    df.to_parquet(cache_dir / "passages.parquet", index=False)


def load_passages_parquet(cache_dir: Path):
    import pandas as pd
    return pd.read_parquet(cache_dir / "passages.parquet")


# ---------------------------------------------------------------------------
# SECTION 7 – Analysis helpers
# ---------------------------------------------------------------------------

def chapter_vectors(
    passage_embs: np.ndarray,
    passage_chapters: list[int] | np.ndarray,
) -> np.ndarray:
    """Mean of passage embeddings per chapter (1–120), L2-renormalised."""
    chapters_arr = np.asarray(passage_chapters)
    n_ch = int(chapters_arr.max())
    vecs: list[np.ndarray] = []
    for ch in range(1, n_ch + 1):
        idx = np.where(chapters_arr == ch)[0]
        if len(idx) > 0:
            mean = passage_embs[idx].mean(axis=0)
        else:
            mean = np.zeros(passage_embs.shape[1])
        norm = np.linalg.norm(mean)
        vecs.append(mean / max(norm, 1e-8))
    return np.stack(vecs, axis=0)


def cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    normed = vecs / np.maximum(norms, 1e-8)
    return normed @ normed.T


def regress_out_length(
    passage_embs: np.ndarray,
    passage_lengths: list[int] | np.ndarray,
) -> np.ndarray:
    log_len = np.log(np.asarray(passage_lengths, dtype=np.float32) + 1.0).reshape(-1, 1)
    X = np.hstack([np.ones((len(log_len), 1)), log_len])
    beta, _, _, _ = np.linalg.lstsq(X, passage_embs, rcond=None)
    residuals = passage_embs - X @ beta
    norms = np.linalg.norm(residuals, axis=-1, keepdims=True)
    return residuals / np.maximum(norms, 1e-8)


def change_point_test(
    ch_vecs: np.ndarray,
    n_permutations: int = 999,
    rng_seed: int = 42,
    skip_first: int = 0,
) -> dict[str, Any]:
    """Best single split of chapters skip_first+1..n into two segments, maximising between-segment distance.

    The first `skip_first` chapters are dropped, not just kept out of the split range. The reported chapter
    is the last one in the first segment (1-based), and the permutation null shuffles the kept chapters only.
    """
    rng = np.random.default_rng(rng_seed)
    vecs = ch_vecs[skip_first:]
    n = vecs.shape[0]

    def best_split(v: np.ndarray) -> tuple[int, float]:
        best_k, best_score = 2, -1.0
        for k in range(2, n - 1):
            m1 = v[:k].mean(axis=0)
            m2 = v[k:].mean(axis=0)
            m1 = m1 / max(np.linalg.norm(m1), 1e-8)
            m2 = m2 / max(np.linalg.norm(m2), 1e-8)
            score = float(1.0 - m1 @ m2)
            if score > best_score:
                best_k, best_score = k, score
        return best_k, best_score

    best_k, best_score = best_split(vecs)
    null_arr = np.array([best_split(vecs[rng.permutation(n)])[1] for _ in range(n_permutations)])
    return {
        "best_split_after_chapter": int(skip_first + best_k),
        "between_segment_distance": float(best_score),
        "permutation_p": float((null_arr >= best_score).mean()),
        "n_permutations": n_permutations,
        "skip_first": skip_first,
    }


def within_cross_similarity(
    ch_vecs: np.ndarray,
    split: int = 80,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(rng_seed)
    a = ch_vecs[:split]
    b = ch_vecs[split:]
    mat_aa = cosine_matrix(a)
    mat_bb = cosine_matrix(b)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    mat_ab = (a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)) @ b_norm.T
    aa_vals = mat_aa[np.triu_indices(len(a), k=1)]
    bb_vals = mat_bb[np.triu_indices(len(b), k=1)]
    ab_vals = mat_ab.ravel()

    def boot_ci(vals: np.ndarray) -> tuple[float, float, float]:
        means = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                          for _ in range(n_bootstrap)])
        return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    w1_mean, w1_lo, w1_hi = boot_ci(aa_vals)
    w2_mean, w2_lo, w2_hi = boot_ci(bb_vals)
    cr_mean, cr_lo, cr_hi = boot_ci(ab_vals)
    return {
        f"within_1_{split}": {"mean": w1_mean, "ci95_lo": w1_lo, "ci95_hi": w1_hi},
        f"within_{split+1}_120": {"mean": w2_mean, "ci95_lo": w2_lo, "ci95_hi": w2_hi},
        "cross": {"mean": cr_mean, "ci95_lo": cr_lo, "ci95_hi": cr_hi},
    }


def passage_level_bootstrap(
    passage_embs: np.ndarray,
    passage_chapters: list[int] | np.ndarray,
    split: int = 80,
    n_bootstrap: int = 500,
    rng_seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap over passages (not chapters) for cross and within-group cosine."""
    rng = np.random.default_rng(rng_seed)
    chaps = np.asarray(passage_chapters)
    idx_a = np.where(chaps <= split)[0]
    idx_b = np.where(chaps > split)[0]

    def sample_within(idx: np.ndarray) -> float:
        if len(idx) < 2:
            return float("nan")
        i, j = rng.choice(len(idx), size=2, replace=False)
        v1 = passage_embs[idx[i]]
        v2 = passage_embs[idx[j]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return float("nan")
        return float((v1 / n1) @ (v2 / n2))

    def sample_cross() -> float:
        i = rng.choice(len(idx_a))
        j = rng.choice(len(idx_b))
        v1 = passage_embs[idx_a[i]]
        v2 = passage_embs[idx_b[j]]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return float("nan")
        return float((v1 / n1) @ (v2 / n2))

    w1_samples = np.array([sample_within(idx_a) for _ in range(n_bootstrap)])
    w2_samples = np.array([sample_within(idx_b) for _ in range(n_bootstrap)])
    cr_samples = np.array([sample_cross() for _ in range(n_bootstrap)])

    def stats(s: np.ndarray) -> dict[str, float]:
        s = s[~np.isnan(s)]
        return {
            "mean": float(s.mean()),
            "ci95_lo": float(np.percentile(s, 2.5)),
            "ci95_hi": float(np.percentile(s, 97.5)),
        }

    return {
        f"within_1_{split}": stats(w1_samples),
        f"within_{split+1}_120": stats(w2_samples),
        "cross": stats(cr_samples),
    }


def logistic_probe_range(
    ch_vecs: np.ndarray,
    train_early: int = 60,
    train_late_start: int = 90,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X_train = np.vstack([ch_vecs[:train_early], ch_vecs[train_late_start:]])
    y_train = np.array([0] * train_early + [1] * (120 - train_late_start))
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_probe = scaler.transform(ch_vecs[train_early:])
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    clf.fit(X_train_sc, y_train)
    probs = clf.predict_proba(X_probe)[:, 1]
    flip_idx = None
    for i in range(len(probs) - 1):
        if probs[i] < 0.5 and probs[i + 1] >= 0.5:
            flip_idx = train_early + i + 1
            break
    return {
        "chapter_range": list(range(train_early + 1, 121)),
        "p_late": [float(p) for p in probs],
        "flip_chapter": flip_idx,
    }


# ---------------------------------------------------------------------------
# SECTION 8 – Placebo boundary analysis
# ---------------------------------------------------------------------------

def placebo_boundaries(
    ch_vecs: np.ndarray,
    b_range: range = range(41, 81),
    window: int = 10,
    train_offset: int = 11,
    train_width: int = 30,
) -> dict[int, dict[str, float]]:
    """For each boundary b, train logistic on [b-train_width-train_offset, b-train_offset]
    vs [b+train_offset, b+train_offset+train_width], score held-out [b-window, b+window].

    Returns dict b -> {auc, step_size}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    n = ch_vecs.shape[0]
    results: dict[int, dict[str, float]] = {}
    for b in b_range:
        # 0-indexed
        b0 = b - 1
        lo_train = b0 - train_offset - train_width
        hi_train_lo = b0 - train_offset
        lo_train_hi = b0 + train_offset
        hi_train = b0 + train_offset + train_width
        if lo_train < 0 or hi_train > n:
            continue
        X_train = np.vstack([
            ch_vecs[lo_train:hi_train_lo],
            ch_vecs[lo_train_hi:hi_train],
        ])
        y_train = np.array(
            [0] * (hi_train_lo - lo_train) + [1] * (hi_train - lo_train_hi)
        )
        if len(np.unique(y_train)) < 2:
            continue
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        # Held-out window around b
        hw_lo = max(0, b0 - window)
        hw_hi = min(n, b0 + window + 1)
        X_held = scaler.transform(ch_vecs[hw_lo:hw_hi])
        y_held = np.array([0 if i + hw_lo < b0 else 1 for i in range(hw_hi - hw_lo)])
        if len(np.unique(y_held)) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X_tr_sc, y_train)
            probs_held = clf.predict_proba(X_held)[:, 1]
            auc = float(roc_auc_score(y_held, probs_held))
            # Step size: mean(P for b+1..b+5) - mean(P for b-4..b)
            lo5 = max(0, b0 - 4) - hw_lo
            hi5 = min(len(probs_held), b0 + 5 - hw_lo)
            mid = b0 - hw_lo
            if mid > 0 and mid < len(probs_held):
                step = float(probs_held[mid:hi5].mean() - probs_held[max(0,lo5):mid].mean())
            else:
                step = float("nan")
        except Exception:
            auc = float("nan")
            step = float("nan")
        results[b] = {"auc": auc, "step_size": step}
    return results


def placebo_rank_and_zscore(
    placebo: dict[int, dict[str, float]],
    target_b: int = 80,
    metric: str = "auc",
) -> dict[str, Any]:
    vals = {b: v[metric] for b, v in placebo.items() if not np.isnan(v[metric])}
    if target_b not in vals:
        return {"rank": None, "zscore": None, "n": len(vals)}
    target_val = vals[target_b]
    sorted_vals = sorted(vals.values(), reverse=True)
    rank = sorted_vals.index(target_val) + 1
    others = [v for b, v in vals.items() if b != target_b]
    mu = float(np.mean(others))
    sigma = float(np.std(others))
    zscore = (target_val - mu) / max(sigma, 1e-8)
    return {"value": target_val, "rank": rank, "n_total": len(vals), "zscore": zscore,
            "mean_others": mu, "std_others": sigma}


# ---------------------------------------------------------------------------
# SECTION 9 – Covariate: dialogue share per chapter
# ---------------------------------------------------------------------------

def dialogue_share_per_chapter(
    passage_chapters: list[int],
    passage_dialogue: list[bool],
) -> np.ndarray:
    """Returns float array of shape [120] with dialogue fraction per chapter."""
    chaps = np.asarray(passage_chapters)
    dial = np.asarray(passage_dialogue, dtype=float)
    shares = np.zeros(120, dtype=float)
    for ch in range(1, 121):
        idx = np.where(chaps == ch)[0]
        if len(idx) > 0:
            shares[ch - 1] = dial[idx].mean()
    return shares


def regress_out_covariate(
    ch_vecs: np.ndarray,
    covariate: np.ndarray,
) -> np.ndarray:
    X = np.column_stack([np.ones(len(covariate)), covariate])
    beta, _, _, _ = np.linalg.lstsq(X, ch_vecs, rcond=None)
    residuals = ch_vecs - X @ beta
    norms = np.linalg.norm(residuals, axis=-1, keepdims=True)
    return residuals / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# SECTION 10 – Plotting
# ---------------------------------------------------------------------------

def _configure_cjk_fonts() -> None:
    import matplotlib
    from matplotlib import font_manager
    for c in [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if Path(c).exists():
            font_manager.fontManager.addfont(c)
            matplotlib.rcParams["font.family"] = [font_manager.FontProperties(fname=c).get_name()]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


def plot_heatmap(
    ch_vecs: np.ndarray,
    out_path: Path,
    title: str,
    split: int = 80,
) -> None:
    import matplotlib.pyplot as plt
    _configure_cjk_fonts()
    mat = cosine_matrix(ch_vecs)
    n = mat.shape[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.axhline(split - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.axvline(split - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Chapter")
    ax.set_ylabel("Chapter")
    ax.set_title(title)
    ticks = list(range(0, n, 20))
    ax.set_xticks(ticks); ax.set_xticklabels([str(t + 1) for t in ticks])
    ax.set_yticks(ticks); ax.set_yticklabels([str(t + 1) for t in ticks])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_logistic_curve(
    probe_result: dict[str, Any],
    out_path: Path,
    title: str,
    split: int = 80,
) -> None:
    import matplotlib.pyplot as plt
    _configure_cjk_fonts()
    chapters = probe_result["chapter_range"]
    probs = probe_result["p_late"]
    flip = probe_result["flip_chapter"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(chapters, probs, color="#2196F3", linewidth=1.5)
    ax.axhline(0.5, color="#999", linewidth=1.0, linestyle=":")
    ax.axvline(split, color="#E91E63", linewidth=1.5, linestyle="--", label=f"ch {split} boundary")
    if flip is not None:
        ax.axvline(flip, color="#FF9800", linewidth=1.5, linestyle="-.", label=f"flip at ch {flip}")
    ax.fill_between(chapters, probs, 0.5,
                    where=[p >= 0.5 for p in probs], alpha=0.15, color="#E91E63")
    ax.fill_between(chapters, probs, 0.5,
                    where=[p < 0.5 for p in probs], alpha=0.15, color="#2196F3")
    ax.set_xlabel("Chapter"); ax.set_ylabel("P(Gao E / late)")
    ax.set_title(title)
    ax.set_xlim(chapters[0], chapters[-1]); ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_placebo(
    placebo_results: dict[str, dict[int, dict[str, float]]],
    out_path: Path,
    metric: str = "auc",
) -> None:
    import matplotlib.pyplot as plt
    _configure_cjk_fonts()
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {
        "trained_full_lc": "#E91E63",
        "trained_dim_64_lc": "#9C27B0",
        "base_qwen3_lc": "#2196F3",
        "stylometry": "#4CAF50",
    }
    for label, placebo in placebo_results.items():
        bs = sorted(placebo.keys())
        vals = [placebo[b][metric] for b in bs]
        color = colors.get(label, "#999")
        ax.plot(bs, vals, label=label, color=color, linewidth=1.5, alpha=0.9)
        # Highlight b=80
        if 80 in placebo:
            ax.scatter([80], [placebo[80][metric]], color=color, s=80, zorder=5)
    ax.axvline(80, color="#FF5722", linewidth=1.5, linestyle="--", label="ch 80 boundary")
    ax.set_xlabel("Boundary b")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Placebo boundary scan: {metric} by boundary")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# SECTION 11 – Full analysis per method
# ---------------------------------------------------------------------------

def run_analysis(
    passage_embs: np.ndarray,
    passage_chapters: list[int],
    passage_lengths: list[int],
    passage_verse: list[bool],
    passage_dialogue: list[bool],
    out_dir: Path,
    label: str,
    split: int = 80,
    skip_cp_first: int = 0,
) -> dict[str, Any]:
    """Run all tests for one embedding method over verse/no-verse × raw/length-corrected."""
    results: dict[str, Any] = {}
    for verse_flag in [None, False]:
        suffix = "" if verse_flag is None else "_no_verse"
        mask = (np.ones(len(passage_chapters), dtype=bool)
                if verse_flag is None
                else np.array([not v for v in passage_verse]))
        for length_flag in [False, True]:
            run_label = label + suffix + ("_length_corrected" if length_flag else "")
            embs = passage_embs[mask]
            chaps = [c for c, m in zip(passage_chapters, mask) if m]
            lens = [l for l, m in zip(passage_lengths, mask) if m]
            if length_flag:
                embs = regress_out_length(embs, lens)
            ch_vecs = chapter_vectors(embs, chaps)
            # (a) change-point
            cp = change_point_test(ch_vecs, skip_first=skip_cp_first)
            cp_no_intro = change_point_test(ch_vecs, skip_first=5)
            # (b) heatmap
            heatmap_path = out_dir / f"heatmap_{run_label}.png"
            plot_heatmap(ch_vecs, heatmap_path, title=run_label, split=split)
            # (c) logistic probe
            lr = logistic_probe_range(ch_vecs)
            probe_plot = out_dir / f"logistic_{run_label}.png"
            plot_logistic_curve(lr, probe_plot, title=run_label, split=split)
            # (d) within/cross (chapter-level)
            wc = within_cross_similarity(ch_vecs, split=split)
            # (e) passage-level bootstrap
            pbs = passage_level_bootstrap(embs, chaps, split=split)
            results[run_label] = {
                "change_point": cp,
                "change_point_no_intro": cp_no_intro,
                "logistic_probe": {
                    "flip_chapter": lr["flip_chapter"],
                    "p_late_by_chapter": {
                        str(ch): float(p)
                        for ch, p in zip(lr["chapter_range"], lr["p_late"])
                    },
                },
                "within_cross": wc,
                "passage_bootstrap": pbs,
                "heatmap_path": str(heatmap_path),
                "logistic_plot_path": str(probe_plot),
            }
    return results


# ---------------------------------------------------------------------------
# SECTION 12 – Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="HLM authorship-change probe v2")
    parser.add_argument("--text", required=True, help="path to pg24264.txt")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--matryoshka", nargs="*", type=int, default=[64, 256])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--base-model", default="artifacts/base-models/qwen-3-0.6b")
    parser.add_argument("--save-embeddings", action="store_true",
                        help="save passage embeddings as float16 .npy to out/cache/")
    parser.add_argument("--from-cache", default=None,
                        help="load embeddings from this cache dir; skip model inference")
    args = parser.parse_args()

    import os
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    text_path = Path(args.text)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.from_cache) if args.from_cache else out_dir / "cache"

    # Resolve base model path (skip existence check when loading from cache)
    base_model_path = Path(args.base_model)
    if not base_model_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        base_model_path = repo_root / base_model_path
    if not args.from_cache and not base_model_path.exists():
        raise FileNotFoundError(f"base model not found: {base_model_path}")

    # -----------------------------------------------------------------------
    # Step 1: Parse and chunk
    # -----------------------------------------------------------------------
    print("Parsing chapters...")
    chapters = parse_chapters(text_path)
    print(f"  Parsed {len(chapters)} chapters")

    passage_texts: list[str] = []
    passage_chapters: list[int] = []
    passage_verse: list[bool] = []
    passage_dialogue: list[bool] = []

    for ch_num in range(1, 121):
        for chunk in chunk_chapter(chapters[ch_num]):
            passage_texts.append(chunk["text"])
            passage_chapters.append(ch_num)
            passage_verse.append(chunk["is_verse"])
            passage_dialogue.append(chunk["has_dialogue"])

    passage_lengths = [len(t) for t in passage_texts]
    n_verse = sum(passage_verse)
    n_dialogue = sum(passage_dialogue)
    print(f"  Total passages: {len(passage_texts)}")
    print(f"  Verse passages: {n_verse}")
    print(f"  Dialogue passages: {n_dialogue}")

    # Spot-check verse
    verse_samples = [(i, passage_texts[i]) for i, v in enumerate(passage_verse) if v][:10]
    print("\n  Verse spot-check (first 10 tagged):")
    for i, t in verse_samples:
        print(f"    ch{passage_chapters[i]:3d}: {repr(t[:60])}")

    # Name-masked texts
    passage_masked: list[str] = [mask_names(t) for t in passage_texts]

    # Save passage data
    passage_data = [
        {
            "chapter": passage_chapters[i],
            "text": passage_texts[i],
            "is_verse": passage_verse[i],
            "has_dialogue": passage_dialogue[i],
            "length": passage_lengths[i],
        }
        for i in range(len(passage_texts))
    ]
    (out_dir / "passages.json").write_text(
        json.dumps(passage_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.save_embeddings:
        save_passages_parquet(
            cache_dir, passage_chapters, passage_lengths,
            passage_verse, passage_dialogue, passage_texts,
        )

    all_results: dict[str, Any] = {
        "source": {
            "url": "https://www.gutenberg.org/ebooks/24264",
            "ebook": "Project Gutenberg #24264",
            "sha256": "ff1526996bf4b81807651921a85e5c1c0f1d1d123c9fa4553057ba6a3ec72011",
            "note": "Traditional/Cheng edition lineage. Mixed encoding.",
        },
        "n_chapters": len(chapters),
        "n_passages": len(passage_texts),
        "n_verse_passages": n_verse,
        "n_dialogue_passages": n_dialogue,
    }

    # Helper: get or compute embeddings
    def get_embs(label: str, compute_fn, *fn_args) -> np.ndarray:
        if args.from_cache:
            return load_embeddings(cache_dir, label)
        embs = compute_fn(*fn_args)
        if args.save_embeddings:
            save_embeddings(cache_dir, label, embs)
        return embs

    # -----------------------------------------------------------------------
    # Step 2: Stylometry (never needs the box)
    # -----------------------------------------------------------------------
    print("\nRunning stylometry...")
    sty_embs = stylometry_vectors(passage_texts)
    sty_masked_embs = stylometry_vectors(passage_masked)
    all_results["stylometry"] = run_analysis(
        sty_embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
        out_dir, "stylometry",
    )
    all_results["stylometry_masked"] = run_analysis(
        sty_masked_embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
        out_dir, "stylometry_masked",
    )
    print("  Stylometry done.")

    # -----------------------------------------------------------------------
    # Step 3: Base model
    # -----------------------------------------------------------------------
    if not args.from_cache:
        print("\nRunning base Qwen3-0.6B embedding...")
    base_embs = get_embs(
        "base_qwen3",
        base_model_embed, passage_texts, str(base_model_path), args.batch_size, args.device,
    )
    base_masked_embs = get_embs(
        "base_qwen3_masked",
        base_model_embed, passage_masked, str(base_model_path), args.batch_size, args.device,
    )
    all_results["base_qwen3"] = run_analysis(
        base_embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
        out_dir, "base_qwen3",
    )
    all_results["base_qwen3_masked"] = run_analysis(
        base_masked_embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
        out_dir, "base_qwen3_masked",
    )
    print("  Base model done.")

    # -----------------------------------------------------------------------
    # Step 4: Trained checkpoint
    # -----------------------------------------------------------------------
    if args.checkpoint or args.from_cache:
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                repo_root_p = Path(__file__).resolve().parents[3]
                checkpoint_path = repo_root_p / checkpoint_path
            print(f"\nRunning trained checkpoint from {checkpoint_path}...")

        def _trained_fn(texts, dims):
            return trained_model_embed(texts, checkpoint_path, dims, args.batch_size, args.device)

        # Full + Matryoshka dims
        if not args.from_cache:
            emb_dict = _trained_fn(passage_texts, args.matryoshka or [])
            emb_dict_masked = _trained_fn(passage_masked, args.matryoshka or [])
            if args.save_embeddings:
                for dim_label, embs in emb_dict.items():
                    save_embeddings(cache_dir, f"trained_{dim_label}", embs)
                for dim_label, embs in emb_dict_masked.items():
                    save_embeddings(cache_dir, f"trained_{dim_label}_masked", embs)
        else:
            # Load from cache
            emb_dict = {}
            emb_dict_masked = {}
            for dim_label in ["full"] + [f"dim_{d}" for d in (args.matryoshka or []) if d]:
                try:
                    emb_dict[dim_label] = load_embeddings(cache_dir, f"trained_{dim_label}")
                    emb_dict_masked[dim_label] = load_embeddings(cache_dir, f"trained_{dim_label}_masked")
                except FileNotFoundError:
                    pass

        for dim_label, embs in emb_dict.items():
            lbl = f"trained_{dim_label}"
            all_results[lbl] = run_analysis(
                embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
                out_dir, lbl,
            )
        for dim_label, embs in emb_dict_masked.items():
            lbl = f"trained_{dim_label}_masked"
            all_results[lbl] = run_analysis(
                embs, passage_chapters, passage_lengths, passage_verse, passage_dialogue,
                out_dir, lbl,
            )
        print("  Trained model done.")

    # -----------------------------------------------------------------------
    # Step 5: Placebo boundary analysis
    # -----------------------------------------------------------------------
    print("\nRunning placebo boundary analysis...")
    placebo_sets: dict[str, dict[int, dict[str, float]]] = {}

    def _run_placebo(label: str, embs: np.ndarray, chapters_list: list[int],
                     lengths_list: list[int], length_correct: bool) -> dict[int, dict[str, float]]:
        e = embs
        if length_correct:
            e = regress_out_length(e, lengths_list)
        cv = chapter_vectors(e, chapters_list)
        return placebo_boundaries(cv)

    placebo_sets["stylometry"] = _run_placebo("stylometry", sty_embs, passage_chapters, passage_lengths, False)
    placebo_sets["stylometry_lc"] = _run_placebo("stylometry_lc", sty_embs, passage_chapters, passage_lengths, True)
    placebo_sets["base_qwen3_lc"] = _run_placebo("base_qwen3_lc", base_embs, passage_chapters, passage_lengths, True)

    if "trained_full" in all_results and (emb_dict or args.from_cache):
        trained_full_embs = (emb_dict.get("full")
                             if not args.from_cache
                             else load_embeddings(cache_dir, "trained_full"))
        if trained_full_embs is not None:
            placebo_sets["trained_full_lc"] = _run_placebo(
                "trained_full_lc", trained_full_embs, passage_chapters, passage_lengths, True
            )
            placebo_sets["trained_full"] = _run_placebo(
                "trained_full", trained_full_embs, passage_chapters, passage_lengths, False
            )
        try:
            trained_64_embs = (emb_dict.get("dim_64")
                               if not args.from_cache
                               else load_embeddings(cache_dir, "trained_dim_64"))
            if trained_64_embs is not None:
                placebo_sets["trained_dim_64_lc"] = _run_placebo(
                    "trained_dim_64_lc", trained_64_embs, passage_chapters, passage_lengths, True
                )
        except Exception:
            pass

    # Rank and z-score for b=80
    placebo_stats: dict[str, Any] = {}
    for label, placebo in placebo_sets.items():
        placebo_stats[label] = {
            "auc": placebo_rank_and_zscore(placebo, 80, "auc"),
            "step_size": placebo_rank_and_zscore(placebo, 80, "step_size"),
        }
    all_results["placebo"] = {
        "by_boundary": {label: {str(b): v for b, v in pl.items()}
                        for label, pl in placebo_sets.items()},
        "b80_stats": placebo_stats,
    }

    # Placebo plots
    for metric in ["auc", "step_size"]:
        placebo_plot_path = out_dir / f"placebo_{metric}.png"
        plot_placebo(
            {k: v for k, v in placebo_sets.items() if k in
             ["stylometry", "base_qwen3_lc", "trained_full_lc", "trained_dim_64_lc"]},
            placebo_plot_path, metric=metric,
        )
    print("  Placebo done.")

    # -----------------------------------------------------------------------
    # Step 6: Dialogue share covariate + change-point without intro
    # -----------------------------------------------------------------------
    print("\nRunning dialogue covariate + change-point without intro...")
    dial_share = dialogue_share_per_chapter(passage_chapters, passage_dialogue)
    all_results["dialogue_share"] = {
        "per_chapter": [float(x) for x in dial_share],
        "mean_1_80": float(dial_share[:80].mean()),
        "mean_81_120": float(dial_share[80:].mean()),
    }

    # For each main embedding variant, run with dialogue regressed out + cp without 1-5
    dial_variants: dict[str, Any] = {}
    for method_label, raw_embs, lengths_list in [
        ("stylometry", sty_embs, passage_lengths),
        ("base_qwen3", base_embs, passage_lengths),
    ]:
        # Length-corrected chapter vectors
        lc_embs = regress_out_length(raw_embs, lengths_list)
        ch_vecs_lc = chapter_vectors(lc_embs, passage_chapters)
        # Regress out dialogue share from chapter vectors
        ch_vecs_dial = regress_out_covariate(ch_vecs_lc, dial_share)
        # CP without intro (skip first 5)
        cp5 = change_point_test(ch_vecs_lc, skip_first=5)
        lr5 = logistic_probe_range(ch_vecs_dial)
        dial_variants[f"{method_label}_lc_dial_corrected"] = {
            "change_point_no_intro": cp5,
            "logistic_probe_dial_corrected": {
                "flip_chapter": lr5["flip_chapter"],
                "p_late_by_chapter": {str(ch): float(p)
                                       for ch, p in zip(lr5["chapter_range"], lr5["p_late"])},
            },
        }

    if emb_dict.get("full") is not None or args.from_cache:
        try:
            tf_embs = (emb_dict["full"] if not args.from_cache
                       else load_embeddings(cache_dir, "trained_full"))
            tf_lc_embs = regress_out_length(tf_embs, passage_lengths)
            ch_vecs_tf_lc = chapter_vectors(tf_lc_embs, passage_chapters)
            ch_vecs_tf_dial = regress_out_covariate(ch_vecs_tf_lc, dial_share)
            cp5_tf = change_point_test(ch_vecs_tf_lc, skip_first=5)
            lr5_tf = logistic_probe_range(ch_vecs_tf_dial)
            dial_variants["trained_full_lc_dial_corrected"] = {
                "change_point_no_intro": cp5_tf,
                "logistic_probe_dial_corrected": {
                    "flip_chapter": lr5_tf["flip_chapter"],
                    "p_late_by_chapter": {str(ch): float(p)
                                           for ch, p in zip(lr5_tf["chapter_range"], lr5_tf["p_late"])},
                },
            }
        except Exception as e:
            print(f"  Dialogue variant for trained_full skipped: {e}")

    all_results["dialogue_covariate"] = dial_variants
    print("  Dialogue covariate done.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    # Per-method change-point and flip summary
    for method in ["stylometry", "base_qwen3", "trained_full", "trained_dim_64"]:
        if method not in all_results:
            continue
        print(f"\n--- {method} ---")
        for variant, res in sorted(all_results[method].items()):
            if not isinstance(res, dict) or "change_point" not in res:
                continue
            cp = res["change_point"]
            cp_ni = res.get("change_point_no_intro", {})
            wc = res["within_cross"]
            lr = res["logistic_probe"]
            w1k = [k for k in wc if k.startswith("within_1_")]
            w2k = [k for k in wc if k.startswith("within_81_")]
            w1 = wc.get(w1k[0], {}).get("mean", float("nan")) if w1k else float("nan")
            w2 = wc.get(w2k[0], {}).get("mean", float("nan")) if w2k else float("nan")
            cr = wc.get("cross", {}).get("mean", float("nan"))
            print(
                f"  [{variant}]\n"
                f"    cp:   ch {cp['best_split_after_chapter']}  dist={cp['between_segment_distance']:.4f}  p={cp['permutation_p']:.4f}\n"
                f"    cp>5: ch {cp_ni.get('best_split_after_chapter', '?')}  dist={cp_ni.get('between_segment_distance', float('nan')):.4f}  p={cp_ni.get('permutation_p', float('nan')):.4f}\n"
                f"    w1={w1:.4f}  w2={w2:.4f}  cross={cr:.4f}\n"
                f"    flip: ch {lr['flip_chapter']}"
            )

    # Placebo summary for b=80
    _nan = float("nan")
    print("\n--- b=80 placebo rank/z-score ---")
    for label, stats in all_results["placebo"]["b80_stats"].items():
        auc_s = stats["auc"]
        step_s = stats["step_size"]
        auc_val = auc_s.get("value") if auc_s.get("value") is not None else _nan
        auc_z = auc_s.get("zscore") if auc_s.get("zscore") is not None else _nan
        step_val = step_s.get("value") if step_s.get("value") is not None else _nan
        step_z = step_s.get("zscore") if step_s.get("zscore") is not None else _nan
        print(f"  {label:30s}  AUC={auc_val:.3f}  "
              f"rank={auc_s.get('rank')}/{auc_s.get('n_total')}  "
              f"z={auc_z:.2f}  "
              f"step={step_val:.3f}  "
              f"step_z={step_z:.2f}")

    # Verse count
    print("\n--- Verse ---")
    print(f"  Tagged: {n_verse} / {len(passage_texts)}")
    if verse_samples:
        print(f"  First tagged: ch{verse_samples[0][0]}: {repr(verse_samples[0][1][:60])}")

    # Dialogue
    ds = all_results["dialogue_share"]
    print("\n--- Dialogue share ---")
    print(f"  Mean ch 1-80:   {ds['mean_1_80']:.3f}")
    print(f"  Mean ch 81-120: {ds['mean_81_120']:.3f}")

    print("=" * 72)

    # Save (sanitize NaN/inf → null for JSON)
    def _sanitize(obj):
        if isinstance(obj, float):
            if obj != obj or obj == float("inf") or obj == float("-inf"):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    result_path = out_dir / "results.json"
    result_path.write_text(
        json.dumps(_sanitize(all_results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nAll results saved to {result_path}")
    print(f"Cache dir: {cache_dir}")
    if args.save_embeddings:
        print(f"Embeddings saved to {cache_dir}/")


if __name__ == "__main__":
    import json as _json_module
    json = _json_module
    main()
