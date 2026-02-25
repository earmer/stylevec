#!/usr/bin/env python3
"""Residual style vectors with instruction-aware embedding prompts."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
SPEAKERS = ["派蒙", "旅行者", "娜维娅", "纳西妲", "温迪",
            "阿贝多", "茜特菈莉", "八重神子", "玛拉妮", "芙宁娜"]

PROMPTS = {
    "baseline":  "",
    "semantic":  "为这个句子生成表示以用于语义相似度计算",
    "style_v1":  "这段文本的风格如何",
    "style_v2":  "分析这段文本的语言风格和说话习惯",
    "style_v3":  "忽略文本内容，仅关注说话方式和语气特征",
    "style_v4":  "判断这段文本的说话人是谁",
    "cluster":   "为这个句子生成表示以用于按说话风格聚类",
}


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    all_rows = []
    for s in SPEAKERS:
        rows = conn.execute(
            "SELECT speaker, origin_text, para_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT 100", (s,)
        ).fetchall()
        all_rows.extend(rows)
        print(f"  {s}: {len(rows)}")
    conn.close()
    speakers_out, origins, paras = zip(*all_rows)
    print(f"Total: {len(all_rows)} rows")
    return list(origins), list(paras), list(speakers_out)


def embed_batch(texts, prompt="", batch_size=256):
    if prompt:
        texts = [f"Instruct: {prompt}\n{t}" for t in texts]
    all_embs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                r = requests.post(OLLAMA_URL, json={"model": MODEL, "input": batch})
                r.raise_for_status()
                all_embs.extend(r.json()["embeddings"])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retry {attempt+1}: {e}")
                time.sleep(2)
        bn = i // batch_size + 1
        if bn % 50 == 0 or bn == total:
            print(f"  [{MODEL}] {bn}/{total} batches")
    return np.array(all_embs, dtype=np.float32)


def embed_cached(texts, prompt_name, suffix):
    safe = MODEL.replace(":", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"naive_{safe}_{prompt_name}_{suffix}.npz"
    if cache_path.exists():
        print(f"Loading cache: {cache_path.name}")
        return np.load(cache_path)["embeddings"]
    prompt = PROMPTS[prompt_name]
    print(f"Embedding {len(texts)} texts [{prompt_name}] ...")
    embs = embed_batch(texts, prompt)
    np.savez_compressed(cache_path, embeddings=embs)
    print(f"Cached to {cache_path.name} ({embs.shape})")
    return embs


def project_out_content(R, V_s, variance_ratio=0.5):
    pca = PCA().fit(V_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_ratio)) + 1
    PC = pca.components_[:k]
    return R - (R @ PC.T) @ PC, k, cumvar[k - 1]


def speaker_consistency(vecs, sample_n=100):
    if len(vecs) < 2:
        return float('nan')
    if len(vecs) > sample_n:
        idx = np.random.choice(len(vecs), sample_n, replace=False)
        vecs = vecs[idx]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.maximum(norms, 1e-10)
    cos_mat = normed @ normed.T
    return float(cos_mat[np.triu_indices(len(normed), k=1)].mean())


def evaluate(style_vecs, labels):
    consistencies = []
    for i in range(len(SPEAKERS)):
        mask = labels == i
        consistencies.append(speaker_consistency(style_vecs[mask]))
    sil = silhouette_score(style_vecs, labels, metric='cosine')
    return np.nanmean(consistencies), float(sil)


def main():
    np.random.seed(42)
    CACHE_DIR.mkdir(exist_ok=True)
    origins, paras, speaker_labels = load_data()
    sp2idx = {s: i for i, s in enumerate(SPEAKERS)}
    labels = np.array([sp2idx[s] for s in speaker_labels])

    summary = []
    for pname in PROMPTS:
        print(f"\n{'='*50}\nPrompt: {pname} = {PROMPTS[pname]!r}\n{'='*50}")
        V_o = embed_cached(origins, pname, "origin")
        V_s = embed_cached(paras, pname, "para")
        R = V_o - V_s

        avg_con, sil = evaluate(R, labels)
        summary.append((pname, "raw", avg_con, sil))
        print(f"  raw_residual     | consistency={avg_con:.4f}  silhouette={sil:.4f}")

        S, k, var = project_out_content(R, V_s)
        avg_con2, sil2 = evaluate(S, labels)
        summary.append((pname, f"pca50(k={k})", avg_con2, sil2))
        print(f"  pca50 (k={k:>3},{var:.0%}) | consistency={avg_con2:.4f}  silhouette={sil2:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Prompt':<12} {'Mode':<14} {'Avg Con':>8} {'Silhou':>8}")
    print("-" * 46)
    for pname, mode, ac, si in summary:
        print(f"{pname:<12} {mode:<14} {ac:>8.4f} {si:>8.4f}")


if __name__ == "__main__":
    main()
