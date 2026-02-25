#!/usr/bin/env python3
"""LDA generalization test: train on A group, evaluate on unseen B group."""

import sqlite3
import numpy as np
import requests
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

CACHE = Path(__file__).resolve().parent.parent / "cache"
DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
PROMPT = "分析这段文本的语言风格和说话习惯"

SPEAKERS_A = ["派蒙", "旅行者", "娜维娅", "纳西妲", "温迪", "阿贝多"]
SPEAKERS_B = ["提纳里", "荒泷一斗", "卡特皮拉", "那维莱特", "钟离"]
PER_SPEAKER = 25


def load_speakers(speakers):
    np.random.seed(42)
    conn = sqlite3.connect(str(DB_PATH))
    rows = []
    for s in speakers:
        r = conn.execute(
            "SELECT speaker, origin_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT ?", (s, PER_SPEAKER)
        ).fetchall()
        rows.extend(r)
        print(f"  {s}: {len(r)}")
    conn.close()
    sp2idx = {s: i for i, s in enumerate(speakers)}
    labels = np.array([sp2idx[r[0]] for r in rows])
    texts = [r[1] for r in rows]
    return texts, labels


def embed_openrouter(texts):
    inp = [f"Instruct: {PROMPT}\n{t}" for t in texts]
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(API_URL, headers=headers,
                      json={"model": MODEL, "input": inp})
    r.raise_for_status()
    return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)


def embed_cached(texts, cache_path):
    if cache_path.exists():
        print(f"  cache: {cache_path.name}")
        return np.load(cache_path)["embeddings"]
    print(f"  embedding {len(texts)} texts ...")
    embs = embed_openrouter(texts)
    np.savez_compressed(cache_path, embeddings=embs)
    return embs


def consistency(vecs, labels, speakers):
    cons = []
    for i in range(len(speakers)):
        v = vecs[labels == i]
        if len(v) < 2:
            cons.append(float('nan'))
            continue
        n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)
        c = n @ n.T
        cons.append(float(c[np.triu_indices(len(n), k=1)].mean()))
    return cons


def main():
    print("=== Loading A group ===")
    texts_a, labels_A = load_speakers(SPEAKERS_A)
    V_A = np.load(CACHE / "naive_8b_style_v2_origin.npz")["embeddings"]
    print(f"A: {V_A.shape}")

    print("\n=== Loading B group ===")
    texts_b, labels_B = load_speakers(SPEAKERS_B)
    V_B = embed_cached(texts_b, CACHE / "naive_8b_style_v2_groupB_origin.npz")
    print(f"B: {V_B.shape}")

    # 去均值（用 A 的均值）
    mean_A = V_A.mean(axis=0, keepdims=True)
    V_A_c = V_A - mean_A
    V_B_c = V_B - mean_A

    # LDA: fit on A, transform both
    lda = LinearDiscriminantAnalysis(n_components=min(len(SPEAKERS_A) - 1, 50))
    V_A_lda = lda.fit_transform(V_A_c, labels_A)
    V_B_lda = lda.transform(V_B_c)

    # Evaluate
    sil_A = silhouette_score(V_A_lda, labels_A, metric="cosine")
    sil_B = silhouette_score(V_B_lda, labels_B, metric="cosine")
    con_A = consistency(V_A_lda, labels_A, SPEAKERS_A)
    con_B = consistency(V_B_lda, labels_B, SPEAKERS_B)

    print(f"\n{'='*40}")
    print("A group (train)")
    print(f"{'='*40}")
    print(f"Silhouette: {sil_A:.4f}")
    for s, c in zip(SPEAKERS_A, con_A):
        print(f"  {s:<8} consistency={c:.4f}")
    print(f"  Avg: {np.nanmean(con_A):.4f}")

    print(f"\n{'='*40}")
    print("B group (unseen speakers)")
    print(f"{'='*40}")
    print(f"Silhouette: {sil_B:.4f}")
    for s, c in zip(SPEAKERS_B, con_B):
        print(f"  {s:<8} consistency={c:.4f}")
    print(f"  Avg: {np.nanmean(con_B):.4f}")


if __name__ == "__main__":
    main()
