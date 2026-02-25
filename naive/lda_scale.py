#!/usr/bin/env python3
"""LDA generalization at scale: top 100 speakers, 4:1 train/test split."""

import sqlite3
import time
import numpy as np
import requests
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE = Path(__file__).resolve().parent.parent / "cache"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
PROMPT = "分析这段文本的语言风格和说话习惯"
PER_SPEAKER = 25


def get_top_speakers(n=100):
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT speaker, COUNT(*) as cnt FROM dialogues "
        "WHERE LENGTH(origin_text)>4 AND para_text IS NOT NULL "
        "AND LENGTH(TRIM(para_text))>0 "
        "GROUP BY speaker ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    speakers = [s for s, _ in rows if s != "？？？"][:n]
    print(f"Top {len(speakers)} speakers (excl ？？？)")
    return speakers


def load_texts(speakers):
    np.random.seed(42)
    conn = sqlite3.connect(str(DB_PATH))
    all_texts, all_labels = [], []
    for i, s in enumerate(speakers):
        rows = conn.execute(
            "SELECT origin_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT ?", (s, PER_SPEAKER)
        ).fetchall()
        all_texts.extend([r[0] for r in rows])
        all_labels.extend([i] * len(rows))
    conn.close()
    return all_texts, np.array(all_labels)


def embed_openrouter(texts, batch_size=64):
    all_embs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = [f"Instruct: {PROMPT}\n{t}" for t in texts[i:i + batch_size]]
        for attempt in range(3):
            try:
                r = requests.post(API_URL,
                                  headers={"Authorization": f"Bearer {API_KEY}"},
                                  json={"model": MODEL, "input": batch})
                r.raise_for_status()
                all_embs.extend([d["embedding"] for d in r.json()["data"]])
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retry {attempt+1}: {e}")
                time.sleep(3)
        bn = i // batch_size + 1
        if bn % 5 == 0 or bn == total:
            print(f"  {bn}/{total} batches")
    return np.array(all_embs, dtype=np.float32)


def embed_cached(texts, cache_name):
    path = CACHE / cache_name
    if path.exists():
        print(f"  cache: {cache_name}")
        return np.load(path)["embeddings"]
    print(f"  embedding {len(texts)} texts ...")
    embs = embed_openrouter(texts)
    np.savez_compressed(path, embeddings=embs)
    return embs


def consistency(vecs, labels, n_classes):
    cons = []
    for i in range(n_classes):
        v = vecs[labels == i]
        if len(v) < 2:
            cons.append(float('nan'))
            continue
        n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)
        c = n @ n.T
        cons.append(float(c[np.triu_indices(len(n), k=1)].mean()))
    return cons


def main():
    CACHE.mkdir(exist_ok=True)
    speakers = get_top_speakers(100)
    n_train = int(len(speakers) * 0.8)
    train_sp = speakers[:n_train]
    test_sp = speakers[n_train:]
    print(f"Train: {len(train_sp)} speakers, Test: {len(test_sp)} speakers")
    print(f"Test speakers: {', '.join(test_sp)}")

    # Load & embed
    print("\n--- Train set ---")
    texts_tr, labels_tr = load_texts(train_sp)
    V_tr = embed_cached(texts_tr, "lda100_train.npz")
    print(f"  shape: {V_tr.shape}")

    print("\n--- Test set ---")
    texts_te, labels_te = load_texts(test_sp)
    V_te = embed_cached(texts_te, "lda100_test.npz")
    print(f"  shape: {V_te.shape}")

    # Center with train mean
    mean = V_tr.mean(axis=0, keepdims=True)
    V_tr_c = V_tr - mean
    V_te_c = V_te - mean

    # LDA at multiple dimensions
    max_dim = len(train_sp) - 1
    DIMS = [d for d in [256, 128, 64, 32, 16] if d <= max_dim]
    print(f"\nLDA dims (max={max_dim}): {DIMS}")
    summary = []
    for n_comp in DIMS:
        lda = LinearDiscriminantAnalysis(n_components=n_comp, solver='eigen', shrinkage='auto')
        V_tr_lda = lda.fit_transform(V_tr_c, labels_tr)
        V_te_lda = lda.transform(V_te_c)

        sil_tr = silhouette_score(V_tr_lda, labels_tr, metric="cosine")
        sil_te = silhouette_score(V_te_lda, labels_te, metric="cosine")
        con_tr = np.nanmean(consistency(V_tr_lda, labels_tr, len(train_sp)))
        con_te = np.nanmean(consistency(V_te_lda, labels_te, len(test_sp)))
        summary.append((n_comp, sil_tr, con_tr, sil_te, con_te))
        print(f"  dim={n_comp:<4} train sil={sil_tr:.4f} con={con_tr:.4f}  |  test sil={sil_te:.4f} con={con_te:.4f}")

    print(f"\n{'='*70}")
    print(f"SUMMARY  (train={len(train_sp)} speakers, test={len(test_sp)} unseen)")
    print(f"{'='*70}")
    print(f"{'Dim':>5} {'Train Sil':>10} {'Train Con':>10} {'Test Sil':>10} {'Test Con':>10}")
    print("-" * 48)
    for dim, str_, ctr, ste, cte in summary:
        print(f"{dim:>5} {str_:>10.4f} {ctr:>10.4f} {ste:>10.4f} {cte:>10.4f}")


if __name__ == "__main__":
    main()
