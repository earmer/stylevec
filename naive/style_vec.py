#!/usr/bin/env python3
"""Residual style vector extraction and validation."""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).parent / "genshin" / "genshin.db"
CACHE_DIR = Path(__file__).parent / "cache"
SPEAKERS = ["派蒙", "旅行者", "娜维娅", "纳西妲", "温迪",
            "阿贝多", "茜特菈莉", "八重神子", "玛拉妮", "芙宁娜"]
MODELS = [("qwen3-embedding:0.6b", 1024), ("embeddinggemma:latest", 768)]
OLLAMA_URL = "http://localhost:11434/api/embed"


def load_data(db_path, speakers, per_speaker=100):
    conn = sqlite3.connect(str(db_path))
    all_rows = []
    for s in speakers:
        rows = conn.execute(
            "SELECT speaker, origin_text, para_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT ?", (s, per_speaker)
        ).fetchall()
        all_rows.extend(rows)
        print(f"  {s}: {len(rows)}")
    conn.close()
    speakers_out, origins, paras = zip(*all_rows)
    print(f"Total: {len(all_rows)} rows ({per_speaker}/speaker)")
    return list(origins), list(paras), list(speakers_out)


def embed_batch(texts, model, batch_size=256):
    all_embs = []
    total = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(3):
            try:
                r = requests.post(OLLAMA_URL, json={"model": model, "input": batch})
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
            print(f"  [{model}] {bn}/{total} batches")
    return np.array(all_embs, dtype=np.float32)


def embed_all(texts, model, cache_path):
    if cache_path.exists():
        print(f"Loading cache: {cache_path.name}")
        return np.load(cache_path)["embeddings"]
    print(f"Embedding {len(texts)} texts with {model}...")
    embs = embed_batch(texts, model)
    np.savez_compressed(cache_path, embeddings=embs)
    print(f"Cached to {cache_path.name} ({embs.shape})")
    return embs


def project_out_content(R, V_s, variance_ratio=0.9):
    pca = PCA().fit(V_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_ratio)) + 1
    PC = pca.components_[:k]
    style = R - (R @ PC.T) @ PC
    return style, k, cumvar[k - 1]


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


def evaluate(style_vecs, labels, speakers, sample_sil=10000):
    results = {}
    for i, s in enumerate(speakers):
        mask = labels == i
        results[s] = (int(mask.sum()), speaker_consistency(style_vecs[mask]))
    # silhouette on subsample
    if len(style_vecs) > sample_sil:
        idx = np.random.choice(len(style_vecs), sample_sil, replace=False)
        sil = silhouette_score(style_vecs[idx], labels[idx], metric='cosine')
    else:
        sil = silhouette_score(style_vecs, labels, metric='cosine')
    results["__silhouette__"] = float(sil)
    return results


def print_run(model, mode, pca_info, results, speakers):
    print(f"\nModel: {model} | Mode: {mode}")
    if pca_info:
        print(f"PCA components: {pca_info[0]} ({pca_info[1]:.1%} variance)")
    print(f"{'Speaker':<10} {'Count':>6}  {'Consistency':>12}")
    print("-" * 32)
    consistencies = []
    for s in speakers:
        cnt, con = results[s]
        print(f"{s:<10} {cnt:>6}  {con:>12.4f}")
        consistencies.append(con)
    print("-" * 32)
    print(f"Avg consistency: {np.nanmean(consistencies):.4f}")
    print(f"Silhouette (cosine): {results['__silhouette__']:.4f}")


def main():
    np.random.seed(42)
    origins, paras, speaker_labels = load_data(DB_PATH, SPEAKERS)
    sp2idx = {s: i for i, s in enumerate(SPEAKERS)}
    labels = np.array([sp2idx[s] for s in speaker_labels])

    all_runs = []
    for model, dim in MODELS:
        safe = model.replace(":", "_").replace("/", "_")
        V_o = embed_all(origins, model, CACHE_DIR / f"{safe}_origin.npz")
        V_s = embed_all(paras, model, CACHE_DIR / f"{safe}_para.npz")
        R = V_o - V_s

        # Run 1: raw residual
        print(f"\nEvaluating {model} / raw_residual...")
        m1 = evaluate(R, labels, SPEAKERS)
        print_run(model, "raw_residual", None, m1, SPEAKERS)
        all_runs.append((model, "raw_residual", np.nanmean([m1[s][1] for s in SPEAKERS]), m1["__silhouette__"]))

        # Run 2: PCA projected
        print(f"\nEvaluating {model} / pca_projected...")
        S, k, var = project_out_content(R, V_s)
        m2 = evaluate(S, labels, SPEAKERS)
        print_run(model, "pca_projected", (k, var), m2, SPEAKERS)
        all_runs.append((model, "pca_projected", np.nanmean([m2[s][1] for s in SPEAKERS]), m2["__silhouette__"]))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Run':<40} {'Avg Con':>8} {'Silhou':>8}")
    print("-" * 58)
    for model, mode, avg_con, sil in all_runs:
        print(f"{model} / {mode:<16} {avg_con:>8.4f} {sil:>8.4f}")


if __name__ == "__main__":
    main()
