#!/usr/bin/env python3
"""Prompt residual experiment with OpenRouter qwen3-embedding-8b."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
PER_SPEAKER = 25
SPEAKERS = ["派蒙", "旅行者", "娜维娅", "纳西妲", "温迪", "阿贝多"]

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
            "ORDER BY RANDOM() LIMIT ?", (s, PER_SPEAKER)
        ).fetchall()
        all_rows.extend(rows)
        print(f"  {s}: {len(rows)}")
    conn.close()
    speakers_out, origins, paras = zip(*all_rows)
    print(f"Total: {len(all_rows)} rows")
    return list(origins), list(paras), list(speakers_out)


def embed_openrouter(texts, prompt=""):
    if prompt:
        texts = [f"Instruct: {prompt}\n{t}" for t in texts]
    headers = {"Authorization": f"Bearer {API_KEY}"}
    all_embs = []
    bs = 64
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        for attempt in range(3):
            try:
                r = requests.post(API_URL, headers=headers,
                                  json={"model": MODEL, "input": batch})
                r.raise_for_status()
                embs = [d["embedding"] for d in r.json()["data"]]
                all_embs.extend(embs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retry {attempt+1}: {e}")
                time.sleep(3)
    return np.array(all_embs, dtype=np.float32)


def embed_cached(texts, prompt_name, suffix):
    cache_path = CACHE_DIR / f"naive_8b_{prompt_name}_{suffix}.npz"
    if cache_path.exists():
        print(f"  cache: {cache_path.name}")
        return np.load(cache_path)["embeddings"]
    print(f"  embedding {len(texts)} texts [{prompt_name}] ...")
    embs = embed_openrouter(texts, PROMPTS[prompt_name])
    np.savez_compressed(cache_path, embeddings=embs)
    return embs


def project_out_content(R, V_s, variance_ratio=0.5):
    pca = PCA().fit(V_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, variance_ratio)) + 1
    PC = pca.components_[:k]
    return R - (R @ PC.T) @ PC, k, cumvar[k - 1]


def speaker_consistency(vecs):
    if len(vecs) < 2:
        return float('nan')
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.maximum(norms, 1e-10)
    cos_mat = normed @ normed.T
    return float(cos_mat[np.triu_indices(len(normed), k=1)].mean())


def evaluate(vecs, labels):
    cons = [speaker_consistency(vecs[labels == i]) for i in range(len(SPEAKERS))]
    sil = silhouette_score(vecs, labels, metric='cosine')
    return np.nanmean(cons), float(sil)


def main():
    np.random.seed(42)
    CACHE_DIR.mkdir(exist_ok=True)
    origins, paras, speaker_labels = load_data()
    sp2idx = {s: i for i, s in enumerate(SPEAKERS)}
    labels = np.array([sp2idx[s] for s in speaker_labels])

    summary = []
    for pname in PROMPTS:
        print(f"\n--- {pname}: {PROMPTS[pname]!r} ---")
        V_o = embed_cached(origins, pname, "origin")
        V_s = embed_cached(paras, pname, "para")

        # 直接用原文 embedding 聚类（无残差）
        ac0, si0 = evaluate(V_o, labels)
        summary.append((pname, "direct", ac0, si0))
        print(f"  direct         | con={ac0:.4f}  sil={si0:.4f}")

        # 残差
        R = V_o - V_s
        ac1, si1 = evaluate(R, labels)
        summary.append((pname, "raw", ac1, si1))
        print(f"  raw_residual   | con={ac1:.4f}  sil={si1:.4f}")

        # PCA 50%
        S, k, var = project_out_content(R, V_s)
        ac2, si2 = evaluate(S, labels)
        summary.append((pname, f"pca50(k={k})", ac2, si2))
        print(f"  pca50 (k={k:>3})  | con={ac2:.4f}  sil={si2:.4f}")

    print(f"\n{'='*62}")
    print("SUMMARY  (qwen3-embedding-8b, 6 speakers x 25)")
    print(f"{'='*62}")
    print(f"{'Prompt':<12} {'Mode':<14} {'Avg Con':>8} {'Silhou':>8}")
    print("-" * 46)
    for pname, mode, ac, si in summary:
        print(f"{pname:<12} {mode:<14} {ac:>8.4f} {si:>8.4f}")


if __name__ == "__main__":
    main()
