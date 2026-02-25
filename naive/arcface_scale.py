#!/usr/bin/env python3
"""ArcFace style extractor at scale: 177 speakers, tiered sampling."""

import sqlite3
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from pathlib import Path
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE = Path(__file__).resolve().parent.parent / "cache"
API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
PROMPT = "分析这段文本的语言风格和说话习惯"
DEVICE = torch.device("mps")
EPOCHS = 500
LR = 1e-3
BATCH = 512


def get_speakers_and_samples():
    """Return [(speaker, sample_n), ...] sorted by count desc, excl ？？？"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT speaker, COUNT(*) as cnt FROM dialogues "
        "WHERE LENGTH(origin_text)>4 AND para_text IS NOT NULL "
        "AND LENGTH(TRIM(para_text))>0 GROUP BY speaker ORDER BY cnt DESC"
    ).fetchall()
    conn.close()
    result = []
    for s, c in rows:
        if s == "？？？" or c < 200:
            continue
        if c >= 10000:
            result.append((s, 1000))
        elif c >= 1000:
            result.append((s, 150))
        else:
            result.append((s, 50))
    return result


def load_texts_split(speaker_samples, val_ratio=0.2):
    """Load texts and split per-speaker into train/val."""
    np.random.seed(42)
    conn = sqlite3.connect(str(DB_PATH))
    tr_texts, tr_labels, val_texts, val_labels = [], [], [], []
    for i, (s, n) in enumerate(speaker_samples):
        rows = conn.execute(
            "SELECT origin_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT ?", (s, n)
        ).fetchall()
        texts = [r[0] for r in rows]
        split = max(1, int(len(texts) * (1 - val_ratio)))
        tr_texts.extend(texts[:split])
        tr_labels.extend([i] * split)
        val_texts.extend(texts[split:])
        val_labels.extend([i] * (len(texts) - split))
    conn.close()
    return tr_texts, np.array(tr_labels), val_texts, np.array(val_labels)


def load_texts(speaker_samples):
    np.random.seed(42)
    conn = sqlite3.connect(str(DB_PATH))
    all_texts, all_labels = [], []
    for i, (s, n) in enumerate(speaker_samples):
        rows = conn.execute(
            "SELECT origin_text FROM dialogues "
            "WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 "
            "ORDER BY RANDOM() LIMIT ?", (s, n)
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
        for attempt in range(5):
            try:
                r = requests.post(API_URL,
                                  headers={"Authorization": f"Bearer {API_KEY}"},
                                  json={"model": MODEL, "input": batch})
                r.raise_for_status()
                all_embs.extend([d["embedding"] for d in r.json()["data"]])
                break
            except Exception as e:
                if attempt == 4:
                    raise
                print(f"  retry {attempt+1}: {e}")
                time.sleep(5)
        bn = i // batch_size + 1
        if bn % 10 == 0 or bn == total:
            print(f"  {bn}/{total} batches", flush=True)
    return np.array(all_embs, dtype=np.float32)


def embed_cached(texts, cache_name):
    path = CACHE / cache_name
    if path.exists():
        print(f"  cache: {cache_name}")
        return np.load(path)["embeddings"]
    print(f"  embedding {len(texts)} texts ...")
    embs = embed_openrouter(texts)
    np.savez_compressed(path, embeddings=embs)
    print(f"  saved {cache_name} {embs.shape}")
    return embs


class ArcFaceHead(nn.Module):
    def __init__(self, in_dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        W = F.normalize(self.W, dim=1)
        cos = x @ W.T
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, cos.size(1)).float()
        return self.s * torch.cos(theta + one_hot * self.m)


def evaluate(backbone, V, labels, n_classes):
    backbone.eval()
    with torch.no_grad():
        out = F.normalize(backbone(torch.tensor(V, device=DEVICE)), dim=-1).cpu().numpy()
    cons = []
    for i in range(n_classes):
        v = out[labels == i]
        if len(v) < 2:
            cons.append(float('nan'))
            continue
        n = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-10)
        c = n @ n.T
        cons.append(float(c[np.triu_indices(len(n), k=1)].mean()))
    sil = silhouette_score(out, labels, metric='cosine')
    return np.nanmean(cons), float(sil)


def train_and_eval(arch_fn, style_dim, n_train_cls,
                   V_tr, labels_tr, V_val, labels_val, V_te, labels_te,
                   n_test_cls):
    backbone = arch_fn(V_tr.shape[1], style_dim).to(DEVICE)
    head = ArcFaceHead(style_dim, n_train_cls).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    V_t = torch.tensor(V_tr, device=DEVICE)
    L_t = torch.tensor(labels_tr, dtype=torch.long, device=DEVICE)

    for epoch in range(EPOCHS):
        backbone.train()
        head.train()
        idx = torch.randperm(len(V_t), device=DEVICE)[:BATCH]
        emb = F.normalize(backbone(V_t[idx]), dim=-1)
        logits = head(emb, L_t[idx])
        loss = loss_fn(logits, L_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f"    epoch {epoch+1} loss={loss.item():.4f}", flush=True)

    con_tr, sil_tr = evaluate(backbone, V_tr, labels_tr, n_train_cls)
    con_val, sil_val = evaluate(backbone, V_val, labels_val, n_train_cls)
    con_te, sil_te = evaluate(backbone, V_te, labels_te, n_test_cls)
    return sil_tr, con_tr, sil_val, con_val, sil_te, con_te


ARCHS = {
    "linear": lambda d_in, d_out: nn.Sequential(nn.Linear(d_in, d_out)),
    "1h-512": lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, d_out)),
    "2h-1k":  lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 1024), nn.ReLU(), nn.Linear(1024, 512),
        nn.ReLU(), nn.Linear(512, d_out)),
    "1h-2k":  lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 2048), nn.ReLU(), nn.Linear(2048, d_out)),
}
STYLE_DIMS = [32, 64, 128]


def main():
    CACHE.mkdir(exist_ok=True)
    sp_samples = get_speakers_and_samples()
    n_total = len(sp_samples)
    n_train = int(n_total * 0.8)
    train_ss = sp_samples[:n_train]
    test_ss = sp_samples[n_train:]

    print(f"Speakers: {n_total} total, train={len(train_ss)}, test={len(test_ss)}")

    # Load full cached embeddings
    V_all_tr = np.load(CACHE / "arcface_scale_train.npz")["embeddings"]
    V_te = np.load(CACHE / "arcface_scale_test.npz")["embeddings"]

    # Split train cache per-speaker into train/val (80/20)
    tr_idx, val_idx, tr_labels, val_labels = [], [], [], []
    offset = 0
    for i, (s, n) in enumerate(train_ss):
        split = max(1, int(n * 0.8))
        tr_idx.extend(range(offset, offset + split))
        tr_labels.extend([i] * split)
        val_idx.extend(range(offset + split, offset + n))
        val_labels.extend([i] * (n - split))
        offset += n
    V_tr = V_all_tr[tr_idx]
    V_val = V_all_tr[val_idx]
    tr_labels = np.array(tr_labels)
    val_labels = np.array(val_labels)

    # Test labels
    te_labels = []
    for i, (s, n) in enumerate(test_ss):
        te_labels.extend([i] * n)
    te_labels = np.array(te_labels)

    print(f"  train: {V_tr.shape}, val: {V_val.shape}, test: {V_te.shape}")

    # Train & eval
    results = []
    for arch_name, arch_fn in ARCHS.items():
        for sd in STYLE_DIMS:
            print(f"\n{arch_name} dim={sd}", flush=True)
            sil_tr, con_tr, sil_val, con_val, sil_te, con_te = train_and_eval(
                arch_fn, sd, len(train_ss),
                V_tr, tr_labels, V_val, val_labels, V_te, te_labels, len(test_ss))
            print(f"  => tr  sil={sil_tr:.4f} con={con_tr:.4f}")
            print(f"     val sil={sil_val:.4f} con={con_val:.4f}")
            print(f"     te  sil={sil_te:.4f} con={con_te:.4f}")
            results.append((arch_name, sd, sil_tr, con_tr, sil_val, con_val, sil_te, con_te))

    print(f"\n{'='*80}")
    print(f"SUMMARY (ArcFace, {len(train_ss)} train / {len(test_ss)} test speakers)")
    print(f"{'='*80}")
    print(f"{'Arch':<8} {'Dim':>4} {'Tr Sil':>8} {'Tr Con':>8} {'Va Sil':>8} {'Va Con':>8} {'Te Sil':>8} {'Te Con':>8}")
    print("-" * 72)
    for arch, sd, str_, ctr, sv, cv, ste, cte in results:
        print(f"{arch:<8} {sd:>4} {str_:>8.4f} {ctr:>8.4f} {sv:>8.4f} {cv:>8.4f} {ste:>8.4f} {cte:>8.4f}")


if __name__ == "__main__":
    main()
