#!/usr/bin/env python3
"""MLP style extractor with ArcFace loss."""

import sqlite3
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE = Path(__file__).resolve().parent.parent / "cache"
DEVICE = torch.device("mps")
EPOCHS = 300
LR = 1e-3
BATCH = 256


class ArcFaceHead(nn.Module):
    def __init__(self, in_dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(n_classes, in_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, labels):
        # x: L2-normalized embeddings
        W = F.normalize(self.W, dim=1)
        cos = x @ W.T
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        one_hot = F.one_hot(labels, cos.size(1)).float()
        logits = self.s * torch.cos(theta + one_hot * self.m)
        return logits


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
STYLE_DIMS = [16, 32, 64, 128]


def get_speakers():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT speaker, COUNT(*) FROM dialogues "
        "WHERE LENGTH(origin_text)>4 AND para_text IS NOT NULL "
        "AND LENGTH(TRIM(para_text))>0 GROUP BY speaker ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    sp = [s for s, _ in rows if s != "？？？"][:100]
    return sp[:80], sp[80:]


def load_labels(speakers):
    np.random.seed(42)
    conn = sqlite3.connect(str(DB_PATH))
    labels = []
    for i, s in enumerate(speakers):
        n = len(conn.execute(
            "SELECT 1 FROM dialogues WHERE speaker=? AND LENGTH(origin_text)>4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text))>0 LIMIT 25", (s,)
        ).fetchall())
        labels.extend([i] * n)
    conn.close()
    return np.array(labels)


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


def train_and_eval(arch_fn, style_dim, n_train_cls, V_tr, labels_tr, V_te, labels_te):
    backbone = arch_fn(V_tr.shape[1], style_dim).to(DEVICE)
    head = ArcFaceHead(style_dim, n_train_cls).to(DEVICE)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=LR)
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

    con_tr, sil_tr = evaluate(backbone, V_tr, labels_tr, n_train_cls)
    con_te, sil_te = evaluate(backbone, V_te, labels_te, len(np.unique(labels_te)))
    return sil_tr, con_tr, sil_te, con_te


def main():
    train_sp, test_sp = get_speakers()
    labels_tr = load_labels(train_sp)
    labels_te = load_labels(test_sp)
    V_tr = np.load(CACHE / "lda100_train.npz")["embeddings"]
    V_te = np.load(CACHE / "lda100_test.npz")["embeddings"]
    print(f"Train: {V_tr.shape}, Test: {V_te.shape}")
    print(f"Train classes: {len(train_sp)}, Test classes: {len(test_sp)}")
    print(f"Device: {DEVICE}\n")

    results = []
    for arch_name, arch_fn in ARCHS.items():
        for sd in STYLE_DIMS:
            print(f"{arch_name} dim={sd} ...", end=" ", flush=True)
            sil_tr, con_tr, sil_te, con_te = train_and_eval(
                arch_fn, sd, len(train_sp), V_tr, labels_tr, V_te, labels_te)
            print(f"tr sil={sil_tr:.4f} con={con_tr:.4f} | te sil={sil_te:.4f} con={con_te:.4f}")
            results.append((arch_name, sd, sil_tr, con_tr, sil_te, con_te))

    print(f"\n{'='*72}")
    print("SUMMARY (ArcFace, 80 train / 20 test speakers)")
    print(f"{'='*72}")
    print(f"{'Arch':<8} {'Dim':>4} {'Tr Sil':>8} {'Tr Con':>8} {'Te Sil':>8} {'Te Con':>8}")
    print("-" * 48)
    for arch, sd, str_, ctr, ste, cte in results:
        print(f"{arch:<8} {sd:>4} {str_:>8.4f} {ctr:>8.4f} {ste:>8.4f} {cte:>8.4f}")


if __name__ == "__main__":
    main()
