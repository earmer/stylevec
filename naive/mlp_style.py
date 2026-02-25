#!/usr/bin/env python3
"""MLP style extractor: train with triplet loss, evaluate generalization."""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import silhouette_score

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
CACHE = Path(__file__).resolve().parent.parent / "cache"
DEVICE = torch.device("mps")
EPOCHS = 200
LR = 1e-3
MARGIN = 0.3
BATCH_TRIPLETS = 512

ARCHS = {
    "small":  lambda d_in, d_out: nn.Sequential(nn.Linear(d_in, d_out)),
    "medium": lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Linear(512, d_out)),
    "large":  lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 1024), nn.ReLU(), nn.Linear(1024, 512),
        nn.ReLU(), nn.Linear(512, d_out)),
    "wide":   lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 2048), nn.ReLU(), nn.Linear(2048, d_out)),
}
STYLE_DIMS = [16, 32, 64, 128]


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


def mine_triplets(embs, labels, n_triplets):
    """Online hard-ish triplet mining."""
    classes = np.unique(labels)
    cls2idx = {c: np.where(labels == c)[0] for c in classes}
    anchors, positives, negatives = [], [], []
    for _ in range(n_triplets):
        c = np.random.choice(classes)
        neg_c = np.random.choice([x for x in classes if x != c])
        idx = cls2idx[c]
        neg_idx = cls2idx[neg_c]
        a, p = np.random.choice(idx, 2, replace=len(idx) < 2)
        n = np.random.choice(neg_idx)
        anchors.append(a)
        positives.append(p)
        negatives.append(n)
    return np.array(anchors), np.array(positives), np.array(negatives)


def evaluate(model, V, labels, n_classes):
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(V, device=DEVICE)).cpu().numpy()
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


def train_and_eval(arch_name, arch_fn, style_dim, V_tr, labels_tr, V_te, labels_te):
    model = nn.Sequential(
        arch_fn(V_tr.shape[1], style_dim),
        nn.Identity(),  # normalize in forward
    ).to(DEVICE)

    class Wrapper(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            return nn.functional.normalize(self.net(x), dim=-1)

    model = Wrapper(arch_fn(V_tr.shape[1], style_dim)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.TripletMarginLoss(margin=MARGIN)
    V_tr_t = torch.tensor(V_tr, device=DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        ai, pi, ni = mine_triplets(V_tr_t.cpu().numpy(), labels_tr, BATCH_TRIPLETS)
        a = model(V_tr_t[ai])
        p = model(V_tr_t[pi])
        n = model(V_tr_t[ni])
        loss = loss_fn(a, p, n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    con_tr, sil_tr = evaluate(model, V_tr, labels_tr, len(np.unique(labels_tr)))
    con_te, sil_te = evaluate(model, V_te, labels_te, len(np.unique(labels_te)))
    return sil_tr, con_tr, sil_te, con_te


def main():
    train_sp, test_sp = get_speakers()
    labels_tr = load_labels(train_sp)
    labels_te = load_labels(test_sp)
    V_tr = np.load(CACHE / "lda100_train.npz")["embeddings"]
    V_te = np.load(CACHE / "lda100_test.npz")["embeddings"]
    print(f"Train: {V_tr.shape}, Test: {V_te.shape}")
    print(f"Device: {DEVICE}\n")

    results = []
    for arch_name in ARCHS:
        for sd in STYLE_DIMS:
            print(f"Training {arch_name} dim={sd} ...", end=" ", flush=True)
            sil_tr, con_tr, sil_te, con_te = train_and_eval(
                arch_name, ARCHS[arch_name], sd, V_tr, labels_tr, V_te, labels_te)
            print(f"train sil={sil_tr:.4f} con={con_tr:.4f} | test sil={sil_te:.4f} con={con_te:.4f}")
            results.append((arch_name, sd, sil_tr, con_tr, sil_te, con_te))

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    print(f"{'Arch':<8} {'Dim':>4} {'Tr Sil':>8} {'Tr Con':>8} {'Te Sil':>8} {'Te Con':>8}")
    print("-" * 48)
    for arch, sd, str_, ctr, ste, cte in results:
        print(f"{arch:<8} {sd:>4} {str_:>8.4f} {ctr:>8.4f} {ste:>8.4f} {cte:>8.4f}")


if __name__ == "__main__":
    main()
