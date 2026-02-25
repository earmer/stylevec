"""分离器：LDA + MLP ArcFace。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

DEVICE = torch.device("mps")
EPOCHS = 300
LR = 1e-3
BATCH = 256


# ── ArcFace Head（复用 naive/mlp_arcface.py） ──────────────────────────

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
        logits = self.s * torch.cos(theta + one_hot * self.m)
        return logits


# ── 架构（裁剪自 naive/mlp_arcface.py，输入 1024 维） ─────────────────

ARCHS = {
    "linear": lambda d_in, d_out: nn.Sequential(nn.Linear(d_in, d_out)),
    "1h-256": lambda d_in, d_out: nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, d_out)),
}
STYLE_DIMS = [32, 64]


# ── LDA ───────────────────────────────────────────────────────────────

def run_lda(train_vecs, train_labels, val_vecs, gen_vecs):
    """LDA 降维，返回变换后的 (train, val, gen) 向量。"""
    n_classes = len(set(train_labels.tolist()))
    n_components = min(n_classes - 1, train_vecs.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(train_vecs, train_labels)
    return lda.transform(train_vecs), lda.transform(val_vecs), lda.transform(gen_vecs)


# ── MLP + ArcFace ─────────────────────────────────────────────────────

def train_mlp_arcface(train_vecs, train_labels, n_classes, arch_name, style_dim):
    """训练 MLP backbone + ArcFace head，返回训练好的 backbone。"""
    backbone = ARCHS[arch_name](train_vecs.shape[1], style_dim).to(DEVICE)
    head = ArcFaceHead(style_dim, n_classes).to(DEVICE)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.tensor(train_vecs, dtype=torch.float32)
    Y = torch.tensor(train_labels, dtype=torch.long)
    n = len(X)

    backbone.train()
    head.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb = X[idx].to(DEVICE)
            yb = Y[idx].to(DEVICE)

            emb = F.normalize(backbone(xb), dim=-1)
            logits = head(emb, yb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    return backbone


def extract_mlp_vecs(backbone, vecs):
    """用训练好的 backbone 提取风格向量。"""
    backbone.eval()
    with torch.no_grad():
        t = torch.tensor(vecs, dtype=torch.float32).to(DEVICE)
        out = F.normalize(backbone(t), dim=-1).cpu().numpy()
    return out
