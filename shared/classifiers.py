"""分离器：LDA + MLP ArcFace。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# ── ArcFace Head ──────────────────────────────────────────────────────

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


# ── 架构 ──────────────────────────────────────────────────────────────

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

def train_mlp_arcface(train_vecs, train_labels, n_classes, arch_name, style_dim, device, epochs=300, lr=1e-3, batch_size=256):
    """训练 MLP backbone + ArcFace head，返回训练好的 backbone。"""
    backbone = ARCHS[arch_name](train_vecs.shape[1], style_dim).to(device)
    head = ArcFaceHead(style_dim, n_classes).to(device)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X = torch.tensor(train_vecs, dtype=torch.float32)
    Y = torch.tensor(train_labels, dtype=torch.long)
    n = len(X)

    backbone.train()
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx].to(device)
            yb = Y[idx].to(device)

            emb = F.normalize(backbone(xb), dim=-1)
            logits = head(emb, yb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return backbone


def extract_mlp_vecs(backbone, vecs, device):
    """用训练好的 backbone 提取风格向量。"""
    backbone.eval()
    with torch.no_grad():
        t = torch.tensor(vecs, dtype=torch.float32).to(device)
        out = F.normalize(backbone(t), dim=-1).cpu().numpy()
    return out

