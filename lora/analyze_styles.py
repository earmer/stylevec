"""
风格向量分析：对所有 ≥100 句说话人生成风格向量，
计算簇内平均距离、最近异类距离，并用多种聚类算法估计可区分风格数。

用法：
    cd lora/
    python analyze_styles.py            # 首次运行编码并缓存
    python analyze_styles.py --no-cache # 强制重新编码
"""

import argparse
import csv
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import StyleModel, MODEL_PATH

# ── 配置 ──────────────────────────────────────────────────────────────────────
TRAINED_DIR  = Path(__file__).resolve().parent / "trained"
DB_PATH      = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
EMB_CACHE    = Path(__file__).resolve().parent / "analysis_embeddings.pkl"
RESULTS_CSV  = Path(__file__).resolve().parent / "style_analysis.csv"
PLOTS_DIR    = Path(__file__).resolve().parent / "plots"

MAX_PER_SPEAKER = 100
MIN_SENTENCES   = 100
MAX_LEN         = 128
EXCLUDE         = ("？？？", "？？")


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = detect_device()
BATCH_SIZE = 128 if DEVICE.type == "cuda" else 64


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_speakers() -> dict[str, list[str]]:
    """从 DB 加载每个说话人的句子，随机取 MAX_PER_SPEAKER 句。"""
    placeholders = ",".join("?" for _ in EXCLUDE)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        f"SELECT speaker FROM dialogues "
        f"WHERE speaker NOT IN ({placeholders}) "
        f"AND LENGTH(origin_text) > 4 AND origin_text IS NOT NULL "
        f"GROUP BY speaker HAVING COUNT(*) >= ? ORDER BY speaker",
        (*EXCLUDE, MIN_SENTENCES),
    ).fetchall()

    rng = np.random.RandomState(42)
    speakers: dict[str, list[str]] = {}
    for (name,) in rows:
        texts = [r[0] for r in conn.execute(
            "SELECT origin_text FROM dialogues "
            "WHERE speaker = ? AND LENGTH(origin_text) > 4 AND origin_text IS NOT NULL",
            (name,),
        ).fetchall()]
        idx = rng.permutation(len(texts))[:MAX_PER_SPEAKER]
        speakers[name] = [texts[i] for i in idx]

    conn.close()
    return speakers


# ── 模型加载 ──────────────────────────────────────────────────────────────────
def load_model() -> StyleModel:
    """加载训练好的 LoRA 模型，自动检测 num_train_speakers。"""
    from safetensors.torch import load_file

    arcface_state = torch.load(TRAINED_DIR / "arcface_head.pt", map_location="cpu")
    num_spk = arcface_state["W"].shape[0]
    print(f"  ArcFace 头检测到 {num_spk} 个训练说话人")

    # adapter_config.json 中 r=64, alpha=128
    model = StyleModel(num_train_speakers=num_spk, lora_r=64, lora_alpha=128)
    adapter_weights = load_file(str(TRAINED_DIR / "adapter_model.safetensors"))
    model.base.load_state_dict(adapter_weights, strict=False)
    model.style_head.load_state_dict(
        torch.load(TRAINED_DIR / "style_head.pt", map_location="cpu")
    )
    model.arcface_head.load_state_dict(arcface_state)
    model = model.to(DEVICE).eval()
    return model


# ── 编码 ──────────────────────────────────────────────────────────────────────
class _TextList(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        return self.texts[i]


def encode_all(
    model: StyleModel, tokenizer, speakers: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """批量编码所有说话人，返回 {speaker: (N, 128) ndarray}。"""
    names = list(speakers.keys())
    all_texts, boundaries = [], [0]
    for name in names:
        all_texts.extend(speakers[name])
        boundaries.append(len(all_texts))

    def collate(batch):
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    loader = DataLoader(
        _TextList(all_texts), batch_size=BATCH_SIZE,
        collate_fn=collate, num_workers=0, pin_memory=DEVICE.type == "cuda",
    )

    all_vecs = []
    done = 0
    with torch.no_grad():
        for ids, mask in loader:
            v = model.encode(ids.to(DEVICE), mask.to(DEVICE))
            all_vecs.append(v.cpu().float().numpy())
            done += len(ids)
            if done % (BATCH_SIZE * 10) == 0:
                print(f"  编码进度: {done}/{len(all_texts)}")

    all_vecs = np.concatenate(all_vecs)  # (total, 128)
    print(f"  编码完成: {all_vecs.shape[0]} 句")

    embeddings = {}
    for i, name in enumerate(names):
        embeddings[name] = all_vecs[boundaries[i]:boundaries[i + 1]]
    return embeddings


# ── 逐人指标 ──────────────────────────────────────────────────────────────────
def per_speaker_metrics(embeddings: dict[str, np.ndarray]):
    """计算每个说话人的簇内平均距离和最近异类距离。"""
    names = list(embeddings.keys())
    S = len(names)

    # 簇内平均距离（严格两两配对余弦距离）
    intra_dists = np.empty(S)
    for i, name in enumerate(names):
        vecs = embeddings[name]  # (N, 128) L2-normalized
        n = len(vecs)
        sim = vecs @ vecs.T  # (N, N)
        # mean pairwise cosine distance = 1 - mean(off-diagonal sim)
        intra_dists[i] = 1.0 - (sim.sum() - n) / (n * (n - 1))

    # centroid 矩阵
    centroids = np.stack([embeddings[n].mean(0) for n in names])  # (S, 128)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
    centroids = centroids / norms  # 归一化

    # centroid 间余弦距离矩阵
    dist_mat = 1.0 - centroids @ centroids.T  # (S, S)
    np.fill_diagonal(dist_mat, np.inf)

    nearest_idx = dist_mat.argmin(axis=1)
    nearest_dists = dist_mat[np.arange(S), nearest_idx]

    results = []
    for i, name in enumerate(names):
        results.append({
            "speaker": name,
            "n": len(embeddings[name]),
            "intra_dist": float(intra_dists[i]),
            "nearest_dist": float(nearest_dists[i]),
            "nearest_name": names[nearest_idx[i]],
            "separability": float(nearest_dists[i] / (intra_dists[i] + 1e-9)),
        })

    return results, centroids, names


# ── 聚类分析 ──────────────────────────────────────────────────────────────────
def clustering_analysis(centroids: np.ndarray, names: list[str]) -> dict:
    """在说话人 centroid 上运行多种聚类算法。"""
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.metrics.pairwise import cosine_distances
    from scipy.cluster.hierarchy import linkage

    X = centroids  # 已归一化
    S = len(names)
    k_range = list(range(2, min(31, S)))

    # ── K-Means + Silhouette ──
    print("\n=== K-Means 轮廓系数 (K=2..30) ===")
    sil_scores, ch_scores, inertias = [], [], []
    km_labels_map = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        km_labels_map[k] = labels
        sil = silhouette_score(X, labels, metric="cosine")
        ch = calinski_harabasz_score(X, labels)
        sil_scores.append(sil)
        ch_scores.append(ch)
        inertias.append(km.inertia_)
        if k <= 10 or k % 5 == 0:
            print(f"  K={k:3d}: silhouette={sil:.4f}  CH={ch:.1f}")

    best_k_sil = k_range[int(np.argmax(sil_scores))]
    best_k_ch = k_range[int(np.argmax(ch_scores))]
    print(f"  最优 K (轮廓系数): {best_k_sil} (sil={max(sil_scores):.4f})")
    print(f"  最优 K (CH 指数):  {best_k_ch} (CH={max(ch_scores):.1f})")

    # ── GMM + BIC ──
    print("\n=== GMM BIC (K=2..30) ===")
    bic_scores = []
    for k in k_range:
        n_features = X.shape[1]
        cov = "full" if S > k * (n_features + 1) else "diag"
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type=cov)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    best_k_bic = k_range[int(np.argmin(bic_scores))]
    print(f"  最优 K (BIC 最小): {best_k_bic}")

    # ── 层次聚类 (Ward) ──
    print("\n=== 层次聚类 (Ward) ===")
    Z = linkage(X, method="ward")
    last_n = min(30, S - 1)
    last_merges = Z[-(last_n):, 2]
    accel = np.diff(last_merges, 2)
    best_k_ward = last_n - int(np.argmax(accel))
    print(f"  merge distance elbow 建议 K: {best_k_ward}")

    # ── DBSCAN ──
    print("\n=== DBSCAN (cosine 距离) ===")
    D = cosine_distances(X)
    dbscan_results = []
    for eps in np.arange(0.05, 0.41, 0.05):
        db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
        labels = db.fit_predict(D)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        dbscan_results.append((eps, n_clusters, n_noise))
        print(f"  eps={eps:.2f}: {n_clusters:3d} 簇, {n_noise:3d} 噪声点")

    return {
        "k_range": k_range,
        "sil_scores": sil_scores,
        "ch_scores": ch_scores,
        "bic_scores": bic_scores,
        "inertias": inertias,
        "best_k_sil": best_k_sil,
        "best_k_ch": best_k_ch,
        "best_k_bic": best_k_bic,
        "best_k_ward": best_k_ward,
        "dbscan_results": dbscan_results,
        "km_labels_map": km_labels_map,
        "linkage_Z": Z,
    }


# ── 可视化 ────────────────────────────────────────────────────────────────────
def make_plots(
    centroids: np.ndarray,
    names: list[str],
    cluster_info: dict,
):
    """生成三张图保存到 plots/ 目录。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy.cluster.hierarchy import dendrogram

    rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    PLOTS_DIR.mkdir(exist_ok=True)

    k_range = cluster_info["k_range"]
    sil_scores = cluster_info["sil_scores"]

    # 1) Silhouette vs K
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, sil_scores, "o-", markersize=4)
    best_k = cluster_info["best_k_sil"]
    best_idx = k_range.index(best_k)
    ax.axvline(best_k, color="red", ls="--", alpha=0.6,
               label=f"best K={best_k} (sil={sil_scores[best_idx]:.4f})")
    ax.set_xlabel("K")
    ax.set_ylabel("Cosine Silhouette Score")
    ax.set_title("K-Means Silhouette vs K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "silhouette_vs_k.png", dpi=150)
    plt.close(fig)
    print(f"  保存: {PLOTS_DIR / 'silhouette_vs_k.png'}")

    # 2) Dendrogram (truncated)
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram(
        cluster_info["linkage_Z"],
        truncate_mode="lastp", p=50,
        leaf_rotation=90, leaf_font_size=8, ax=ax,
    )
    ax.set_title("Hierarchical Clustering (Ward, last 50 merges)")
    ax.set_ylabel("Merge Distance")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "dendrogram.png", dpi=150)
    plt.close(fig)
    print(f"  保存: {PLOTS_DIR / 'dendrogram.png'}")

    # 3) t-SNE of centroids, colored by K-Means best-K
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, metric="cosine", random_state=42, perplexity=30)
    coords = tsne.fit_transform(centroids)  # (S, 2)
    labels = cluster_info["km_labels_map"][best_k]

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", s=30, alpha=0.8,
    )
    # 标注部分说话人名（每簇最多3个，避免拥挤）
    from collections import Counter
    labeled_per_cluster = Counter()
    for i, name in enumerate(names):
        cl = labels[i]
        if labeled_per_cluster[cl] < 3:
            ax.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=6, alpha=0.7)
            labeled_per_cluster[cl] += 1
    ax.set_title(f"t-SNE of Speaker Centroids (K-Means K={best_k})")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tsne_centroids.png", dpi=150)
    plt.close(fig)
    print(f"  保存: {PLOTS_DIR / 'tsne_centroids.png'}")


# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="强制重新编码")
    args = parser.parse_args()

    print(f"设备: {DEVICE}")

    # 1. 加载数据
    print("\n[1/5] 加载说话人数据...")
    speakers = load_speakers()
    total_sents = sum(len(v) for v in speakers.values())
    print(f"  说话人: {len(speakers)}, 总句数: {total_sents}")

    # 2. 编码（或从缓存加载）
    if EMB_CACHE.exists() and not args.no_cache:
        print(f"\n[2/5] 从缓存加载嵌入: {EMB_CACHE}")
        with open(EMB_CACHE, "rb") as f:
            embeddings = pickle.load(f)
        print(f"  缓存中 {len(embeddings)} 个说话人")
    else:
        print("\n[2/5] 加载模型并编码...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), trust_remote_code=True
        )
        model = load_model()
        embeddings = encode_all(model, tokenizer, speakers)
        del model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
        with open(EMB_CACHE, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"  嵌入已缓存到 {EMB_CACHE}")

    # 3. 逐人指标
    print("\n[3/5] 计算逐人指标...")
    results, centroids, spk_names = per_speaker_metrics(embeddings)
    results.sort(key=lambda x: x["separability"], reverse=True)

    print(f"\n{'说话人':<10} {'N':>4} {'簇内距离':>10} {'最近异类':>10} "
          f"{'最近说话人':<10} {'可分性':>8}")
    print("-" * 62)
    for r in results[:30]:
        print(f"{r['speaker']:<10} {r['n']:>4} {r['intra_dist']:>10.4f} "
              f"{r['nearest_dist']:>10.4f} {r['nearest_name']:<10} "
              f"{r['separability']:>8.2f}")
    if len(results) > 30:
        print(f"  ... 共 {len(results)} 个说话人，完整结果见 CSV")

    intra = [r["intra_dist"] for r in results]
    nearest = [r["nearest_dist"] for r in results]
    print(f"\n簇内距离  mean={np.mean(intra):.4f}  median={np.median(intra):.4f}")
    print(f"最近异类  mean={np.mean(nearest):.4f}  median={np.median(nearest):.4f}")

    # 4. 聚类分析
    print("\n[4/5] 聚类分析...")
    cluster_info = clustering_analysis(centroids, spk_names)

    # 5. 可视化
    print("\n[5/5] 生成可视化...")
    make_plots(centroids, spk_names, cluster_info)

    # 保存 CSV
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["speaker", "n", "intra_dist", "nearest_dist",
                  "nearest_name", "separability"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["speaker"]):
            writer.writerow(r)
    print(f"\n逐人指标已保存到 {RESULTS_CSV}")

    # 总结
    print("\n" + "=" * 50)
    print("总结")
    print("=" * 50)
    print(f"分析说话人数:       {len(results)}")
    print(f"K-Means 最优 K:     {cluster_info['best_k_sil']} (silhouette)")
    print(f"K-Means 最优 K:     {cluster_info['best_k_ch']} (CH index)")
    print(f"GMM BIC 最优 K:     {cluster_info['best_k_bic']}")
    print(f"层次聚类建议 K:     {cluster_info['best_k_ward']}")


if __name__ == "__main__":
    main()
