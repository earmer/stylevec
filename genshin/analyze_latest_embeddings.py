from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="genshin/genshin.db")
    parser.add_argument("--checkpoint", default="paper_replication/checkpoints/latest")
    parser.add_argument("--model-name", default="base-models/xlm-roberta-base")
    parser.add_argument("--out-dir", default="genshin/embedding_analysis/latest_20260505-005138")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--label-top-speakers", type=int, default=30)
    parser.add_argument("--projection-sample", type=int, default=0, help="0 means project all rows.")
    parser.add_argument("--cosine-pairs", type=int, default=1_000_000)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--no-tsne", action="store_true")
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--umap-components", type=int, default=30)
    parser.add_argument("--umap-epochs", type=int, default=200)
    return parser.parse_args()


def configure_threads(jobs: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, str(jobs))
    torch.set_num_threads(jobs)


def load_rows(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT id, speaker, origin_text, para_text
            FROM dialogues
            WHERE origin_text IS NOT NULL AND origin_text <> ''
              AND para_text IS NOT NULL AND para_text <> ''
            ORDER BY id
            """,
            conn,
        )
    return df


def load_model(model_name: str, checkpoint: Path, device: torch.device):
    base = AutoModel.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def mean_pool_normalize(outputs, attention_mask: torch.Tensor) -> torch.Tensor:
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1)


def encode_texts(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_len: int,
    label: str,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    use_cuda = device.type == "cuda"
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                outputs = model(**enc)
                embs = mean_pool_normalize(outputs, enc["attention_mask"])
        chunks.append(embs.float().cpu().numpy())
        done = min(start + batch_size, len(texts))
        print(f"{label}: encoded {done}/{len(texts)}", flush=True)
    return np.concatenate(chunks, axis=0)


def load_or_encode(
    path: Path,
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_len: int,
    label: str,
    overwrite: bool,
) -> np.ndarray:
    if path.exists() and not overwrite:
        print(f"{label}: loading existing embeddings from {path}", flush=True)
        return np.load(path)
    embeddings = encode_texts(texts, model, tokenizer, device, batch_size, max_len, label)
    np.save(path, embeddings)
    return embeddings


def choose_projection_rows(df: pd.DataFrame, max_rows: int) -> np.ndarray:
    if max_rows <= 0 or max_rows >= len(df):
        return np.arange(len(df))
    # Preserve common speakers while keeping the scatter bounded.
    return (
        df.groupby("speaker", group_keys=False)
        .sample(frac=max_rows / len(df), random_state=42)
        .index.to_numpy()
    )[:max_rows]


def pca_reduce(matrix: np.ndarray, components: int) -> np.ndarray:
    n_components = min(components, matrix.shape[1], matrix.shape[0] - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    return np.ascontiguousarray(pca.fit_transform(matrix).astype(np.float32, copy=False))


def run_umap(matrix: np.ndarray, jobs: int, neighbors: int, min_dist: float, epochs: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="euclidean",
        init="random",
        n_epochs=epochs,
        n_jobs=jobs,
        low_memory=False,
        angular_rp_forest=False,
        verbose=True,
    )
    return reducer.fit_transform(matrix)


def run_tsne(matrix50: np.ndarray, jobs: int) -> np.ndarray:
    try:
        from openTSNE import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            metric="euclidean",
            initialization="pca",
            negative_gradient_method="fft",
            n_jobs=jobs,
            random_state=42,
            verbose=True,
        )
        return np.asarray(tsne.fit(matrix50))
    except Exception as exc:
        print(f"openTSNE failed ({exc}); falling back to sklearn TSNE", flush=True)
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            n_jobs=jobs,
            random_state=42,
            verbose=2,
        )
        return tsne.fit_transform(matrix50)


def save_projection(path: Path, df: pd.DataFrame, coords: np.ndarray, rows: np.ndarray) -> None:
    out = df.iloc[rows][["id", "speaker"]].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out.to_csv(path, index=False)


def plot_projection(
    path: Path,
    df: pd.DataFrame,
    coords: np.ndarray,
    rows: np.ndarray,
    title: str,
    label_speakers: bool,
    top_n: int,
) -> None:
    view = df.iloc[rows].copy()
    speaker_counts = df["speaker"].value_counts()
    top = set(speaker_counts.head(top_n).index)
    ranks = view["speaker"].map(lambda s: speaker_counts.index.get_loc(s) if s in speaker_counts.index else top_n)
    colors = np.where(view["speaker"].isin(top), ranks, top_n)

    plt.figure(figsize=(14, 12), dpi=180)
    plt.scatter(coords[:, 0], coords[:, 1], c=colors, s=2, alpha=0.45, cmap="tab20", linewidths=0)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])

    if label_speakers:
        for speaker in speaker_counts.head(top_n).index:
            mask = view["speaker"].to_numpy() == speaker
            if not mask.any():
                continue
            xy = coords[mask]
            x, y = np.median(xy[:, 0]), np.median(xy[:, 1])
            plt.text(
                x,
                y,
                speaker,
                fontsize=8,
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def project_and_plot(
    name: str,
    matrix: np.ndarray,
    df: pd.DataFrame,
    rows: np.ndarray,
    out_dir: Path,
    jobs: int,
    label_speakers: bool,
    top_n: int,
    skip_tsne: bool,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_components: int,
    umap_epochs: int,
) -> None:
    selected = matrix[rows]
    print(f"{name}: PCA input {selected.shape}", flush=True)
    matrix_reduced = pca_reduce(selected, umap_components)

    print(f"{name}: running UMAP with n_jobs={jobs}", flush=True)
    umap_coords = run_umap(matrix_reduced, jobs, umap_neighbors, umap_min_dist, umap_epochs)
    umap_stem = "umap-original-embedding-with-name-tag" if name == "original" else "umap-para-embedding"
    save_projection(out_dir / f"{umap_stem}.csv", df, umap_coords, rows)
    plot_projection(out_dir / f"{umap_stem}.png", df, umap_coords, rows, f"{name} UMAP", label_speakers, top_n)

    if skip_tsne:
        return

    print(f"{name}: running t-SNE with n_jobs={jobs}", flush=True)
    tsne_coords = run_tsne(matrix_reduced, jobs)
    tsne_stem = "tsne-original-embedding-with-name-tag" if name == "original" else "tsne-para-embedding"
    save_projection(out_dir / f"{tsne_stem}.csv", df, tsne_coords, rows)
    plot_projection(out_dir / f"{tsne_stem}.png", df, tsne_coords, rows, f"{name} t-SNE", label_speakers, top_n)


def cosine_statistics(matrix: np.ndarray, num_pairs: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    if n < 2:
        raise ValueError("Need at least two embeddings for within-set cosine statistics.")
    left = rng.integers(0, n, size=num_pairs)
    right = rng.integers(0, n - 1, size=num_pairs)
    right = right + (right >= left)
    cosines = np.einsum("ij,ij->i", matrix[left], matrix[right])
    return {
        "sampled_pairs": int(num_pairs),
        "mean": float(cosines.mean()),
        "median": float(np.median(cosines)),
        "std": float(cosines.std()),
        "min": float(cosines.min()),
        "p01": float(np.quantile(cosines, 0.01)),
        "p05": float(np.quantile(cosines, 0.05)),
        "p25": float(np.quantile(cosines, 0.25)),
        "p75": float(np.quantile(cosines, 0.75)),
        "p95": float(np.quantile(cosines, 0.95)),
        "p99": float(np.quantile(cosines, 0.99)),
        "max": float(cosines.max()),
    }


def main() -> None:
    args = parse_args()
    configure_threads(args.jobs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(Path(args.db))
    print(f"rows={len(df)} speakers={df['speaker'].nunique()}", flush=True)
    df[["id", "speaker", "origin_text", "para_text"]].to_parquet(out_dir / "rows.parquet", index=False)

    original_path = out_dir / "original_embeddings.npy"
    para_path = out_dir / "para_embeddings.npy"
    embeddings_ready = original_path.exists() and para_path.exists() and not args.overwrite_embeddings

    if embeddings_ready:
        model = tokenizer = None
        device = torch.device("cpu")
        print("embedding files found; skipping model load and GPU encoding", flush=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = load_model(args.model_name, Path(args.checkpoint), device)
        print(f"device={device} checkpoint={args.checkpoint}", flush=True)

    original = load_or_encode(
        original_path,
        df["origin_text"].tolist(),
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_len,
        "original",
        args.overwrite_embeddings,
    )
    para = load_or_encode(
        para_path,
        df["para_text"].tolist(),
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_len,
        "para",
        args.overwrite_embeddings,
    )

    original_stats = cosine_statistics(original, args.cosine_pairs, seed=42)
    para_stats = cosine_statistics(para, args.cosine_pairs, seed=43)
    (out_dir / "cosine-statics-original-embeddings.json").write_text(
        json.dumps(original_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "cosine-statics-para-embeddings.json").write_text(
        json.dumps(para_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = {
        "checkpoint": str(args.checkpoint),
        "rows": int(len(df)),
        "speakers": int(df["speaker"].nunique()),
        "original_embedding_cosine_statistics": original_stats,
        "para_embedding_cosine_statistics": para_stats,
        "jobs": int(args.jobs),
        "batch_size": int(args.batch_size),
        "projection_sample": int(args.projection_sample),
        "no_tsne": bool(args.no_tsne),
        "umap_neighbors": int(args.umap_neighbors),
        "umap_min_dist": float(args.umap_min_dist),
        "umap_components": int(args.umap_components),
        "umap_epochs": int(args.umap_epochs),
        "diagrams": [
            "umap-original-embedding-with-name-tag.png",
            "umap-para-embedding.png",
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = choose_projection_rows(df, args.projection_sample)
    np.save(out_dir / "projection_rows.npy", rows)
    project_and_plot(
        "original",
        original,
        df,
        rows,
        out_dir,
        args.jobs,
        True,
        args.label_top_speakers,
        args.no_tsne,
        args.umap_neighbors,
        args.umap_min_dist,
        args.umap_components,
        args.umap_epochs,
    )
    project_and_plot(
        "para",
        para,
        df,
        rows,
        out_dir,
        args.jobs,
        False,
        args.label_top_speakers,
        args.no_tsne,
        args.umap_neighbors,
        args.umap_min_dist,
        args.umap_components,
        args.umap_epochs,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
