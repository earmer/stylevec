from __future__ import annotations

import argparse
import json
import os
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


LANG_COLORS = {
    "en": "#1f77b4",
    "zh": "#d62728",
    "ja": "#2ca02c",
    "fr": "#9467bd",
    "ru": "#ff7f0e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/msynthstel/data/translated")
    parser.add_argument("--checkpoint", default="paper_replication/checkpoints/remote_20260505-005138_step-195633")
    parser.add_argument("--model-name", default="base-models/xlm-roberta-base")
    parser.add_argument("--out-dir", default="datasets/msynthstel/embedding_analysis/latest_20260505-005138")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cosine-pairs", type=int, default=500_000)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)
    parser.add_argument("--umap-components", type=int, default=30)
    parser.add_argument("--umap-epochs", type=int, default=200)
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--joint-umap", action="store_true")
    return parser.parse_args()


def configure_threads(jobs: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, str(jobs))
    torch.set_num_threads(jobs)


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    frames = []
    for lang_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        path = lang_dir / f"{split}-00000-of-00001.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["positive"].notna() & df["negative"].notna()].copy()
        df["lang"] = lang_dir.name
        df["source_row"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {split} parquet files under {data_dir}")

    pairs = pd.concat(frames, ignore_index=True)
    positive = pairs[["lang", "source_row", "feature", "feature_clean", "positive"]].rename(columns={"positive": "text"})
    positive["polarity"] = "positive"
    negative = pairs[["lang", "source_row", "feature", "feature_clean", "negative"]].rename(columns={"negative": "text"})
    negative["polarity"] = "negative"
    rows = pd.concat([positive, negative], ignore_index=True)
    rows = rows[rows["text"].notna() & (rows["text"] != "")].copy()
    rows.insert(0, "sample_id", np.arange(len(rows)))
    return rows


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
    chunks = []
    use_cuda = device.type == "cuda"
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                outputs = model(**enc)
                embeddings = mean_pool_normalize(outputs, enc["attention_mask"])
        chunks.append(embeddings.float().cpu().numpy())
        print(f"{label}: encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.concatenate(chunks, axis=0)


def load_or_encode(
    path: Path,
    rows: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_len: int,
    split: str,
    overwrite: bool,
) -> np.ndarray:
    if path.exists() and not overwrite:
        print(f"{split}: loading existing embeddings from {path}", flush=True)
        return np.load(path)
    embeddings = encode_texts(rows["text"].tolist(), model, tokenizer, device, batch_size, max_len, split)
    np.save(path, embeddings)
    return embeddings


def random_cosines(matrix: np.ndarray, num_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = matrix.shape[0]
    left = rng.integers(0, n, size=num_pairs)
    right = rng.integers(0, n - 1, size=num_pairs)
    right = right + (right >= left)
    return np.einsum("ij,ij->i", matrix[left], matrix[right])


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def cosine_diagnostics(rows: pd.DataFrame, matrix: np.ndarray, num_pairs: int, seed: int) -> dict:
    stats = {
        "rows": int(len(rows)),
        "sampled_pairs": int(num_pairs),
        "random_all": summarize(random_cosines(matrix, num_pairs, seed)),
    }
    for polarity in ("positive", "negative"):
        idx = np.flatnonzero(rows["polarity"].to_numpy() == polarity)
        stats[f"random_{polarity}"] = summarize(random_cosines(matrix[idx], min(num_pairs, len(idx) * 50), seed + len(idx)))

    mean = matrix.mean(axis=0)
    centered = matrix - mean
    centered /= np.clip(np.linalg.norm(centered, axis=1, keepdims=True), 1e-9, None)
    stats["mean_vector_norm"] = float(np.linalg.norm(mean))
    stats["centered_random_all"] = summarize(random_cosines(centered, num_pairs, seed + 10))

    pca = PCA(n_components=min(20, matrix.shape[1], matrix.shape[0] - 1), svd_solver="randomized", random_state=42)
    pca.fit(matrix)
    evr = pca.explained_variance_ratio_
    stats["pca"] = {
        "top1": float(evr[0]),
        "top5_sum": float(evr[:5].sum()),
        "top20_sum": float(evr.sum()),
    }

    pos = np.flatnonzero(rows["polarity"].to_numpy() == "positive")
    neg = np.flatnonzero(rows["polarity"].to_numpy() == "negative")
    if len(pos) == len(neg):
        stats["paired_positive_negative"] = summarize(np.einsum("ij,ij->i", matrix[pos], matrix[neg]))
    return stats


def run_umap(matrix: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    import umap

    # PCA to ~30 dim — reduces UMAP input size significantly, speeding up NN search
    reduced = PCA(
        n_components=min(args.umap_components, matrix.shape[1], matrix.shape[0] - 1),
        svd_solver="randomized",
        random_state=42,
    ).fit_transform(matrix)
    reduced = np.ascontiguousarray(reduced.astype(np.float32, copy=False))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        init="random",
        n_epochs=args.umap_epochs,
        n_jobs=args.jobs,
        low_memory=False,
        random_state=42,
        verbose=True,
    )
    return reducer.fit_transform(reduced)


def save_plot(path: Path, rows: pd.DataFrame, coords: np.ndarray, title: str) -> None:
    colors = rows["lang"].map(LANG_COLORS).fillna("#7f7f7f").to_numpy()
    neg = rows["polarity"].to_numpy() == "negative"
    pos = ~neg

    plt.figure(figsize=(12, 10), dpi=180)
    plt.scatter(coords[pos, 0], coords[pos, 1], c=colors[pos], s=5, alpha=0.55, linewidths=0, marker="o")
    plt.scatter(coords[neg, 0], coords[neg, 1], c=colors[neg], s=5, alpha=0.45, linewidths=0, marker="x")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=lang, markerfacecolor=color, markersize=7)
        for lang, color in LANG_COLORS.items()
    ]
    plt.legend(handles=handles, loc="best", frameon=True, fontsize=8)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def prepare_split(args: argparse.Namespace, split: str, model, tokenizer, device: torch.device) -> tuple[pd.DataFrame, np.ndarray, dict]:
    out_dir = Path(args.out_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_split(Path(args.data_dir), split)
    rows.to_parquet(out_dir / "rows.parquet", index=False)
    embeddings = load_or_encode(
        out_dir / "embeddings.npy",
        rows,
        model,
        tokenizer,
        device,
        args.batch_size,
        args.max_len,
        split,
        args.overwrite_embeddings,
    )
    diagnostics = cosine_diagnostics(rows, embeddings, min(args.cosine_pairs, len(rows) * 100), seed=42 if split == "train" else 43)
    (out_dir / "cosine_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows, embeddings, diagnostics


def save_umap_outputs(args: argparse.Namespace, split: str, rows: pd.DataFrame, coords: np.ndarray) -> None:
    out_dir = Path(args.out_dir) / split
    projection = rows[["sample_id", "lang", "polarity", "feature", "feature_clean", "source_row"]].copy()
    projection["x"] = coords[:, 0]
    projection["y"] = coords[:, 1]
    projection.to_csv(out_dir / "umap.csv", index=False)
    save_plot(out_dir / "umap.png", rows, coords, f"mSynthSTEL {split} UMAP")


def main() -> None:
    args = parse_args()
    configure_threads(args.jobs)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model(args.model_name, Path(args.checkpoint), device)
    print(f"device={device} checkpoint={args.checkpoint}", flush=True)

    summary = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "splits": {},
        "umap": {
            "jobs": args.jobs,
            "neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "components": args.umap_components,
            "epochs": args.umap_epochs,
            "joint_train_test": bool(args.joint_umap),
        },
    }

    split_data = {}
    for split in ("train", "test"):
        rows, embeddings, diagnostics = prepare_split(args, split, model, tokenizer, device)
        split_data[split] = (rows, embeddings)
        summary["splits"][split] = diagnostics

    if args.joint_umap:
        print("running one joint UMAP for train+test", flush=True)
        train_rows, train_embeddings = split_data["train"]
        test_rows, test_embeddings = split_data["test"]
        all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
        all_coords = run_umap(all_embeddings, args)
        train_n = len(train_rows)
        save_umap_outputs(args, "train", train_rows, all_coords[:train_n])
        save_umap_outputs(args, "test", test_rows, all_coords[train_n:])
    else:
        for split, (rows, embeddings) in split_data.items():
            coords = run_umap(embeddings, args)
            save_umap_outputs(args, split, rows, coords)

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
