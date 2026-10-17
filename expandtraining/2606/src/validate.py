from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prebuilt import load_or_build
from config import configured_matryoshka_dims, load_config, workspace_dir
from dataset import TripletTextDataset, make_triplet_collate
from losses import embedding_spread_diagnostics, scalar_stats, weighted_triplet_margin_loss
from model import build_qwen_style_encoder, encode_triplet, load_trainable_state_dict
from report import per_rule_table, provenance_probe, render_report, speaker_retrieval
from probes import length_leakage_metrics, vector_probe_metrics


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("anchor", "positive", "negative"):
        moved[key] = {name: value.to(device, non_blocking=non_blocking) for name, value in batch[key].items()}
    moved["margin"] = batch["margin"].to(device, non_blocking=non_blocking)
    moved["weight"] = batch["weight"].to(device, non_blocking=non_blocking)
    return moved


@torch.no_grad()
def evaluate_triplet_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    prefix_dims: list[int] | None = None,
    combine_forward: bool = False,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.eval()
    losses: list[torch.Tensor] = []
    pos_cos: list[torch.Tensor] = []
    neg_cos: list[torch.Tensor] = []
    violations: list[torch.Tensor] = []
    pre_norms: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    lengths: list[int] = []
    length_buckets: list[int] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
        anchor, positive, negative, norms = encode_triplet(model, batch, combine_forward=combine_forward)
        loss, stats = weighted_triplet_margin_loss(
            anchor,
            positive,
            negative,
            batch["margin"],
            batch["weight"],
        )
        losses.append(loss.detach().float().cpu().reshape(1))
        pos_cos.append(stats["pos_cos"].cpu())
        neg_cos.append(stats["neg_cos"].cpu())
        violations.append(stats["violations"].cpu())
        pre_norms.append(norms.detach().float().cpu())
        embeddings.extend([anchor.cpu(), positive.cpu(), negative.cpu()])
        for role in ("anchor", "positive", "negative"):
            for text in batch[f"{role}_text"]:
                length = len(text)
                lengths.append(length)
                length_buckets.append(length // 40)
    if not losses:
        raise ValueError("validation loader produced no batches")
    loss_tensor = torch.cat(losses)
    pos_tensor = torch.cat(pos_cos)
    neg_tensor = torch.cat(neg_cos)
    violation_tensor = torch.cat(violations)
    norm_tensor = torch.cat(pre_norms)
    embedding_tensor = torch.cat(embeddings, dim=0)
    metrics = {"loss": float(loss_tensor.mean().item())}
    metrics.update(scalar_stats(pos_tensor, "pos_cos"))
    metrics.update(scalar_stats(neg_tensor, "neg_cos"))
    metrics["violation_ratio"] = float(violation_tensor.mean().item())
    metrics["pre_norm_mean"] = float(norm_tensor.mean().item())
    metrics["pre_norm_std"] = float(norm_tensor.std(unbiased=False).item())
    spread = embedding_spread_diagnostics(embedding_tensor)
    metrics["top_eigen_ratio"] = float(spread["top_eigen_ratio"].item())
    metrics["effective_rank"] = float(spread["effective_rank"].item())
    embedding_matrix = embedding_tensor.numpy()
    metrics.update(vector_probe_metrics(embedding_matrix, prefix_dims or ()))
    metrics.update(length_leakage_metrics(embedding_matrix, lengths, length_buckets))
    return metrics


@torch.no_grad()
def collect_validation_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int | None = None,
    prefix_dims: list[int] | None = None,
    combine_forward: bool = False,
    non_blocking: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    all_pos: list[torch.Tensor] = []
    all_neg: list[torch.Tensor] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
        anchor, positive, negative, _norms = encode_triplet(model, batch, combine_forward=combine_forward)
        all_pos.append(F.cosine_similarity(anchor, positive, dim=-1).cpu())
        all_neg.append(F.cosine_similarity(anchor, negative, dim=-1).cpu())
        for role, embeddings in (("anchor", anchor), ("positive", positive), ("negative", negative)):
            texts = batch[f"{role}_text"]
            for index, embedding in enumerate(embeddings.detach().cpu()):
                text = texts[index]
                rows.append(
                    {
                        "triplet_id": batch["triplet_id"][index],
                        "role": role,
                        "rule": batch["rule"][index],
                        "heldout": bool(batch["meta"][index].get("heldout", False)),
                        "speaker": speaker_of(batch["meta"][index], role),
                        "char_length": len(text),
                        "length_bucket": len(text) // 40,
                        "text_preview": text[:120],
                    }
                )
                vectors.append(embedding.numpy())
        # Stop only on a batch boundary (and only if asked): never leave a triplet half-collected.
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError("no validation embeddings collected")
    matrix = np.stack(vectors, axis=0)
    frame = pd.DataFrame(rows)
    summary = cosine_summary(torch.cat(all_pos), torch.cat(all_neg), matrix, frame, prefix_dims or ())
    return frame, matrix, summary


def speaker_of(meta: dict[str, Any], role: str) -> str:
    if role == "negative":
        return str(meta.get("negative_speaker", ""))
    return str(meta.get("speaker", ""))


def select_validation_triplets(triplets: list[Any], per_rule: int) -> list[Any]:
    """Fixed N per (rule, heldout) bucket, evenly spread; every val rule present is represented."""
    buckets: dict[tuple[str, bool], list[Any]] = {}
    for row in triplets:
        if row.split == "val":
            buckets.setdefault((row.rule, bool(row.meta.get("heldout", False))), []).append(row)
    chosen: list[Any] = []
    for key in sorted(buckets):
        rows = sorted(buckets[key], key=lambda r: r.triplet_id)
        if len(rows) > per_rule:
            picks = np.unique(np.linspace(0, len(rows) - 1, per_rule).astype(int))
            rows = [rows[i] for i in picks]
        chosen.extend(rows)
    return chosen


def cosine_summary(
    pos_cos: torch.Tensor,
    neg_cos: torch.Tensor,
    matrix: np.ndarray,
    rows: pd.DataFrame,
    prefix_dims: list[int] | tuple[int, ...] = (),
) -> dict[str, float]:
    metrics = {}
    metrics.update(scalar_stats(pos_cos, "pos_cos"))
    metrics.update(scalar_stats(neg_cos, "neg_cos"))
    metrics["violation_proxy_ratio"] = float((pos_cos <= neg_cos).float().mean().item())
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    mean_norm = float(np.linalg.norm(matrix.mean(axis=0)))
    metrics["embedding_mean_norm"] = mean_norm
    if centered.shape[0] >= 3:
        _u, singular_values, _vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular_values**2
        metrics["pca_top_component_ratio"] = float(variance[0] / max(float(variance.sum()), 1e-12))
    else:
        metrics["pca_top_component_ratio"] = 1.0
    metrics.update(vector_probe_metrics(matrix, prefix_dims))
    metrics.update(length_leakage_metrics(matrix, rows["char_length"], rows["length_bucket"]))
    return metrics


def write_umap_outputs(rows: pd.DataFrame, embeddings: np.ndarray, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if embeddings.shape[0] >= 4:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(10, embeddings.shape[0] - 1))
        coords = reducer.fit_transform(embeddings)
        method = "umap"
    else:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].T if vt.shape[0] >= 2 else np.pad(vt[:1].T, ((0, 0), (0, 1)))
        coords = centered @ basis
        method = "pca"
    plot_rows = rows.copy()
    plot_rows["x"] = coords[:, 0]
    plot_rows["y"] = coords[:, 1]
    plot_rows["method"] = method
    plot_rows.to_csv(out_dir / "umap.csv", index=False)

    fig, axis = plt.subplots(figsize=(6, 5))
    for role, role_rows in plot_rows.groupby("role"):
        axis.scatter(role_rows["x"], role_rows["y"], s=18, label=role, alpha=0.8)
    axis.set_title(f"validation embedding map ({method})")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "umap.png", dpi=160)
    plt.close(fig)


def load_checkpoint_for_validation(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint missing training_state.pt: {checkpoint_dir}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if "model_trainable" not in state:
        raise ValueError(f"checkpoint missing model_trainable: {state_path}")
    load_trainable_state_dict(model, state["model_trainable"])


def select_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["runtime"]["device"])
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA, but CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("config requested MPS, but MPS is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported runtime.device: {requested}")
    return torch.device(requested)


def run_validation(config_path: str, run_name: str, resume: str | None = None, dataset: str | None = None) -> Path:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    run_dir = workspace_dir() / "outputs" / "runs" / run_name
    checkpoint_dir = Path(resume) if resume else run_dir / "checkpoints" / "latest"
    out_dir = run_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    triplets, _manifest = load_or_build(config, dataset)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    collate = make_triplet_collate(tokenizer, int(config["model"]["max_length"]))
    selected = select_validation_triplets(triplets, int(config["validation"].get("per_rule", 64)))
    val_dataset = TripletTextDataset(selected, "val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["validation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    device = select_device(config)
    model = build_qwen_style_encoder(config).to(device)
    load_checkpoint_for_validation(model, checkpoint_dir)

    rows, embeddings, summary = collect_validation_embeddings(
        model,
        val_loader,
        device,
        max_samples=None,
        prefix_dims=configured_matryoshka_dims(config),
        combine_forward=bool(config["train"].get("combine_triplet_forward", False)),
        non_blocking=False,
    )
    rows.to_parquet(out_dir / "rows.parquet", index=False)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "cosine_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_umap_outputs(rows, embeddings, out_dir)
    table = per_rule_table(rows, embeddings)
    retrieval = {
        "heldout speakers": speaker_retrieval(rows, embeddings, heldout=True),
        "seen speakers": speaker_retrieval(rows, embeddings, heldout=False),
    }
    probe = provenance_probe(rows, embeddings)
    table.to_csv(out_dir / "per_rule.csv", index=False)
    (out_dir / "retrieval_probe.json").write_text(
        json.dumps({"retrieval": retrieval, "provenance": probe}, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = [
        "# Validation Report",
        "",
        f"- checkpoint: `{checkpoint_dir}`",
        f"- rows: `{len(rows)}`",
        f"- embedding_dim: `{embeddings.shape[1]}`",
        f"- positive cosine mean: `{summary['pos_cos_mean']:.4f}`",
        f"- negative cosine mean: `{summary['neg_cos_mean']:.4f}`",
        f"- violation proxy ratio: `{summary['violation_proxy_ratio']:.4f}`",
        f"- PCA top component ratio: `{summary['pca_top_component_ratio']:.4f}`",
        f"- length linear R2: `{summary['length_linear_r2']:.4f}`",
        f"- length/cosine delta corr: `{summary['length_cosine_delta_corr']:.4f}`",
        f"- same-bucket cosine lift: `{summary['length_same_minus_diff_cos']:.4f}`",
    ]
    report += [""] + render_report(table, retrieval, probe)
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", default=None, help="prebuilt dataset name or path; default: build from raw")
    args = parser.parse_args()
    out_dir = run_validation(args.config, args.run_name, args.resume, args.dataset)
    print(f"validation outputs: {out_dir}")


if __name__ == "__main__":
    main()
