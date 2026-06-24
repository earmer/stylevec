from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import load_config, workspace_dir
from import_bookraw import load_bookraw_samples
from import_genshin import load_genshin_samples, load_top_speakers
from import_msynthstel import load_msynthstel_samples
from import_paratarget import load_paratarget_samples
from model import build_qwen_style_encoder, load_trainable_state_dict
from records import SampleRecord, normalize_text
from validate import select_device


def configure_plot_fonts() -> None:
    import matplotlib
    from matplotlib import font_manager

    candidates = [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams["font.family"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return


class SampleTextDataset(Dataset[SampleRecord]):
    def __init__(self, samples: list[SampleRecord]) -> None:
        if not samples:
            raise ValueError("empty sample dataset")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleRecord:
        return self.samples[index]


class SampleCollate:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows: list[SampleRecord]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [normalize_text(row.text) for row in rows],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "encoded": encoded,
            "rows": rows,
        }


def load_all_samples(config: dict[str, Any]) -> list[SampleRecord]:
    speakers = load_top_speakers(config)
    genshin_samples, _genshin_filters = load_genshin_samples(config, speakers)
    paratarget_samples, _paratarget_filters = load_paratarget_samples(config, speakers)
    msynthstel_samples, _msynthstel_filters = load_msynthstel_samples(config)
    bookraw_samples, _bookraw_filters = load_bookraw_samples(config)
    return genshin_samples + paratarget_samples + msynthstel_samples + bookraw_samples


def proportional_domain_targets(samples: list[SampleRecord], max_samples: int) -> dict[str, int]:
    by_domain = Counter(row.domain for row in samples)
    total = sum(by_domain.values())
    if max_samples >= total:
        return dict(by_domain)
    raw = {domain: count * max_samples / total for domain, count in by_domain.items()}
    targets = {domain: max(1, int(math.floor(value))) for domain, value in raw.items()}
    while sum(targets.values()) < max_samples:
        domain = max(raw, key=lambda item: raw[item] - targets[item])
        if targets[domain] < by_domain[domain]:
            targets[domain] += 1
        else:
            raw[domain] = -1.0
    while sum(targets.values()) > max_samples:
        domain = max(targets, key=targets.get)
        targets[domain] -= 1
    return targets


def round_robin_by_style(rows: list[SampleRecord], target: int, rng: np.random.Generator) -> list[SampleRecord]:
    grouped: defaultdict[str, list[SampleRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.style_id].append(row)
    buckets = list(grouped.values())
    for bucket in buckets:
        rng.shuffle(bucket)
    rng.shuffle(buckets)
    selected: list[SampleRecord] = []
    cursor = 0
    while len(selected) < target and buckets:
        bucket = buckets[cursor % len(buckets)]
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            buckets.remove(bucket)
            cursor = 0
        else:
            cursor += 1
    return selected


def sample_records(samples: list[SampleRecord], max_samples: int, seed: int) -> list[SampleRecord]:
    rng = np.random.default_rng(seed)
    targets = proportional_domain_targets(samples, max_samples)
    selected: list[SampleRecord] = []
    for domain, target in sorted(targets.items()):
        domain_rows = [row for row in samples if row.domain == domain]
        selected.extend(round_robin_by_style(domain_rows, target, rng))
    rng.shuffle(selected)
    return selected


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> None:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"checkpoint missing training_state.pt: {checkpoint_dir}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, state["model_trainable"])


@torch.no_grad()
def embed_samples(
    config: dict[str, Any],
    checkpoint_dir: Path,
    samples: list[SampleRecord],
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model_path"],
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    device = select_device(config)
    model = build_qwen_style_encoder(config).to(device)
    load_checkpoint(model, checkpoint_dir)
    model.eval()

    loader = DataLoader(
        SampleTextDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=SampleCollate(tokenizer, int(config["model"]["max_length"])),
    )
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for batch in loader:
        encoded = {name: value.to(device) for name, value in batch["encoded"].items()}
        output = model(**encoded)
        embeddings = output["embeddings"].detach().float().cpu().numpy()
        vectors.append(embeddings)
        for row in batch["rows"]:
            rows.append(
                {
                    "sample_id": row.sample_id,
                    "domain": row.domain,
                    "style_id": row.style_id,
                    "style_type": row.style_type,
                    "split": row.split,
                    "language": row.language,
                    "quality": row.quality,
                    "text_preview": normalize_text(row.text)[:80],
                    "polarity": row.meta.get("polarity", ""),
                    "source_speaker": row.meta.get("source_speaker", ""),
                    "target_speaker": row.meta.get("target_speaker", ""),
                }
            )
    return pd.DataFrame(rows), np.concatenate(vectors, axis=0)


def write_umap(rows: pd.DataFrame, embeddings: np.ndarray, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import umap

    configure_plot_fonts()
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=min(30, max(2, embeddings.shape[0] - 1)),
        min_dist=0.05,
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    plot_rows = rows.copy()
    plot_rows["x"] = coords[:, 0]
    plot_rows["y"] = coords[:, 1]
    plot_rows.to_csv(out_dir / "sample_umap.csv", index=False)

    fig, axis = plt.subplots(figsize=(12, 9))
    for domain, domain_rows in plot_rows.groupby("domain"):
        axis.scatter(domain_rows["x"], domain_rows["y"], s=8, alpha=0.65, label=f"{domain} ({len(domain_rows)})")
    axis.set_title("raw sample embedding UMAP by domain")
    axis.set_xlabel("umap_x")
    axis.set_ylabel("umap_y")
    axis.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_umap_by_domain.png", dpi=180)
    plt.close(fig)

    top_styles = set(plot_rows["style_id"].value_counts().head(24).index)
    style_plot = plot_rows.copy()
    style_plot["style_group"] = style_plot["style_id"].where(style_plot["style_id"].isin(top_styles), "other")
    fig, axis = plt.subplots(figsize=(12, 9))
    for style_id, style_rows in style_plot.groupby("style_group"):
        alpha = 0.18 if style_id == "other" else 0.7
        size = 5 if style_id == "other" else 9
        axis.scatter(style_rows["x"], style_rows["y"], s=size, alpha=alpha, label=f"{style_id} ({len(style_rows)})")
    axis.set_title("raw sample embedding UMAP by frequent style_id")
    axis.set_xlabel("umap_x")
    axis.set_ylabel("umap_y")
    axis.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_umap_by_style.png", dpi=180)
    plt.close(fig)


def redraw_from_csv(csv_path: Path) -> Path:
    import matplotlib.pyplot as plt

    configure_plot_fonts()
    plot_rows = pd.read_csv(csv_path)
    out_dir = csv_path.parent

    fig, axis = plt.subplots(figsize=(12, 9))
    for domain, domain_rows in plot_rows.groupby("domain"):
        axis.scatter(domain_rows["x"], domain_rows["y"], s=8, alpha=0.65, label=f"{domain} ({len(domain_rows)})")
    axis.set_title("raw sample embedding UMAP by domain")
    axis.set_xlabel("umap_x")
    axis.set_ylabel("umap_y")
    axis.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_umap_by_domain.png", dpi=180)
    plt.close(fig)

    top_styles = set(plot_rows["style_id"].value_counts().head(24).index)
    style_plot = plot_rows.copy()
    style_plot["style_group"] = style_plot["style_id"].where(style_plot["style_id"].isin(top_styles), "other")
    fig, axis = plt.subplots(figsize=(12, 9))
    for style_id, style_rows in style_plot.groupby("style_group"):
        alpha = 0.18 if style_id == "other" else 0.7
        size = 5 if style_id == "other" else 9
        axis.scatter(style_rows["x"], style_rows["y"], s=size, alpha=alpha, label=f"{style_id} ({len(style_rows)})")
    axis.set_title("raw sample embedding UMAP by frequent style_id")
    axis.set_xlabel("umap_x")
    axis.set_ylabel("umap_y")
    axis.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_umap_by_style.png", dpi=180)
    plt.close(fig)
    return out_dir


def run(config_path: str, run_name: str, checkpoint: str, max_samples: int, batch_size: int) -> Path:
    config = load_config(config_path)
    samples = load_all_samples(config)
    selected = sample_records(samples, max_samples=max_samples, seed=int(config["runtime"]["seed"]))
    out_dir = workspace_dir() / "outputs" / "runs" / run_name / "sample_umap"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, embeddings = embed_samples(config, Path(checkpoint), selected, batch_size)
    np.save(out_dir / "sample_embeddings.npy", embeddings)
    rows.to_parquet(out_dir / "sample_rows.parquet", index=False)
    counts = {
        "all_samples": len(samples),
        "selected_samples": len(selected),
        "all_by_domain": dict(Counter(row.domain for row in samples)),
        "selected_by_domain": dict(Counter(row.domain for row in selected)),
    }
    (out_dir / "sample_counts.json").write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    write_umap(rows, embeddings, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--redraw-csv", default=None)
    args = parser.parse_args()
    if args.redraw_csv:
        out_dir = redraw_from_csv(Path(args.redraw_csv))
        print(f"sample UMAP outputs: {out_dir}")
        return
    out_dir = run(args.config, args.run_name, args.checkpoint, args.max_samples, args.batch_size)
    print(f"sample UMAP outputs: {out_dir}")
