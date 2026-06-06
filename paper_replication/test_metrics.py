"""Comprehensive StyleDistance evaluation — all four metrics in one run.

Metrics:
  1. Cosine Silhouette Score — style features as class labels
  2. Intra-class Cosine Consistency — mean pairwise cosine within each feature
  3. STEL Accuracy (Section 5.1) — pair test sentences to correct anchors
  4. STEL-or-Content Accuracy (Section 5.1) — same-style vs same-content

Usage:
  uv run python test_metrics.py                                   # interactive ckpt picker, local data
  uv run python test_metrics.py --checkpoint ../artifacts/paper_replication/checkpoints/latest
  uv run python test_metrics.py --checkpoint ckpt --reference
  uv run python test_metrics.py --reference both              # both reference models
  uv run python test_metrics.py --use-remote-data                 # download from HuggingFace
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── sys.path: enable imports from parent (shared/) ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from evaluate import (  # same-directory imports
    Embedder,
    OurEmbedder,
    ReferenceEmbedder,
    build_stel_instances,
    build_stel_or_content_instances,
    eval_stel,
    eval_stel_or_content,
    get_device,
)
from shared.evaluate import consistency, silhouette


# ── Baseline embedder (raw roberta-base, no LoRA) ──────────────────────────

class BaselineEmbedder(Embedder):
    """Frozen roberta-base with mean pooling + L2 norm, no LoRA adapter."""

    def __init__(self, model_name: str, device: torch.device):
        self.model = AutoModel.from_pretrained(model_name).to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            out = self.model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
        h = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(self.device)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1).cpu()


# ── Data loading ───────────────────────────────────────────────────────────

def load_test_data(use_local: bool, config: Config) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load multilingual SynthSTEL test pairs and group by feature."""
    if use_local:
        from config import load_multilingual_pairs
        test_pairs = load_multilingual_pairs(config, "test")
    else:
        from datasets import load_dataset
        test_pairs = [dict(row) for row in load_dataset(config.dataset_name)["test"]]

    by_feature = defaultdict(list)
    for row in test_pairs:
        by_feature[row["feature"]].append(row)

    print(f"Test pairs: {len(test_pairs)}  Features: {len(by_feature)}")
    return test_pairs, by_feature


# ── Embedding extraction for silhouette / consistency ──────────────────────

def extract_feature_embeddings(
    embedder: Embedder, pairs_by_feature: dict[str, list[dict]]
):
    """Encode all positive test sentences, return (vecs, labels, feature_names)."""
    feature_names = sorted(pairs_by_feature.keys())
    all_texts = []
    all_labels = []

    for feat_idx, feat in enumerate(feature_names):
        positives = [p["positive"] for p in pairs_by_feature[feat]]
        all_texts.extend(positives)
        all_labels.extend([feat_idx] * len(positives))

    # Batch encode (400 sentences fits in one go, but use batches of 64 for safety)
    embs = []
    bs = 64
    for i in range(0, len(all_texts), bs):
        embs.append(embedder.encode_batch(all_texts[i : i + bs]))
    vecs = torch.cat(embs).numpy().astype(np.float64)
    labels = np.array(all_labels, dtype=np.int64)
    return vecs, labels, feature_names


# ── Metrics wrappers ───────────────────────────────────────────────────────

def compute_style_metrics(vecs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute cosine silhouette and intra-class consistency."""
    sil = silhouette(vecs, labels)
    n_classes = len(set(labels.tolist()))
    con = consistency(vecs, labels, n_classes)
    return {"silhouette": sil, "consistency": con}


# ── Checkpoint selection menu ──────────────────────────────────────────────

def select_checkpoint(config: Config) -> str | None:
    """Interactive checkpoint picker. Returns path or None for baseline."""
    ckpt_dir = config.output_dir
    if not ckpt_dir.exists():
        print(f"No checkpoints directory ({ckpt_dir}). Using baseline roberta-base.")
        return None

    # Find all checkpoint directories (exclude latest/, tf-logs/, .lock, etc.)
    step_dirs = sorted(
        [d for d in ckpt_dir.iterdir()
         if d.is_dir() and d.name not in ("latest", "tf-logs") and not d.name.startswith(".")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not step_dirs:
        print(f"No checkpoints found in {ckpt_dir}. Using baseline roberta-base.")
        return None

    # Load history as a list (chronological order, matches mtime-ascending dirs)
    history_list: list[dict] = []
    history_path = ckpt_dir / "history.jsonl"
    if history_path.exists():
        for line in history_path.read_text().strip().splitlines():
            if line.strip():
                history_list.append(json.loads(line))

    # Sort dirs by mtime ascending (oldest first) to match history.jsonl append order
    step_dirs_chrono = sorted(
        step_dirs,
        key=lambda d: d.stat().st_mtime,
        reverse=False,
    )

    # Match each checkpoint dir to its history entry (chronological order)
    # step_dirs_chrono[i] ↔ history_list[i]
    dir_meta: dict[str, dict] = {}  # dirname → {val_loss, epoch, ...}
    for idx, d in enumerate(step_dirs_chrono):
        hist = history_list[idx] if idx < len(history_list) else {}
        dir_meta[d.name] = {"val_loss": hist.get("val_loss"), "epoch": hist.get("epoch")}

    # Resolve latest
    latest_dir = ckpt_dir / "latest"
    latest_target = None
    if latest_dir.exists():
        latest_safe = latest_dir / "adapter_model.safetensors"
        latest_size = latest_safe.stat().st_size if latest_safe.exists() else 0
        for d in step_dirs:
            safe = d / "adapter_model.safetensors"
            if safe.exists() and safe.stat().st_size == latest_size:
                latest_target = d.name
                break

    # Sort by val_loss ascending (best first), None values last
    def _sort_key(d):
        vl = dir_meta.get(d.name, {}).get("val_loss")
        return (0, vl) if vl is not None else (1, 0)

    step_dirs_sorted = sorted(
        [d for d in step_dirs if (d / "adapter_model.safetensors").exists()],
        key=_sort_key,
    )

    print("\nAvailable checkpoints (sorted by val_loss, best first):")
    choices = []
    for i, d in enumerate(step_dirs_sorted, 1):
        meta = dir_meta.get(d.name, {})
        val_loss = meta.get("val_loss")
        epoch = meta.get("epoch")
        tag = "  (latest)" if d.name == latest_target else ""
        loss_str = f"val_loss={val_loss:.6f}" if val_loss is not None else "val_loss=?"
        epoch_str = f"epoch={epoch}" if epoch is not None else ""
        print(f"  [{i}] {d.name}  {loss_str}  {epoch_str}{tag}")
        choices.append((i, str(d)))

    print("  [0] Skip — use baseline roberta-base (no LoRA)")

    while True:
        try:
            sel = input("\nSelect checkpoint [0]: ").strip()
            if sel == "" or sel == "0":
                return None
            sel_idx = int(sel)
            for idx, path in choices:
                if idx == sel_idx:
                    return path
            print(f"Invalid selection: {sel}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print()
            return None


# ── Output formatting ──────────────────────────────────────────────────────

def print_results(label: str, style_metrics: dict, stel: dict, soc: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    print(f"  Cosine Silhouette Score:         {style_metrics['silhouette']:.4f}")
    print(f"  Intra-class Cosine Consistency:  {style_metrics['consistency']:.4f}")

    print(f"\n  STEL Accuracy (Section 5.1):")
    print(f"    accuracy: {stel['accuracy']:.4f} ({stel['correct']}/{stel['total']})")

    print(f"\n  STEL-or-Content Accuracy (Section 5.1):")
    print(f"    accuracy: {soc['accuracy']:.4f} ({soc['correct']}/{soc['total']})")
    print(f"    per-feature:")
    for feat, acc in sorted(soc["per_feature"].items()):
        print(f"      {feat + ':':50s} {acc:.3f}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Comprehensive StyleDistance evaluation — silhouette, consistency, STEL, S-o-C"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained LoRA checkpoint dir (interactive menu if omitted)")
    parser.add_argument("--use-remote-data", action="store_true",
                        help="Download test data from HuggingFace instead of local parquet")
    parser.add_argument("--reference", type=str, default=None, nargs="?",
                        const="styledistance",
                        choices=["styledistance", "synthetic", "both"],
                        help="Compare against reference model(s): styledistance, synthetic, or both")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if GPU available")
    parser.add_argument("--num-instances", type=int, default=500,
                        help="Number of STEL/S-o-C task instances (default: 500)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"device={device}  num_instances={args.num_instances}")

    # ── Load test data ──
    print("Loading test data...")
    test_pairs, pairs_by_feature = load_test_data(not args.use_remote_data, config)

    # ── Seed for reproducible STEL/S-o-C instances ──
    random.seed(42)

    # ── Resolve checkpoint ──
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = select_checkpoint(config)

    # ── Build embedders ──
    embedders: list[tuple[str, Embedder]] = []

    if ckpt_path:
        embedders.append(("ours", OurEmbedder(ckpt_path, config.model_name, device)))
    else:
        print("No checkpoint selected. Using baseline roberta-base (frozen, no LoRA).")
        embedders.append(("baseline (roberta-base, no LoRA)",
                         BaselineEmbedder(config.model_name, device)))

    if args.reference:
        base_models = Path(__file__).resolve().parent.parent / "artifacts" / "base-models"
        ref_models = {
            "styledistance": ("StyleDistance/styledistance (reference)", "styledistance"),
            "synthetic": ("StyleDistance/synthetic_only (reference)", "styledistance_synthetic_only"),
        }
        keys = ["styledistance", "synthetic"] if args.reference == "both" else [args.reference]
        for key in keys:
            label, local_dir = ref_models[key]
            path = base_models / local_dir
            if path.exists():
                embedders.append((label, ReferenceEmbedder(str(path), device)))
            else:
                print(f"Local model not found at {path}, trying HuggingFace...")
                embedders.append((label, ReferenceEmbedder(f"StyleDistance/{local_dir}", device)))

    if not embedders:
        print("No embedder configured. Provide --checkpoint or --compare-reference.")
        return

    # ── Build STEL/S-o-C instances once (same for all embedders) ──
    stel_instances = build_stel_instances(pairs_by_feature, args.num_instances)
    soc_instances = build_stel_or_content_instances(pairs_by_feature, args.num_instances)
    print(f"STEL instances: {len(stel_instances)}  S-o-C instances: {len(soc_instances)}")

    # ── Evaluate each embedder ──
    for label, embedder in embedders:
        # Style-space metrics
        vecs, lbls, feat_names = extract_feature_embeddings(embedder, pairs_by_feature)
        style_metrics = compute_style_metrics(vecs, lbls)

        # Paper metrics
        stel = eval_stel(embedder, stel_instances)
        soc = eval_stel_or_content(embedder, soc_instances)

        print_results(label, style_metrics, stel, soc)

    print()


if __name__ == "__main__":
    main()
