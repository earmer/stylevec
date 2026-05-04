"""STEL and STEL-or-Content evaluation — exact tasks from the paper (Section 5.1).

STEL: Given 2 anchors (A1, A2) of different styles and 2 test sentences (S1, S2)
      matching those styles, pair each test to the correct anchor by cosine similarity.

STEL-or-Content: Given 1 anchor (A) and 2 test sentences, pick which one shares A's
      style.  The distractor is a paraphrase of A (same content, different style) —
      requiring content-independent style representations to succeed.

Usage:
    uv run python evaluate.py --checkpoint checkpoints/best_model --use-local-data
    uv run python evaluate.py --compare-reference --use-local-data
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import Config


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Embedder:
    """Unified interface for both our model and sentence-transformers reference."""
    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


class OurEmbedder(Embedder):
    def __init__(self, checkpoint: str, model_name: str, device: torch.device):
        from peft import PeftModel
        from transformers import AutoModel
        from model import StyleDistance

        base = AutoModel.from_pretrained(model_name)
        model = StyleDistance.__new__(StyleDistance)
        torch.nn.Module.__init__(model)
        model.encoder = PeftModel.from_pretrained(base, checkpoint)
        model.encoder.enable_input_require_grads()
        model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            return self.model.encode(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            ).cpu()


class ReferenceEmbedder(Embedder):
    def __init__(self, model_id: str, device: torch.device):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(model_id, device=str(device))

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        return torch.from_numpy(self.st.encode(texts, normalize_embeddings=True))


# ── Task instance builders ────────────────────────────────────────────────────

def build_stel_instances(pairs_by_feature: dict[str, list[dict]], num_instances: int = 500) -> list[dict]:
    features = list(pairs_by_feature.keys())
    instances = []
    for _ in range(num_instances):
        fx, fy = random.sample(features, 2)
        pair_x_a, pair_x_b = random.sample(pairs_by_feature[fx], 2)
        pair_y_c, pair_y_d = random.sample(pairs_by_feature[fy], 2)
        instances.append({
            "A1": pair_x_a["positive"],
            "A2": pair_y_c["positive"],
            "S1": pair_x_b["positive"],
            "S2": pair_y_d["positive"],
            "features": (fx, fy),
        })
    return instances


def build_stel_or_content_instances(pairs_by_feature: dict[str, list[dict]], num_instances: int = 500) -> list[dict]:
    instances = []
    for feat, pairs in pairs_by_feature.items():
        if len(pairs) < 2:
            continue
        per_feat = max(1, num_instances // len(pairs_by_feature))
        for _ in range(per_feat):
            pair_a, pair_b = random.sample(pairs, 2)
            instances.append({
                "A": pair_a["positive"],
                "S_correct": pair_b["positive"],
                "S_wrong": pair_a["negative"],
                "feature": feat,
            })
    return instances


# ── Evaluation functions ──────────────────────────────────────────────────────

def eval_stel(embedder: Embedder, instances: list[dict]) -> dict:
    correct = 0
    for inst in instances:
        embs = embedder.encode_batch([inst["A1"], inst["A2"], inst["S1"], inst["S2"]])
        e_a1, e_a2, e_s1, e_s2 = embs
        correct_score = (torch.dot(e_a1, e_s1) + torch.dot(e_a2, e_s2)).item()
        wrong_score = (torch.dot(e_a1, e_s2) + torch.dot(e_a2, e_s1)).item()
        if correct_score > wrong_score:
            correct += 1
    return {"accuracy": correct / len(instances), "correct": correct, "total": len(instances)}


def eval_stel_or_content(embedder: Embedder, instances: list[dict]) -> dict:
    by_feature: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for inst in instances:
        embs = embedder.encode_batch([inst["A"], inst["S_correct"], inst["S_wrong"]])
        e_a, e_correct, e_wrong = embs
        feat = inst["feature"]
        by_feature[feat]["total"] += 1
        if torch.dot(e_a, e_correct) > torch.dot(e_a, e_wrong):
            by_feature[feat]["correct"] += 1

    total_correct = sum(v["correct"] for v in by_feature.values())
    total = sum(v["total"] for v in by_feature.values())
    per_feature = {f: v["correct"] / v["total"] for f, v in by_feature.items()}
    return {"accuracy": total_correct / total, "correct": total_correct, "total": total, "per_feature": per_feature}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STEL / STEL-or-Content evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained LoRA checkpoint")
    parser.add_argument("--compare-reference", action="store_true",
                        help="Also evaluate the official StyleDistance/styledistance model")
    parser.add_argument("--use-local-data", action="store_true")
    parser.add_argument("--num-instances", type=int, default=500)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"device={device}")

    # Load test data
    print("Loading test data...")
    if args.use_local_data:
        from config import load_multilingual_pairs
        test_pairs = load_multilingual_pairs(config, "test")
    else:
        from datasets import load_dataset
        test_pairs = [dict(row) for row in load_dataset(config.dataset_name)["test"]]

    by_feature = defaultdict(list)
    for row in test_pairs:
        by_feature[row["feature"]].append(row)
    print(f"Test pairs: {len(test_pairs)}  Features: {len(by_feature)}")

    # Build task instances
    random.seed(42)
    stel_instances = build_stel_instances(by_feature, args.num_instances)
    soc_instances = build_stel_or_content_instances(by_feature, args.num_instances)
    print(f"STEL instances: {len(stel_instances)}  S-o-C instances: {len(soc_instances)}")

    # Evaluate each checkpoint
    checkpoints = []
    if args.checkpoint:
        checkpoints.append(("ours", OurEmbedder(args.checkpoint, config.model_name, device)))
    if args.compare_reference:
        checkpoints.append(("reference (StyleDistance/styledistance)",
                           ReferenceEmbedder("StyleDistance/styledistance", device)))

    if not checkpoints:
        print("No --checkpoint provided and --compare-reference not set. Nothing to evaluate.")
        return

    for label, embedder in checkpoints:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")

        stel = eval_stel(embedder, stel_instances)
        soc = eval_stel_or_content(embedder, soc_instances)

        print(f"  STEL accuracy:             {stel['accuracy']:.4f} ({stel['correct']}/{stel['total']})")
        print(f"  STEL-or-Content accuracy:  {soc['accuracy']:.4f} ({soc['correct']}/{soc['total']})")
        print("  Per-feature S-o-C:")
        for feat, acc in sorted(soc["per_feature"].items()):
            print(f"    {feat}: {acc:.3f}")


if __name__ == "__main__":
    main()
