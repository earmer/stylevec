"""Named (explainable) embedding dimensions: dimension t of the embedding is trained to track Jev trait t.

Each step draws `named_trait_batch` texts that Jev scored (scripts/jev_label.py: training lines and their own
neutral paraphrases, so a trait that a paraphrase removes is taught as a contrast), embeds them, and adds
`named_trait_weight` x mean over traits of (1 - Pearson correlation between embedding dimension t and the
trait's score) across the batch. Correlation is scale-free, so the dimensions keep whatever norm the contrastive
loss gives them; the remaining dimensions are free for the speaker loss. Off when train.named_trait_weight is 0.

With train.named_trait_pair_files the batch is named_trait_batch / 2 same-content pairs instead (a line and its
own paraphrase, a line and its Jev-verified tone twin, a book passage and its verified style rewrite: rows
{"a", "b", "scores_a", "scores_b"} from scripts/trait_pairs.py), and a second term asks each dimension's
within-pair difference to correlate with the trait's difference across the pairs: the shared content cancels in
the difference, so only the style change can explain it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from traits import TRAITS


def named_trait_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    train = config["train"]
    return {
        "weight": float(train.get("named_trait_weight", 0.0)),
        "files": list(train.get("named_trait_files", [])),
        "names": list(train.get("named_trait_names", list(TRAITS))),
        "batch": int(train.get("named_trait_batch", 64)),
        "pair_files": list(train.get("named_trait_pair_files", [])),
    }


class NamedTraitSampler:
    def __init__(self, settings: dict[str, Any], seed: int = 0) -> None:
        self.settings, self.rng = settings, random.Random(seed)
        unknown = set(settings["names"]) - set(TRAITS)
        if unknown:
            raise ValueError(f"unknown named traits: {sorted(unknown)}")
        vector = lambda scores: [float(scores[name][0]) for name in settings["names"]]  # noqa: E731
        self.pairs: list[tuple[str, str]] = []
        if settings.get("pair_files"):
            pair_scores = []
            for path in settings["pair_files"]:
                for line in open(Path(path), encoding="utf-8"):
                    row = json.loads(line)
                    self.pairs.append((row["a"], row["b"]))
                    pair_scores.append([vector(row["scores_a"]), vector(row["scores_b"])])
            if len(self.pairs) < settings["batch"] // 2:
                raise ValueError(f"named trait pairs need >= {settings['batch'] // 2} pairs, got {len(self.pairs)}")
            self.pair_scores = torch.tensor(pair_scores)  # (pairs, 2, traits)
            return
        rows: dict[str, list[float]] = {}
        for path in settings["files"]:
            for line in open(Path(path), encoding="utf-8"):
                row = json.loads(line)
                rows[row["text"]] = vector(row["scores"])
        if len(rows) < settings["batch"]:
            raise ValueError(f"named traits need >= {settings['batch']} labelled texts, got {len(rows)}")
        self.texts = sorted(rows)
        self.scores = torch.tensor([rows[text] for text in self.texts])

    def sample(self) -> tuple[list[str], torch.Tensor]:
        """batch texts and their scores; in pair mode [a0, b0, a1, b1, ...]."""
        if self.pairs:
            index = self.rng.sample(range(len(self.pairs)), self.settings["batch"] // 2)
            return [text for i in index for text in self.pairs[i]], self.pair_scores[index].flatten(0, 1)
        index = self.rng.sample(range(len(self.texts)), self.settings["batch"])
        return [self.texts[i] for i in index], self.scores[index]


def named_trait_loss(embeddings: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Mean over traits of 1 - corr(embedding dim t, score t) across the batch. A trait with no spread in the
    batch (all neutral) contributes nothing."""
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        z = embeddings.float()[:, : scores.shape[1]]
        y = scores.to(z.device, torch.float32)
        z = z - z.mean(dim=0)
        y = y - y.mean(dim=0)
        spread = y.norm(dim=0) > 1e-6
        corr = (z * y).sum(dim=0) / (z.norm(dim=0) * y.norm(dim=0)).clamp_min(1e-6)
        loss = (1.0 - corr[spread]).mean() if spread.any() else z.sum() * 0.0
    return loss, {"named_trait_loss": loss.detach(), "named_trait_corr": corr[spread].mean().detach()}


def named_trait_pair_loss(embeddings: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """For rows [a0, b0, a1, b1, ...]: the per-text loss plus the same correlation loss on within-pair
    differences (a - b), where the pair's shared content cancels."""
    text_loss, stats = named_trait_loss(embeddings, scores)
    delta_loss, delta_stats = named_trait_loss(embeddings[0::2] - embeddings[1::2], scores[0::2] - scores[1::2])
    stats.update({"named_trait_delta_loss": delta_stats["named_trait_loss"],
                  "named_trait_delta_corr": delta_stats["named_trait_corr"]})
    return text_loss + delta_loss, stats


def named_trait_step_loss(model: torch.nn.Module, tokenizer: Any, sampler: NamedTraitSampler | None, max_length: int,
                          device: torch.device) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """The weighted named-trait loss for one micro-batch, or (None, {}) when it is off."""
    if sampler is None:
        return None, {}
    texts, scores = sampler.sample()
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    output = model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))
    loss, stats = (named_trait_pair_loss if sampler.pairs else named_trait_loss)(output["embeddings"], scores)
    return loss * sampler.settings["weight"], stats
