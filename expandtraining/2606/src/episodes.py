"""Episode InfoNCE: contrast speakers on pooled lines, not single lines.

Line-level InfoNCE at a low temperature pushes each line away from the other speakers' lines; when two
speakers' voices overlap, utterance content is the cheapest tie-breaker, and the line leak grows with it.
Here each step draws P training speakers and 2k of each one's real lines from different scenes (db ids at
least `min_gap` apart), embeds them, averages each speaker's two halves into a query and a key episode, and
runs InfoNCE between speakers on the episodes: line-specific content averages out before the loss sees it.
"""
from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F

EPISODE_RULES = ("same_speaker_real", "same_speaker_topic_matched", "origin_vs_para")


def episode_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """train.episode_weight > 0 turns it on. episode_speakers (P) x 2 x episode_k lines per step."""
    train = config["train"]
    return {
        "weight": float(train.get("episode_weight", 0.0)),
        "speakers": int(train.get("episode_speakers", 12)),
        "k": int(train.get("episode_k", 4)),
        "temperature": float(train.get("episode_temperature", 0.05)),
        "min_gap": int(train.get("episode_min_gap", 150)),
    }


def episode_pool(triplets: list[Any]) -> dict[str, dict[str, int]]:
    """Each training speaker's distinct real lines with their db ids (anchors and real positives of the
    Genshin speaker rules; held-out speakers and the paraphrase negatives never enter)."""
    pool: dict[str, dict[str, int]] = {}
    for row in triplets:
        if row.split != "train" or row.rule not in EPISODE_RULES or row.meta.get("heldout"):
            continue
        speaker = row.meta.get("speaker")
        if not speaker:
            continue
        lines = pool.setdefault(str(speaker), {})
        lines.setdefault(row.anchor_text, int(row.meta.get("anchor_db_id", -1)))
        if "positive_db_id" in row.meta:
            lines.setdefault(row.positive_text, int(row.meta["positive_db_id"]))
    return pool


def spread_lines(lines: dict[str, int], count: int, min_gap: int, rng: random.Random) -> list[str] | None:
    """`count` lines whose db ids are pairwise at least min_gap apart (different scenes), or None."""
    order = list(lines.items())
    rng.shuffle(order)
    picked: list[tuple[str, int]] = []
    for text, db_id in order:
        if all(abs(db_id - other) >= min_gap for _text, other in picked):
            picked.append((text, db_id))
            if len(picked) == count:
                return [text for text, _db_id in picked]
    return None


class EpisodeSampler:
    def __init__(self, pool: dict[str, dict[str, int]], settings: dict[str, Any], seed: int = 0) -> None:
        self.pool, self.settings = pool, settings
        self.rng = random.Random(seed)
        need = 2 * settings["k"]
        self.speakers = sorted(name for name, lines in pool.items() if len(lines) >= need)
        if len(self.speakers) < settings["speakers"]:
            raise ValueError(f"episode InfoNCE needs {settings['speakers']} speakers with {need}+ lines, "
                             f"the training data has {len(self.speakers)}")

    def sample(self) -> list[str]:
        """P * 2k texts, speaker by speaker: [s0 query lines, s0 key lines, s1 query lines, ...]."""
        settings, texts = self.settings, []
        candidates = self.speakers[:]
        self.rng.shuffle(candidates)
        for speaker in candidates:
            lines = spread_lines(self.pool[speaker], 2 * settings["k"], settings["min_gap"], self.rng)
            if lines is not None:
                texts.extend(lines)
                if len(texts) == settings["speakers"] * 2 * settings["k"]:
                    return texts
        raise ValueError("too few speakers with lines from enough different scenes for one episode batch")


def episode_infonce(embeddings: torch.Tensor, speakers: int, k: int,
                    temperature: float) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Symmetric InfoNCE between each speaker's query episode and every speaker's key episode (the mean of k
    unit line embeddings, renormalised)."""
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        lines = F.normalize(embeddings.float(), dim=-1).reshape(speakers, 2, k, -1)
        episodes = F.normalize(lines.mean(dim=2), dim=-1)
        logits = episodes[:, 0] @ episodes[:, 1].T / float(temperature)
    targets = torch.arange(speakers, device=embeddings.device)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
    return loss, {"episode_loss": loss.detach(),
                  "episode_accuracy": (logits.argmax(dim=1) == targets).float().mean().detach(),
                  "episode_pos_cos": torch.diagonal(logits).mean().detach() * float(temperature)}


def episode_step_loss(model: torch.nn.Module, tokenizer: Any, sampler: EpisodeSampler | None, max_length: int,
                      device: torch.device) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """The weighted episode loss for one micro-batch, or (None, {}) when episodes are off."""
    if sampler is None:
        return None, {}
    settings = sampler.settings
    tokens = tokenizer(sampler.sample(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    output = model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))
    loss, stats = episode_infonce(output["embeddings"], settings["speakers"], settings["k"], settings["temperature"])
    return loss * settings["weight"], stats
