"""Episode InfoNCE: contrast speakers on pooled lines, not single lines.

Line-level InfoNCE at a low temperature pushes each line away from the other speakers' lines; when two
speakers' voices overlap, utterance content is the cheapest tie-breaker, and the line leak grows with it.
Here each step draws P training speakers and 2k of each one's real lines from different scenes (db ids at
least `min_gap` apart), embeds them, averages each speaker's two halves into two episodes, and runs InfoNCE over
all 2P episodes (the other half is the positive): line-specific content averages out before the loss sees it.

train.episode_file swaps the Genshin speakers for authors from a jsonl of {"author", "topic", "text"} rows
(scripts/build_episode_file.py: Reddit training authors, topic = subreddit): every line of an author's episode
pair then comes from a different topic, so the two halves share voice and nothing else.
"""
from __future__ import annotations

import json
import random
from typing import Any

import torch
import torch.nn.functional as F

from model import RunningCenter

EPISODE_RULES = ("same_speaker_real", "same_speaker_topic_matched", "origin_vs_para")


def episode_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """train.episode_weight > 0 turns it on. episode_speakers (P) x 2 x episode_k lines per step."""
    train = config["train"]
    return {
        "weight": float(train.get("episode_weight", 0.0)),
        "speakers": int(train.get("episode_speakers", 16)),
        "k": int(train.get("episode_k", 3)),
        "temperature": float(train.get("episode_temperature", 0.07)),
        "min_gap": int(train.get("episode_min_gap", 150)),
        "file": str(train.get("episode_file", "") or ""),
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


def episode_pool_from_file(path: str) -> dict[str, dict[str, str]]:
    """Each author's distinct texts with their topics, from train.episode_file."""
    from config import resolve_repo_path

    pool: dict[str, dict[str, str]] = {}
    for line in resolve_repo_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pool.setdefault(str(row["author"]), {}).setdefault(str(row["text"]), str(row["topic"]))
    return pool


def apart(a: int | str, b: int | str, min_gap: int) -> bool:
    """Db ids at least min_gap apart (different scenes); topics simply different."""
    return a != b if isinstance(a, str) else abs(a - b) >= min_gap


def spread_lines(lines: dict[str, int | str], count: int, min_gap: int, rng: random.Random) -> list[str] | None:
    """`count` lines pairwise apart (different scenes or topics), or None."""
    order = list(lines.items())
    rng.shuffle(order)
    picked: list[tuple[str, int | str]] = []
    for text, key in order:
        if all(apart(key, other, min_gap) for _text, other in picked):
            picked.append((text, key))
            if len(picked) == count:
                return [text for text, _db_id in picked]
    return None


class EpisodeSampler:
    def __init__(self, pool: dict[str, dict[str, int | str]], settings: dict[str, Any], seed: int = 0) -> None:
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


def episode_infonce(embeddings: torch.Tensor, speakers: int, k: int, temperature: float,
                    center: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """InfoNCE over all 2P episodes (each the mean of k unit line embeddings, renormalised): an episode's
    positive is the other half of its speaker, its negatives the 2P - 2 episodes of the other speakers.
    center subtracts the batch mean first: every line of a file episode shares one domain (Reddit, English),
    whose common direction would otherwise dominate the pooled means."""
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        raw = embeddings.float()
        if center:
            raw = raw - raw.mean(dim=0, keepdim=True)
        lines = F.normalize(raw, dim=-1).reshape(speakers * 2, k, -1)
        episodes = F.normalize(lines.mean(dim=1), dim=-1)
        cos = episodes @ episodes.T
        logits = (cos / float(temperature)).masked_fill(
            torch.eye(2 * speakers, dtype=torch.bool, device=cos.device), float("-inf"))
    targets = torch.arange(2 * speakers, device=embeddings.device) ^ 1
    loss = F.cross_entropy(logits, targets)
    return loss, {"episode_loss": loss.detach(),
                  "episode_accuracy": (logits.argmax(dim=1) == targets).float().mean().detach(),
                  "episode_pos_cos": cos[torch.arange(2 * speakers), targets].mean().detach()}


def episode_step_loss(model: torch.nn.Module, tokenizer: Any, sampler: EpisodeSampler | None, max_length: int,
                      device: torch.device) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """The weighted episode loss for one micro-batch, or (None, {}) when episodes are off."""
    if sampler is None:
        return None, {}
    settings = sampler.settings
    tokens = tokenizer(sampler.sample(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # Episode lines must not move the running center (model.center): a file's lines (Reddit) would pull it
    # off the training data, and validation uses the frozen center.
    centers = [module for module in model.modules() if isinstance(module, RunningCenter) and module.training]
    for module in centers:
        module.train(False)
    try:
        output = model(input_ids=tokens["input_ids"].to(device), attention_mask=tokens["attention_mask"].to(device))
    finally:
        for module in centers:
            module.train(True)
    loss, stats = episode_infonce(output["embeddings"], settings["speakers"], settings["k"], settings["temperature"],
                                  center=bool(settings["file"]))
    return loss * settings["weight"], stats
