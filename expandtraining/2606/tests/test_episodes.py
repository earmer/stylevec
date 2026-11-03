from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from episodes import EpisodeSampler, episode_infonce, episode_pool, spread_lines
from records import TripletRecord


def record(speaker: str, anchor: str, a_id: int, positive: str, p_id: int, rule: str = "same_speaker_real",
           split: str = "train", heldout: bool = False) -> TripletRecord:
    return TripletRecord(anchor, positive, "neg", rule, 1.0, 0.1, split,
                         {"speaker": speaker, "heldout": heldout, "anchor_db_id": a_id, "positive_db_id": p_id})


def test_pool_keeps_training_speakers_real_lines_only():
    rows = [record("A", "a1", 0, "a2", 500), record("A", "a1", 0, "a3", 900, rule="same_speaker_topic_matched"),
            record("B", "b1", 0, "b2", 400, split="val"), record("C", "c1", 0, "c2", 400, heldout=True),
            record("D", "d1", 0, "d2", 400, rule="trait_positive_vs_negative")]
    assert episode_pool(rows) == {"A": {"a1": 0, "a2": 500, "a3": 900}}


def test_spread_lines_come_from_different_scenes():
    lines = {f"l{i}": i * 50 for i in range(40)}  # every third line is 150 ids on
    picked = spread_lines(lines, 6, 150, random.Random(0))
    ids = sorted(lines[t] for t in picked)
    assert len(picked) == 6 and all(b - a >= 150 for a, b in zip(ids, ids[1:]))
    assert spread_lines(lines, 20, 150, random.Random(0)) is None


def test_sampler_returns_whole_episodes_and_skips_speakers_without_enough_scenes():
    pool = {f"s{i}": {f"s{i}_{j}": j * 200 for j in range(8)} for i in range(5)}
    pool["narrow"] = {f"n{j}": j for j in range(8)}  # one scene: never usable
    sampler = EpisodeSampler(pool, {"speakers": 4, "k": 2, "min_gap": 150, "temperature": 0.05, "weight": 1.0})
    texts = sampler.sample()
    assert len(texts) == 16 and not any(t.startswith("n") for t in texts)
    for block in range(4):
        assert len({t.split("_")[0] for t in texts[block * 4:(block + 1) * 4]}) == 1
    with pytest.raises(ValueError):
        EpisodeSampler(pool, {"speakers": 7, "k": 2, "min_gap": 150, "temperature": 0.05, "weight": 1.0})


def test_episode_infonce_rewards_voice_shared_across_a_speakers_lines_and_ignores_line_content():
    torch.manual_seed(0)
    speakers, k, dim = 8, 4, 32
    voice = torch.randn(speakers, 1, 1, dim)
    content = torch.randn(speakers, 2, k, dim) * 2.0  # every line says something else
    good = (voice * 3 + content).reshape(-1, dim)
    loss_good, stats = episode_infonce(good, speakers, k, 0.1)
    loss_bad, _ = episode_infonce(content.reshape(-1, dim), speakers, k, 0.1)
    assert loss_good < loss_bad and float(stats["episode_accuracy"]) == 1.0
    embeddings = good.clone().requires_grad_(True)
    episode_infonce(embeddings, speakers, k, 0.1)[0].backward()
    assert embeddings.grad.abs().sum() > 0


def test_episode_infonce_uses_the_other_speakers_same_half_episodes_as_negatives():
    speakers, k, dim = 3, 2, 8
    basis = torch.eye(dim)
    halves = torch.stack([basis[0], basis[0], basis[1], basis[1], basis[2], basis[2]])  # each speaker's halves agree
    clean = halves[:, None, :].expand(-1, k, -1).reshape(-1, dim)
    clash = clean.reshape(speakers, 2, k, dim).clone()
    clash[1, 0] = basis[0]  # speaker 1's first half looks exactly like speaker 0's first half
    loss_clean, _ = episode_infonce(clean, speakers, k, 0.1)
    loss_clash, stats = episode_infonce(clash.reshape(-1, dim), speakers, k, 0.1)
    assert loss_clash > loss_clean + 1.0 and float(stats["episode_accuracy"]) < 1.0


def test_file_pool_episodes_take_every_line_from_a_different_topic(tmp_path: Path):
    import json

    from episodes import episode_pool_from_file

    rows = [{"author": f"a{i}", "topic": f"t{j % 5}", "text": f"a{i}_{j}"} for i in range(4) for j in range(10)]
    rows += [{"author": "mono", "topic": "t0", "text": f"mono_{j}"} for j in range(10)]  # one topic: never usable
    path = tmp_path / "episodes.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    pool = episode_pool_from_file(str(path))
    assert pool["a0"]["a0_7"] == "t2"
    texts = EpisodeSampler(pool, {"speakers": 3, "k": 2, "min_gap": 150, "temperature": 0.05, "weight": 1.0}).sample()
    assert len(texts) == 12 and not any(t.startswith("mono") for t in texts)
    topic = {r["text"]: r["topic"] for r in rows}
    for block in range(3):
        assert len({topic[t] for t in texts[block * 4:(block + 1) * 4]}) == 4


def test_centering_removes_a_shared_domain_direction():
    shared = torch.tensor([10.0, 0, 0, 0])
    voices = torch.tensor([[0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    lines = torch.stack([shared + voices[s] + 0.01 * torch.randn(4) for s in (0, 0, 1, 1) for _ in range(2)])
    plain, _ = episode_infonce(lines, 2, 2, 0.1)
    centered, stats = episode_infonce(lines, 2, 2, 0.1, center=True)
    assert centered < plain and stats["episode_accuracy"] == 1.0
