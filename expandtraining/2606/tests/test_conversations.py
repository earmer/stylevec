from __future__ import annotations

import copy
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_triplets import build_genshin_triplets
from config import load_config, validate_config
from conversation_batches import conversation_stats, group_by_conversation
from import_genshin import load_genshin_v2_samples
from records import TripletRecord

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dryrun.yaml"
SPEAKERS = ["甲", "乙", "丙"]

# talk -> [(speaker, line)], turns in order. Talk 3 has 乙 echo 甲's line; 派蒙 is a hub.
TALKS = {
    1: [("甲", "今天的风很大，船怕是走不了"), ("乙", "那就再等一天吧"), ("派蒙", "派蒙肚子饿了"), ("丙", "我去码头看看情况")],
    2: [("甲", "这把剑是祖父留下的"), ("丙", "剑上的花纹真特别"), ("乙", "听说是稻妻的工艺")],
    3: [("甲", "你见过深渊教团吗"), ("乙", "你见过深渊教团吗？"), ("丙", "别提那些家伙了")],
    4: [("甲", "晚饭想吃什么"), ("乙", "随便做点就好"), ("丙", "我请客，去万民堂")],
    5: [("甲", "雨停了就出发"), ("乙", "记得带上地图"), ("丙", "地图在我这里")],
}


def make_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "create table dialogues (id integer primary key, talk_id integer, turn_id integer, N text,"
        " speaker text not null, origin_text text, para_text text, similarity real, dup_of integer)"
    )
    row_id = 0
    for talk, turns in TALKS.items():
        for turn, (speaker, line) in enumerate(turns):
            row_id += 1
            connection.execute(
                "insert into dialogues values (?, ?, ?, '[]', ?, ?, ?, 0.9, null)",
                (row_id, talk, talk * 100 + turn, speaker, line, line + "（平述）"),
            )
    # A duplicate row: skipped entirely.
    connection.execute("insert into dialogues values (99, 1, 199, '[]', '甲', '重复的一句', '重复', 0.9, 1)")
    connection.commit()
    connection.close()


@pytest.fixture()
def v2_config(tmp_path: Path) -> dict:
    db = tmp_path / "genshin_v2.db"
    make_db(db)
    config = copy.deepcopy(load_config(CONFIG))
    config["paths"]["genshin_v2_db"] = str(db)
    config["data"].update({"genshin_source": "v2", "genshin_lines_per_speaker": 0, "heldout_speakers": 0,
                           "genshin_scene_neighbors": 3, "genshin_echo_jaccard": 0.5})
    return config


def test_v2_neighbours_are_the_anchors_own_conversation(v2_config: dict) -> None:
    samples, counts = load_genshin_v2_samples(v2_config, SPEAKERS)
    assert {s.text for s in samples}.isdisjoint({"重复的一句"})
    by_text = {s.text: s for s in samples}
    talk_of = {line: talk for talk, turns in TALKS.items() for _speaker, line in turns}
    for sample in samples:
        assert sample.meta["talk_id"] == talk_of[sample.text]
        for speaker, text, _distance in sample.meta["scene_neighbors"]:
            assert speaker != sample.style_id and talk_of[text] == sample.meta["talk_id"]
    # Non-top speakers can be negatives; hubs only after every non-hub line.
    first = by_text["今天的风很大，船怕是走不了"].meta["scene_neighbors"]
    assert [n[0] for n in first] == ["乙", "丙", "派蒙"]
    # 乙 echoing 甲's question is not offered as a negative for it.
    echo = by_text["你见过深渊教团吗"].meta["scene_neighbors"]
    assert [n[1] for n in echo] == ["别提那些家伙了"]


def test_v2_train_anchors_never_see_validation_lines(v2_config: dict) -> None:
    samples, _ = load_genshin_v2_samples(v2_config, SPEAKERS)
    val_texts = {s.text for s in samples if s.split == "val"}
    assert val_texts
    for sample in samples:
        if sample.split == "train":
            assert not {n[1] for n in sample.meta["scene_neighbors"]} & val_texts


def test_v2_far_positives_come_from_other_conversations(v2_config: dict) -> None:
    samples, _ = load_genshin_v2_samples(v2_config, SPEAKERS)
    rows = build_genshin_triplets(v2_config, samples, "train") + build_genshin_triplets(v2_config, samples, "val")
    talk_of = {line: talk for talk, turns in TALKS.items() for _speaker, line in turns}
    assert {r.rule for r in rows} >= {"same_speaker_real", "same_speaker_topic_matched"}
    for row in rows:
        assert row.meta["talk_id"] == talk_of[row.anchor_text]
        if row.rule in {"same_speaker_real", "same_speaker_topic_matched", "origin_vs_para"}:
            assert talk_of[row.positive_text] != row.meta["talk_id"]
        if row.rule == "same_speaker_topic_matched":
            assert talk_of[row.negative_text] == row.meta["talk_id"]


def triplet(rule: str, speaker: str, talk: int | None, tag: str) -> TripletRecord:
    meta = {"speaker": speaker, **({"talk_id": talk} if talk is not None else {})}
    return TripletRecord(f"a{tag}", f"p{tag}", f"n{tag}", rule, 1.0, 0.1, "train", meta)


def test_grouping_swaps_within_a_rule_and_keeps_counts() -> None:
    pool = [triplet("same_speaker_topic_matched", s, t, f"{s}{t}") for t in range(20) for s in "ABCD"]
    pool += [triplet("same_speaker_real", s, t, f"r{s}{t}") for t in range(20) for s in "ABCD"]
    other = [triplet("generated_to_target", "X", None, f"g{i}") for i in range(40)]
    rows = (pool + other)[:]
    random.Random(0).shuffle(rows)
    out = group_by_conversation(rows, pool, batch_size=16, group=3, seed=0)
    assert len(out) == len(rows)
    for start in range(0, len(rows), 16):
        before, after = rows[start : start + 16], out[start : start + 16]
        assert Counter(r.rule for r in before) == Counter(r.rule for r in after)
        assert [r for r in before if "talk_id" not in r.meta] == [r for r in after if "talk_id" not in r.meta]
    assert conversation_stats(out, 16)["grouped_share"] > conversation_stats(rows, 16)["grouped_share"] + 0.3
    assert group_by_conversation(rows, pool, 16, 0, 0) == rows


def test_conversation_group_needs_the_curriculum() -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["train"]["conversation_group"] = 3
    config.setdefault("curriculum", {})["enabled"] = False
    with pytest.raises(ValueError, match="curriculum"):
        validate_config(config)
    config["curriculum"]["enabled"] = True
    validate_config(config)
    config["train"]["conversation_group"] = 1
    with pytest.raises(ValueError, match="conversation_group"):
        validate_config(config)
    config["train"]["conversation_group"] = 0
    config["data"]["genshin_source"] = "v3"
    with pytest.raises(ValueError, match="genshin_source"):
        validate_config(config)
