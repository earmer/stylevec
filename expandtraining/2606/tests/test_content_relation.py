from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report import content_relation

HANZI = "风起地大树璃月港码头稻妻城祭典须弥教令院枫丹歌剧纳塔火山蒙德酒馆雪山遗迹深渊骑士团冒险家协会商人旅店花园"


def lines(pull: str, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """8 speakers x 10 origin_vs_para triplets. Each of 40 topics is shared by exactly two lines (random
    speakers, so no habitual topic). The two share a topic direction (`attract`), sit on its opposite poles
    (`repel`: same topic pushed apart), or ignore it (`blind`); plus a speaker direction and noise."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(80)
    topic_of = order // 2  # line -> topic, two lines per topic
    pole = np.where(order % 2 == 0, 1.0, -1.0 if pull == "repel" else 1.0) * (pull != "blind")
    names = ["".join(rng.choice(list(HANZI), 4, replace=False)) for _ in range(40)]
    rows, vectors = [], []
    for line in range(80):
        speaker, topic = line // 10, int(topic_of[line])
        style = np.zeros(48)
        style[speaker] = 1.0
        style[8 + topic] = pole[line]
        style += rng.normal(0.0, 0.2, 48)
        for role, preview in (("anchor", f"{names[topic]}第{line}句"), ("positive", f"他句{line}"),
                              ("negative", f"{names[topic]}之事")):
            rows.append({"triplet_id": f"t{line}", "role": role, "rule": "origin_vs_para",
                         "speaker": "" if role == "negative" else f"s{speaker}", "text_preview": preview})
            vectors.append(style if role == "anchor" else rng.normal(0.0, 1.0, 48))
    return pd.DataFrame(rows), np.stack(vectors)


def test_content_relation_reads_the_sign_of_the_line_level_pull() -> None:
    attract, repel, blind = (content_relation(*lines(pull)) for pull in ("attract", "repel", "blind"))
    assert attract["content_rel_line"] > 0.15 and attract["content_rel_line_top"] > 0.03
    assert repel["content_rel_line"] < -0.15 and repel["content_rel_line_top"] < -0.03
    assert abs(blind["content_rel_line"]) < 0.05


def test_content_relation_needs_paraphrased_lines() -> None:
    rows, vectors = lines("attract")
    assert content_relation(rows.assign(rule="same_speaker_real"), vectors) == {}
