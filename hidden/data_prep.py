"""数据准备：从 genshin.db 查询 top-12 说话人，划分 train/val/gen。"""

import sqlite3
import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
SEED = 42

# 训练组（ranks 1,3,4,5,6,7,10,12）
TRAIN_SPEAKERS = ["派蒙", "娜维娅", "纳西妲", "温迪", "阿贝多", "茜特菈莉", "芙宁娜", "赛诺"]
# 泛化组（ranks 2,8,9,11）
GEN_SPEAKERS = ["旅行者", "八重神子", "玛拉妮", "玛薇卡"]

N_PER_SPEAKER = 100
N_TRAIN = 80
N_VAL = 20


def prepare_data():
    """返回 dict with keys train/val/gen，每个含 texts, labels, speakers。"""
    conn = sqlite3.connect(DB_PATH)
    rng = np.random.RandomState(SEED)

    train_texts, train_labels, train_speakers = [], [], []
    val_texts, val_labels, val_speakers = [], [], []
    gen_texts, gen_labels, gen_speakers = [], [], []

    # 训练组：8 人，label 0-7
    for label, speaker in enumerate(TRAIN_SPEAKERS):
        rows = conn.execute(
            "SELECT para_text FROM dialogues "
            "WHERE speaker = ? AND LENGTH(origin_text) > 4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text)) > 0 "
            "ORDER BY RANDOM()",
            (speaker,),
        ).fetchall()
        texts = [r[0] for r in rows]
        if len(texts) < N_PER_SPEAKER:
            print(f"WARNING: {speaker} only has {len(texts)} texts")
        # 固定 seed 打乱后取前 100
        idx = rng.permutation(len(texts))[:N_PER_SPEAKER]
        selected = [texts[i] for i in idx]

        train_texts.extend(selected[:N_TRAIN])
        train_labels.extend([label] * N_TRAIN)
        train_speakers.extend([speaker] * N_TRAIN)

        val_texts.extend(selected[N_TRAIN:N_PER_SPEAKER])
        val_labels.extend([label] * N_VAL)
        val_speakers.extend([speaker] * N_VAL)

    # 泛化组：4 人，独立 label 0-3
    for label, speaker in enumerate(GEN_SPEAKERS):
        rows = conn.execute(
            "SELECT para_text FROM dialogues "
            "WHERE speaker = ? AND LENGTH(origin_text) > 4 "
            "AND para_text IS NOT NULL AND LENGTH(TRIM(para_text)) > 0 "
            "ORDER BY RANDOM()",
            (speaker,),
        ).fetchall()
        texts = [r[0] for r in rows]
        if len(texts) < N_PER_SPEAKER:
            print(f"WARNING: {speaker} only has {len(texts)} texts")
        idx = rng.permutation(len(texts))[:N_PER_SPEAKER]
        selected = [texts[i] for i in idx]

        gen_texts.extend(selected)
        gen_labels.extend([label] * N_PER_SPEAKER)
        gen_speakers.extend([speaker] * N_PER_SPEAKER)

    conn.close()

    data = {
        "train": {
            "texts": train_texts,
            "labels": np.array(train_labels),
            "speakers": train_speakers,
        },
        "val": {
            "texts": val_texts,
            "labels": np.array(val_labels),
            "speakers": val_speakers,
        },
        "gen": {
            "texts": gen_texts,
            "labels": np.array(gen_labels),
            "speakers": gen_speakers,
        },
    }

    print(f"Data prepared: train={len(train_texts)}, val={len(val_texts)}, gen={len(gen_texts)}")
    print(f"Train speakers: {TRAIN_SPEAKERS}")
    print(f"Gen speakers:   {GEN_SPEAKERS}")
    return data
