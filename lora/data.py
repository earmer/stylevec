"""数据准备：从 genshin.db 查询 ≥100 句说话人，按说话人 70/15/15 划分。"""

import sqlite3
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
SEED = 42
MAX_PER_SPEAKER = 200
MIN_SENTENCES = 100


class TextDataset(Dataset):
    def __init__(self, texts: list, labels: list):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def load_data():
    """
    返回:
        train_ds, val_acc_ds, val_ds, test_ds: TextDataset
        num_train_speakers: int
        info: dict {"train": [...], "val": [...], "test": [...]}

    说话人级 70/15/15 split：
      - train speakers: 70%，其句子再 80/20 分为 train_ds / val_acc_ds
        （val_acc_ds 用于计算 ArcFace 在已见说话人上的泛化准确率）
      - val speakers:   15%，全部句子 → val_ds（用于 val_sil）
      - test speakers:  15%，全部句子 → test_ds（用于 test_sil）
    """
    conn = sqlite3.connect(DB_PATH)
    rng = np.random.RandomState(SEED)

    # 查询 ≥100 句的说话人（排除？？？），按名称排序保证可复现
    rows = conn.execute(
        "SELECT speaker, COUNT(*) as cnt FROM dialogues "
        "WHERE speaker != '？？？' "
        "AND LENGTH(origin_text) > 4 "
        "AND origin_text IS NOT NULL "
        "AND LENGTH(TRIM(origin_text)) > 0 "
        "GROUP BY speaker HAVING cnt >= ? "
        "ORDER BY speaker",
        (MIN_SENTENCES,),
    ).fetchall()
    speakers = [r[0] for r in rows]
    N = len(speakers)

    n_train = int(0.70 * N)
    n_val_end = int(0.85 * N)
    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_val_end]
    test_speakers = speakers[n_val_end:]

    print(f"说话人总数: {N}  train: {len(train_speakers)}  val: {len(val_speakers)}  test: {len(test_speakers)}")

    def fetch_texts(speaker_list, label_offset=0):
        texts, labels = [], []
        for i, speaker in enumerate(speaker_list):
            result = conn.execute(
                "SELECT origin_text FROM dialogues "
                "WHERE speaker = ? "
                "AND LENGTH(origin_text) > 4 "
                "AND origin_text IS NOT NULL "
                "AND LENGTH(TRIM(origin_text)) > 0",
                (speaker,),
            ).fetchall()
            all_texts = [r[0] for r in result]
            idx = rng.permutation(len(all_texts))[:MAX_PER_SPEAKER]
            selected = [all_texts[j] for j in idx]
            texts.extend(selected)
            labels.extend([label_offset + i] * len(selected))
        return texts, labels

    # train speakers：取句子后再 80/20 切分
    tr_texts_all, tr_labels_all = fetch_texts(train_speakers, label_offset=0)
    n_total = len(tr_texts_all)
    perm = rng.permutation(n_total)
    n_tr = int(0.80 * n_total)
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    train_texts = [tr_texts_all[i] for i in tr_idx]
    train_labels = [tr_labels_all[i] for i in tr_idx]
    val_acc_texts = [tr_texts_all[i] for i in va_idx]
    val_acc_labels = [tr_labels_all[i] for i in va_idx]

    # val/test speakers：全部句子用于 silhouette
    val_texts, val_labels = fetch_texts(val_speakers, label_offset=0)
    test_texts, test_labels = fetch_texts(test_speakers, label_offset=0)

    conn.close()

    print(f"train: {len(train_texts)} 句  val_acc: {len(val_acc_texts)} 句")
    print(f"val_sil: {len(val_texts)} 句 ({len(val_speakers)} 人)  test_sil: {len(test_texts)} 句 ({len(test_speakers)} 人)")

    info = {"train": train_speakers, "val": val_speakers, "test": test_speakers}
    return (
        TextDataset(train_texts, train_labels),
        TextDataset(val_acc_texts, val_acc_labels),
        TextDataset(val_texts, val_labels),
        TextDataset(test_texts, test_labels),
        len(train_speakers),
        info,
    )


def make_collate_fn(tokenizer, max_len: int = 128):
    def collate(batch):
        import torch
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long)
    return collate
