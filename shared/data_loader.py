"""统一数据加载接口：消除 load_data/load_cached_data 的重复和不一致。"""

import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config, DataConfig
from .data_types import DatasetTuple, DatasetInfo


class TextDataset(Dataset):
    """原始文本数据集。"""
    def __init__(self, texts: list, labels: list):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class TokenizedDataset(Dataset):
    """预处理后的 tokenized 数据集。"""
    def __init__(self, input_ids: list, attention_masks: list, labels: list):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


class PKSampler(torch.utils.data.Sampler):
    """每 batch 选 P 个说话人各 K 条。"""
    def __init__(self, labels, p: int, k: int):
        self.labels = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
        self.p, self.k = p, k
        self.speaker_to_indices = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.speaker_to_indices[lbl].append(idx)
        self.speakers = list(self.speaker_to_indices.keys())

    def __iter__(self):
        import random
        speakers = self.speakers[:]
        random.shuffle(speakers)
        shuffled = {}
        for s in speakers:
            indices = self.speaker_to_indices[s][:]
            random.shuffle(indices)
            shuffled[s] = indices
        p, k = self.p, self.k
        for start in range(0, len(speakers) - len(speakers) % p, p):
            group = speakers[start:start + p]
            for s in group:
                pool = shuffled[s]
                for j in range(k):
                    yield pool[j % len(pool)]

    def __len__(self):
        return (len(self.speakers) // self.p) * self.p * self.k


class DataLoader:
    """统一的数据加载器：处理缓存和非缓存情况。"""

    def __init__(self, config: Config):
        self.config = config
        self.data_cfg = config.data
        self.rng = np.random.RandomState(self.data_cfg.seed)

    def load(self) -> DatasetTuple:
        """加载数据（自动选择缓存或非缓存）。"""
        if self.config.train.use_core:
            return self._load_core()
        return self._load_full()

    def _load_full(self) -> DatasetTuple:
        """加载全量数据。"""
        if self.config.train.use_cache:
            return self._load_cached_full()
        return self._load_raw_full()

    def _load_core(self) -> DatasetTuple:
        """加载核心数据。"""
        if self.config.train.use_cache:
            return self._load_cached_core()
        return self._load_raw_core()

    def _load_raw_full(self) -> DatasetTuple:
        """从数据库加载全量数据。"""
        conn = sqlite3.connect(self.data_cfg.db_path)

        rows = conn.execute(
            "SELECT speaker, COUNT(*) as cnt FROM dialogues "
            "WHERE speaker != '？？？' "
            "AND LENGTH(origin_text) > 4 "
            "AND origin_text IS NOT NULL "
            "AND LENGTH(TRIM(origin_text)) > 0 "
            "GROUP BY speaker HAVING cnt >= ? "
            "ORDER BY speaker",
            (self.data_cfg.min_sentences,),
        ).fetchall()

        speakers = [r[0] for r in rows]
        N = len(speakers)
        n_train = int(self.data_cfg.train_split * N)
        train_speakers = speakers[:n_train]
        val_speakers = speakers[n_train:]

        print(f"说话人总数: {N}  train: {len(train_speakers)}  val: {len(val_speakers)}")

        # 加载训练数据
        tr_texts_all, tr_labels_all = self._fetch_texts(conn, train_speakers, 0)
        perm = self.rng.permutation(len(tr_texts_all))
        n_tr = int(self.data_cfg.val_split * len(tr_texts_all))

        train_texts = [tr_texts_all[i] for i in perm[:n_tr]]
        train_labels = [tr_labels_all[i] for i in perm[:n_tr]]
        val_acc_texts = [tr_texts_all[i] for i in perm[n_tr:]]
        val_acc_labels = [tr_labels_all[i] for i in perm[n_tr:]]

        # 加载验证数据
        val_texts, val_labels = self._fetch_texts(conn, val_speakers, 0)

        conn.close()

        print(f"train: {len(train_texts)} 句  val_acc: {len(val_acc_texts)} 句")
        print(f"val_sil: {len(val_texts)} 句 ({len(val_speakers)} 人)")

        all_train_texts = train_texts + val_acc_texts
        all_train_labels = train_labels + val_acc_labels

        info = DatasetInfo(
            train_speakers=train_speakers,
            val_speakers=val_speakers,
            num_train_speakers=len(train_speakers),
            num_val_speakers=len(val_speakers),
        )

        return DatasetTuple(
            train=TextDataset(train_texts, train_labels),
            val_acc=TextDataset(val_acc_texts, val_acc_labels),
            val=TextDataset(val_texts, val_labels),
            all_train=TextDataset(all_train_texts, all_train_labels),
            info=info,
        )

    def _load_raw_core(self) -> DatasetTuple:
        """从数据库加载核心数据。"""
        conn = sqlite3.connect(self.data_cfg.db_path)

        sorted_speakers = sorted(self.data_cfg.core_speakers)
        N = len(sorted_speakers)
        n_train = int(self.data_cfg.train_split * N)
        train_speakers = sorted_speakers[:n_train]
        val_speakers = sorted_speakers[n_train:]

        print(f"核心说话人: {N}  train: {len(train_speakers)}  val: {len(val_speakers)}")

        # 加载训练数据
        tr_texts_all, tr_labels_all = self._fetch_texts_core(conn, train_speakers, 0)
        perm = self.rng.permutation(len(tr_texts_all))
        n_tr = int(self.data_cfg.val_split * len(tr_texts_all))

        train_texts = [tr_texts_all[i] for i in perm[:n_tr]]
        train_labels = [tr_labels_all[i] for i in perm[:n_tr]]
        val_acc_texts = [tr_texts_all[i] for i in perm[n_tr:]]
        val_acc_labels = [tr_labels_all[i] for i in perm[n_tr:]]

        # 加载验证数据
        val_texts, val_labels = self._fetch_texts_core(conn, val_speakers, 0)

        conn.close()

        print(f"train: {len(train_texts)} 句  val_acc: {len(val_acc_texts)} 句")
        print(f"val_sil: {len(val_texts)} 句 ({len(val_speakers)} 人)")

        all_train_texts = train_texts + val_acc_texts
        all_train_labels = train_labels + val_acc_labels

        info = DatasetInfo(
            train_speakers=train_speakers,
            val_speakers=val_speakers,
            num_train_speakers=len(train_speakers),
            num_val_speakers=len(val_speakers),
        )

        return DatasetTuple(
            train=TextDataset(train_texts, train_labels),
            val_acc=TextDataset(val_acc_texts, val_acc_labels),
            val=TextDataset(val_texts, val_labels),
            all_train=TextDataset(all_train_texts, all_train_labels),
            info=info,
        )

    def _load_cached_full(self) -> DatasetTuple:
        """从缓存加载全量数据。"""
        cache_dir = self.data_cfg.cache_dir

        def load_cache(name):
            path = cache_dir / f"{name}_cache.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        print("Loading cached data...")
        train_cache = load_cache("train")
        val_acc_cache = load_cache("val_acc")
        val_cache = load_cache("val")
        all_train_cache = load_cache("all_train")
        meta = load_cache("meta")

        def to_ds(cache):
            return TokenizedDataset(
                cache["input_ids"],
                cache["attention_masks"],
                cache["labels"],
            )

        info = DatasetInfo(
            train_speakers=meta["info"]["train"],
            val_speakers=meta["info"]["val"],
            num_train_speakers=meta["num_train_speakers"],
            num_val_speakers=len(meta["info"]["val"]),
        )

        return DatasetTuple(
            train=to_ds(train_cache),
            val_acc=to_ds(val_acc_cache),
            val=to_ds(val_cache),
            all_train=to_ds(all_train_cache),
            info=info,
        )

    def _load_cached_core(self) -> DatasetTuple:
        """从缓存加载核心数据。"""
        cache_dir = self.data_cfg.core_cache_dir

        def load_cache(name):
            path = cache_dir / f"{name}_cache.pkl"
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

        print("Loading cached core data...")
        train_cache = load_cache("train")
        val_acc_cache = load_cache("val_acc")
        val_cache = load_cache("val")
        all_train_cache = load_cache("all_train")
        meta = load_cache("meta")

        def to_ds(cache):
            return TokenizedDataset(
                cache["input_ids"],
                cache["attention_masks"],
                cache["labels"],
            )

        info = DatasetInfo(
            train_speakers=meta["info"]["train"],
            val_speakers=meta["info"]["val"],
            num_train_speakers=meta["num_train_speakers"],
            num_val_speakers=len(meta["info"]["val"]),
        )

        return DatasetTuple(
            train=to_ds(train_cache),
            val_acc=to_ds(val_acc_cache),
            val=to_ds(val_cache),
            all_train=to_ds(all_train_cache),
            info=info,
        )

    def _fetch_texts(self, conn, speaker_list: list[str], label_offset: int):
        """从数据库获取文本。"""
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
            idx = self.rng.permutation(len(all_texts))[:self.data_cfg.max_per_speaker]
            selected = [all_texts[j] for j in idx]
            texts.extend(selected)
            labels.extend([label_offset + i] * len(selected))
        return texts, labels

    def _fetch_texts_core(self, conn, speaker_list: list[str], label_offset: int):
        """从数据库获取核心说话人的文本（处理别名）。"""
        texts, labels = [], []
        for i, speaker in enumerate(speaker_list):
            db_names = [speaker]
            if speaker == "「少女」":
                db_names.append("哥伦比娅")
            all_texts = []
            for name in db_names:
                rows = conn.execute(
                    "SELECT origin_text FROM dialogues "
                    "WHERE speaker = ? AND LENGTH(origin_text) > 4 "
                    "AND origin_text IS NOT NULL AND LENGTH(TRIM(origin_text)) > 0",
                    (name,),
                ).fetchall()
                all_texts.extend(r[0] for r in rows)
            texts.extend(all_texts)
            labels.extend([label_offset + i] * len(all_texts))
        return texts, labels
