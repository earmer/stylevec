"""统一数据结构：消除返回值不一致的问题。"""

from dataclasses import dataclass
from typing import NamedTuple
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetInfo:
    """数据集元信息。"""
    train_speakers: list[str]
    val_speakers: list[str]
    num_train_speakers: int
    num_val_speakers: int


class DatasetTuple(NamedTuple):
    """统一的数据集返回格式。"""
    train: Dataset
    val_acc: Dataset      # 用于计算 ArcFace 准确率
    val: Dataset          # 用于计算 silhouette
    all_train: Dataset    # 训练集 + val_acc 的并集
    info: DatasetInfo


@dataclass
class EvalData:
    """评估数据：消除 evaluate_all() 的 9 个参数问题。"""
    vecs: torch.Tensor
    labels: torch.Tensor
    n_classes: int

    @property
    def num_samples(self) -> int:
        return len(self.vecs)


@dataclass
class EvalMetrics:
    """评估指标。"""
    train_sil: float
    train_cons: float
    val_sil: float
    val_cons: float
    gen_sil: float = float('nan')
    gen_cons: float = float('nan')

    def to_dict(self) -> dict:
        return {
            'train_sil': self.train_sil,
            'train_cons': self.train_cons,
            'val_sil': self.val_sil,
            'val_cons': self.val_cons,
            'gen_sil': self.gen_sil,
            'gen_cons': self.gen_cons,
        }
