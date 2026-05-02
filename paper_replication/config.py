"""Exact hyperparameters from the StyleDistance paper (Appendix D, Table 5)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Model ---
    model_name: str = str(Path(__file__).resolve().parent.parent / "base-models" / "roberta-base")
    max_seq_len: int = 512

    # --- LoRA (Section 4.2 + Appendix D) ---
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_target_modules: str = "all-linear"

    # --- Training (Appendix D) ---
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    triplet_margin: float = 0.1
    max_epochs: int = 20  # early stopping will cut this short

    # --- Early stopping ---
    early_stopping_patience: int = 1
    early_stopping_threshold: float = 0.0

    # --- Data ---
    dataset_name: str = "StyleDistance/synthstel"
    num_triplets_per_feature: int = 100 * 99  # 100 pairs → ~9900 triplets per feature
    train_val_split: float = 0.9

    # --- Hardware ---
    dtype: str = "bfloat16"
    num_workers: int = 4

    # --- Output ---
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "checkpoints")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / ".cache")
