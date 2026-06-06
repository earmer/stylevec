"""Exact hyperparameters from the StyleDistance paper (Appendix D, Table 5)."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- Model ---
    model_name: str = str(Path(__file__).resolve().parent.parent / "base-models" / "xlm-roberta-base")
    max_seq_len: int = 512

    # --- LoRA (Section 4.2 + Appendix D) ---
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    lora_target_modules: str = "all-linear"

    # --- Training (Appendix D) ---
    batch_size: int = 512
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    triplet_margin: float = 0.1
    max_grad_norm: float = 1.0
    max_epochs: int = 20  # early stopping will cut this short

    # --- Early stopping ---
    early_stopping_patience: int = 1
    early_stopping_threshold: float = 0.0

    # --- Data ---
    dataset_name: str = "StyleDistance/synthstel"
    num_triplets_per_feature: int = 100 * 99  # 100 pairs → ~9900 triplets per feature
    max_pairs_per_feature: int = 0  # 0 = no limit; set to e.g. 30 to reduce steps/epoch
    train_val_split: float = 0.9

    # --- Multilingual ---
    language_list: list[str] = field(default_factory=lambda: ["en", "zh", "ja", "fr", "ru"])
    translated_data_path: Path = field(default_factory=lambda:
        Path(__file__).resolve().parent.parent / "datasets" / "msynthstel" / "data" / "translated"
    )

    # --- Hardware ---
    dtype: str = "bfloat16"
    num_workers: int = 4

    # --- Output ---
    output_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "checkpoints")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent / ".cache")


def load_multilingual_pairs(config: Config, split: str) -> list[dict]:
    """Load translated SynthSTEL from all languages, filter null positives.

    Returns list of dicts with keys: positive, negative, feature, feature_clean, lang.
    """
    import pandas as pd

    frames = []
    for lang in config.language_list:
        parquet_path = config.translated_data_path / lang / f"{split}-00000-of-00001.parquet"
        if not parquet_path.exists():
            print(f"  [WARN] Missing: {parquet_path} — skipping {lang}")
            continue
        df = pd.read_parquet(parquet_path)
        df["lang"] = lang
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    before = len(all_data)
    all_data = all_data[all_data["positive"].notna()].copy()
    after = len(all_data)
    if before != after:
        print(f"  Filtered {before - after} null-positive rows (skipped features)")

    return all_data.to_dict("records")
