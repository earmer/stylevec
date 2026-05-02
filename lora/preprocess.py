"""预处理脚本：一次性tokenize所有数据并缓存。"""

import argparse
import pickle
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared import Config, DataLoader as SharedDataLoader


def preprocess_dataset(texts, tokenizer, max_len: int):
    """Tokenize所有文本并返回缓存数据。"""
    print(f"Tokenizing {len(texts)} texts...")
    all_input_ids = []
    all_attention_masks = []
    for i, text in enumerate(texts):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(texts)}")
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        all_input_ids.append(enc["input_ids"].squeeze(0))
        all_attention_masks.append(enc["attention_mask"].squeeze(0))
    return all_input_ids, all_attention_masks


def save_cache(cache_dir, name, input_ids, attention_masks, labels):
    """保存缓存文件。"""
    path = cache_dir / f"{name}_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
        }, f)
    print(f"  => saved: {path}")


def main(core: bool = False):
    """主预处理函数。"""
    # 从配置读取参数
    config = Config()
    config.train.use_core = core
    config.train.use_cache = False  # Load from raw database, not cache
    cache_dir = config.data.core_cache_dir if core else config.data.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config.model.model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    data_loader = SharedDataLoader(config)
    datasets = data_loader.load()

    # 预处理每个数据集
    for name, ds in [("train", datasets.train), ("val_acc", datasets.val_acc),
                     ("val", datasets.val), ("all_train", datasets.all_train)]:
        print(f"\nPreprocessing {name}...")
        ids, masks = preprocess_dataset(ds.texts, tokenizer, config.data.max_len)
        save_cache(cache_dir, name, ids, masks, ds.labels)

    # 保存元数据
    meta_path = cache_dir / "meta_cache.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "num_train_speakers": datasets.info.num_train_speakers,
            "info": {
                "train": datasets.info.train_speakers,
                "val": datasets.info.val_speakers,
            },
        }, f)
    print(f"  => saved: {meta_path}")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", action="store_true", help="Preprocess 48-person core subset")
    args = parser.parse_args()
    main(core=args.core)

