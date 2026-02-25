"""预处理脚本：一次性 tokenize 48人核心角色数据并缓存。"""

import pickle
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_core import load_core_data
from model import MODEL_PATH

CACHE_DIR = Path(__file__).resolve().parent / "cache_core"
MAX_LEN = 128


def preprocess_dataset(texts, tokenizer, max_len=MAX_LEN):
    """Tokenize 所有文本，返回 (input_ids_list, attention_masks_list)。"""
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


def save_cache(name, input_ids, attention_masks, labels):
    path = CACHE_DIR / f"{name}_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump({
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "labels": labels,
        }, f)
    print(f"  => saved: {path}")


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading core data...")
    train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_core_data()

    for name, ds in [("train", train_ds), ("val_acc", val_acc_ds),
                     ("val", val_ds), ("all_train", all_train_ds)]:
        print(f"\nPreprocessing {name}...")
        ids, masks = preprocess_dataset(ds.texts, tokenizer)
        save_cache(name, ids, masks, ds.labels)

    meta_path = CACHE_DIR / "meta_cache.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({"num_train_speakers": num_train_speakers, "info": info}, f)
    print(f"  => saved: {meta_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
