"""预处理脚本：一次性tokenize所有数据并缓存。"""

import pickle
from pathlib import Path
from transformers import AutoTokenizer
from data import load_data, TextDataset, MAX_PER_SPEAKER, MIN_SENTENCES
from model import MODEL_PATH

CACHE_DIR = Path(__file__).resolve().parent / "cache"
MAX_LEN = 128


def preprocess_dataset(texts, labels, tokenizer, max_len=MAX_LEN):
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


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading data...")
    train_ds, val_acc_ds, val_ds, num_train_speakers, info = load_data()

    # 预处理train数据
    print("\nPreprocessing train dataset...")
    train_input_ids, train_attention_masks = preprocess_dataset(
        train_ds.texts, train_ds.labels, tokenizer
    )
    train_cache = {
        "input_ids": train_input_ids,
        "attention_masks": train_attention_masks,
        "labels": train_ds.labels,
    }
    train_cache_path = CACHE_DIR / "train_cache.pkl"
    with open(train_cache_path, "wb") as f:
        pickle.dump(train_cache, f)
    print(f"  => saved: {train_cache_path}")

    # 预处理val_acc数据
    print("\nPreprocessing val_acc dataset...")
    val_acc_input_ids, val_acc_attention_masks = preprocess_dataset(
        val_acc_ds.texts, val_acc_ds.labels, tokenizer
    )
    val_acc_cache = {
        "input_ids": val_acc_input_ids,
        "attention_masks": val_acc_attention_masks,
        "labels": val_acc_ds.labels,
    }
    val_acc_cache_path = CACHE_DIR / "val_acc_cache.pkl"
    with open(val_acc_cache_path, "wb") as f:
        pickle.dump(val_acc_cache, f)
    print(f"  => saved: {val_acc_cache_path}")

    # 预处理val数据
    print("\nPreprocessing val dataset...")
    val_input_ids, val_attention_masks = preprocess_dataset(
        val_ds.texts, val_ds.labels, tokenizer
    )
    val_cache = {
        "input_ids": val_input_ids,
        "attention_masks": val_attention_masks,
        "labels": val_ds.labels,
    }
    val_cache_path = CACHE_DIR / "val_cache.pkl"
    with open(val_cache_path, "wb") as f:
        pickle.dump(val_cache, f)
    print(f"  => saved: {val_cache_path}")

    # 预处理all_train数据（用于silhouette）
    print("\nPreprocessing all_train dataset...")
    all_train_texts = train_ds.texts + val_acc_ds.texts
    all_train_labels = train_ds.labels + val_acc_ds.labels
    all_train_input_ids, all_train_attention_masks = preprocess_dataset(
        all_train_texts, all_train_labels, tokenizer
    )
    all_train_cache = {
        "input_ids": all_train_input_ids,
        "attention_masks": all_train_attention_masks,
        "labels": all_train_labels,
    }
    all_train_cache_path = CACHE_DIR / "all_train_cache.pkl"
    with open(all_train_cache_path, "wb") as f:
        pickle.dump(all_train_cache, f)
    print(f"  => saved: {all_train_cache_path}")

    # 保存speaker元数据
    meta_cache_path = CACHE_DIR / "meta_cache.pkl"
    with open(meta_cache_path, "wb") as f:
        pickle.dump({"num_train_speakers": num_train_speakers, "info": info}, f)
    print(f"  => saved: {meta_cache_path}")

    print("\nPreprocessing complete!")


