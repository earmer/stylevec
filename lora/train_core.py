"""LoRA 风格向量训练（48人核心角色子集）。"""

import argparse
import csv
import math
import pickle
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hidden"))
from evaluate import silhouette, consistency  # noqa: E402

from data import TextDataset, TokenizedDataset, make_collate_fn, cached_collate_fn
import preprocess_core
from model import StyleModel, MODEL_PATH, LORA_R, LORA_ALPHA

DB_PATH = Path(__file__).resolve().parent.parent / "genshin" / "genshin.db"
SEED = 42

# 48 人核心角色（DB 名 → 显示名）
# 「少女」= 哥伦比娅，数据合并
CORE_SPEAKERS = [
    "娜维娅","纳西妲","温迪","阿贝多","茜特菈莉","八重神子","玛拉妮","芙宁娜",
    "玛薇卡","赛诺","艾尔海森","林尼","提纳里","荒泷一斗","那维莱特","钟离",
    "枫原万叶","迪希雅","宵宫","恰斯卡","希诺宁","胡桃","莫娜","莱欧斯利",
    "夜兰","奈芙尔","凯亚","神里绫华","妮露","刻晴","柯莱","菈乌玛",
    "珊瑚宫心海","安柏","琴","「少女」","可莉","香菱","卡维","卡齐娜",
    "欧洛伦","基尼奇","砂糖","烟绯","夏洛蒂","魈","雅珂达","菲林斯",
]


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = detect_device()
EPOCHS = 20
LR = 2e-4
MAX_LEN = 128


def load_core_data():
    """加载 48 人核心角色数据，85/15 说话人级 split。"""
    conn = sqlite3.connect(DB_PATH)
    rng = np.random.RandomState(SEED)

    # 按名称排序保证可复现
    sorted_speakers = sorted(CORE_SPEAKERS)
    N = len(sorted_speakers)
    n_train = int(0.85 * N)
    train_speakers = sorted_speakers[:n_train]
    val_speakers = sorted_speakers[n_train:]

    print(f"核心说话人: {N}  train: {len(train_speakers)}  val: {len(val_speakers)}")

    def fetch_texts(speaker_list, label_offset=0):
        texts, labels = [], []
        for i, speaker in enumerate(speaker_list):
            # 「少女」= 哥伦比娅，查两个名字合并
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

    # train speakers 句子 80/20 split
    tr_all_texts, tr_all_labels = fetch_texts(train_speakers)
    perm = rng.permutation(len(tr_all_texts))
    n_tr = int(0.80 * len(perm))

    train_texts = [tr_all_texts[i] for i in perm[:n_tr]]
    train_labels = [tr_all_labels[i] for i in perm[:n_tr]]
    val_acc_texts = [tr_all_texts[i] for i in perm[n_tr:]]
    val_acc_labels = [tr_all_labels[i] for i in perm[n_tr:]]

    val_texts, val_labels = fetch_texts(val_speakers)
    conn.close()

    print(f"train: {len(train_texts)} 句  val_acc: {len(val_acc_texts)} 句")
    print(f"val_sil: {len(val_texts)} 句 ({len(val_speakers)} 人)")

    all_train_texts = train_texts + val_acc_texts
    all_train_labels = train_labels + val_acc_labels

    info = {"train": train_speakers, "val": val_speakers}
    return (
        TextDataset(train_texts, train_labels),
        TextDataset(val_acc_texts, val_acc_labels),
        TextDataset(val_texts, val_labels),
        TextDataset(all_train_texts, all_train_labels),
        len(train_speakers),
        info,
    )


def load_cached_core_data():
    """从缓存加载预处理的 tokenized 数据。"""
    cache_dir = preprocess_core.CACHE_DIR

    def load_cache(name):
        with open(cache_dir / f"{name}_cache.pkl", "rb") as f:
            return pickle.load(f)

    train_cache = load_cache("train")
    val_acc_cache = load_cache("val_acc")
    val_cache = load_cache("val")
    all_train_cache = load_cache("all_train")
    meta = load_cache("meta")

    def to_ds(cache):
        return TokenizedDataset(cache["input_ids"], cache["attention_masks"], cache["labels"])

    return (
        to_ds(train_cache),
        to_ds(val_acc_cache),
        to_ds(val_cache),
        to_ds(all_train_cache),
        meta["num_train_speakers"],
        meta["info"],
    )


def save_checkpoint(model: StyleModel, run_ts: str, epoch: int, ckpt_dir: Path):
    name = f"{run_ts}-epoch-{epoch:02d}"
    ckpt_path = ckpt_dir / name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.base.save_pretrained(str(ckpt_path))
    torch.save(model.style_head.state_dict(), ckpt_path / "style_head.pt")
    torch.save(model.arcface_head.state_dict(), ckpt_path / "arcface_head.pt")
    print(f"  => saved: {ckpt_path}")


def collect_embeddings(model: StyleModel, loader: DataLoader):
    model.eval()
    vecs, labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, lbl in loader:
            v = model.encode(input_ids.to(DEVICE, non_blocking=True), attention_mask.to(DEVICE, non_blocking=True))
            vecs.append(v.cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(vecs), np.concatenate(labels)


def compute_acc(model: StyleModel, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, lbl in loader:
            lbl = lbl.to(DEVICE, non_blocking=True)
            style_norm = model.encode(input_ids.to(DEVICE, non_blocking=True), attention_mask.to(DEVICE, non_blocking=True))
            logits = model.arcface_head(style_norm, lbl)
            correct += (logits.argmax(dim=-1) == lbl).sum()
            total += len(lbl)
    return float(correct.item() / total) if total > 0 else 0.0


def main():
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high")
    use_amp = DEVICE.type == "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=LORA_R)
    parser.add_argument("--alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessed cached data")
    args = parser.parse_args()
    rank, alpha, num_workers = args.rank, args.alpha, args.workers

    if args.batch is not None:
        batch = args.batch
    elif DEVICE.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        batch = max(1, (int(free_bytes / 1e9) // 8) * 8 or 1)
    else:
        import psutil
        batch = max(1, (int(psutil.virtual_memory().available / 1e9) // 8) * 8 or 1)

    grad_accum = args.grad if args.grad is not None else 1

    results_csv = Path(__file__).resolve().parent / f"results_core_r{rank}.csv"
    ckpt_dir = Path(__file__).resolve().parent / f"checkpoints_core_r{rank}"

    # 数据
    if not args.no_cache:
        if not (preprocess_core.CACHE_DIR / "train_cache.pkl").exists():
            print("Cache not found, preprocessing...")
            preprocess_core.main()
        print("Loading from cache...")
        train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_cached_core_data()
        collate = cached_collate_fn
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_core_data()
        collate = make_collate_fn(tokenizer, MAX_LEN)

    print(f"num_train_speakers = {num_train_speakers}")

    _pw = num_workers > 0
    train_loader     = DataLoader(train_ds,     batch_size=batch, shuffle=True,  collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    val_acc_loader   = DataLoader(val_acc_ds,   batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    val_loader       = DataLoader(val_ds,       batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    all_train_loader = DataLoader(all_train_ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)

    model = StyleModel(num_train_speakers, lora_r=rank, lora_alpha=alpha).to(DEVICE)
    if args.grad is not None:
        model.base.gradient_checkpointing_enable()
    model.base.print_trainable_parameters()
    compiled_model = torch.compile(model) if DEVICE.type == "cuda" else model

    steps_per_epoch = math.ceil(len(train_ds) / batch)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_opt_steps = opt_steps_per_epoch * EPOCHS
    warmup_steps = math.ceil(total_opt_steps * 0.05)
    print(f"device={DEVICE}  batch={batch}  grad_accum={grad_accum}  effective={batch * grad_accum}")
    print(f"num_workers={num_workers}  opt_steps/epoch={opt_steps_per_epoch}  total={total_opt_steps}  warmup={warmup_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        fused=DEVICE.type == "cuda",
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss",
            "train_sil", "val_sil",
            "train_cons", "val_cons",
            "train_acc", "val_acc",
        ])

    run_ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Legend: sil=silhouette, cons=consistency, acc=ArcFace accuracy, tr=train, va=val")
    print()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (input_ids, attention_mask, lbl) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            lbl = lbl.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, _, loss = compiled_model(input_ids, attention_mask, lbl)
            (loss / grad_accum).backward()
            total_loss += loss.item()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)

        tr_vecs, tr_labels = collect_embeddings(model, all_train_loader)
        va_vecs, va_labels = collect_embeddings(model, val_loader)

        tr_sil  = silhouette(tr_vecs, tr_labels)
        va_sil  = silhouette(va_vecs, va_labels)
        tr_cons = consistency(tr_vecs, tr_labels, num_train_speakers)
        va_cons = consistency(va_vecs, va_labels, len(info["val"]))

        tr_acc = compute_acc(model, train_loader)
        va_acc = compute_acc(model, val_acc_loader)

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"sil  tr={tr_sil:+.4f}  va={va_sil:+.4f} | "
            f"acc  tr={tr_acc:.3f}  va={va_acc:.3f}"
        )

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.4f}",
                f"{tr_sil:.4f}", f"{va_sil:.4f}",
                f"{tr_cons:.4f}", f"{va_cons:.4f}",
                f"{tr_acc:.4f}", f"{va_acc:.4f}",
            ])

        save_checkpoint(model, run_ts, epoch, ckpt_dir)


if __name__ == "__main__":
    main()
