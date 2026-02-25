"""LoRA 风格向量训练：20 epoch，每 epoch 评估 silhouette + ArcFace 准确率。"""

import argparse
import csv
import math
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

from data import load_data, make_collate_fn, TextDataset
from model import StyleModel, MODEL_PATH

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
            v = model.encode(input_ids.to(DEVICE), attention_mask.to(DEVICE))
            vecs.append(v.cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(vecs), np.concatenate(labels)


def compute_acc(model: StyleModel, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, lbl in loader:
            lbl = lbl.to(DEVICE)
            style_norm = model.encode(input_ids.to(DEVICE), attention_mask.to(DEVICE))
            logits = model.arcface_head(style_norm, lbl)
            correct += (logits.argmax(dim=-1) == lbl).sum().item()
            total += len(lbl)
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad", type=int, default=1)
    args = parser.parse_args()
    rank = args.rank

    if args.batch is not None:
        batch = args.batch
    elif DEVICE.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        batch = max(1, int(free_bytes / 1e9))
        batch = (batch // 8) * 8 or 1
    else:
        import psutil
        batch = max(1, int(psutil.virtual_memory().available / 1e9))
        batch = (batch // 8) * 8 or 1

    grad_accum = args.grad

    results_csv = Path(__file__).resolve().parent / f"results_r{rank}.csv"
    ckpt_dir = Path(__file__).resolve().parent / f"checkpoints_r{rank}"

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_acc_ds, val_ds, num_train_speakers, info = load_data()
    print(f"num_train_speakers = {num_train_speakers}")

    collate = make_collate_fn(tokenizer, MAX_LEN)
    train_loader    = DataLoader(train_ds,    batch_size=batch, shuffle=True,  collate_fn=collate)
    val_acc_loader  = DataLoader(val_acc_ds,  batch_size=batch, shuffle=False, collate_fn=collate)
    val_loader      = DataLoader(val_ds,      batch_size=batch, shuffle=False, collate_fn=collate)

    # train_sil 用全部 train speakers 的句子（train + val_acc 合并）
    all_train_ds = TextDataset(
        train_ds.texts + val_acc_ds.texts,
        train_ds.labels + val_acc_ds.labels,
    )
    all_train_loader = DataLoader(all_train_ds, batch_size=batch, shuffle=False, collate_fn=collate)

    model = StyleModel(num_train_speakers, lora_r=rank).to(DEVICE)
    model.base.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(train_ds) / batch)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_opt_steps = opt_steps_per_epoch * EPOCHS
    warmup_steps = math.ceil(total_opt_steps * 0.05)
    print(f"device={DEVICE}  batch={batch}  grad_accum={grad_accum}  effective={batch * grad_accum}")
    print(f"opt_steps/epoch={opt_steps_per_epoch}  total={total_opt_steps}  warmup={warmup_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
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
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            lbl = lbl.to(DEVICE)
            _, _, loss = model(input_ids, attention_mask, lbl)
            (loss / grad_accum).backward()
            total_loss += loss.item()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
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
