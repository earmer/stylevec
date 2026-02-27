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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from evaluate import silhouette, consistency  # noqa: E402

from data import (
    load_data, load_core_data, load_cached_data, load_cached_core_data,
    make_collate_fn, cached_collate_fn, TextDataset,
    CACHE_DIR, CORE_CACHE_DIR,
)
import preprocess
from model import StyleModel, MODEL_PATH, LORA_R, LORA_ALPHA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from device import detect_device  # noqa: E402

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
    if model.layer_fusion is not None:
        torch.save(model.layer_fusion.state_dict(), ckpt_path / "layer_fusion.pt")
    if model.attn_pool is not None:
        torch.save(model.attn_pool.state_dict(), ckpt_path / "attn_pool.pt")
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
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessed cached data")
    parser.add_argument("--core", action="store_true", help="Train on 48-person core subset")
    parser.add_argument("--pk", type=int, nargs=2, metavar=("P", "K"))
    parser.add_argument("--fusion-layers", type=int, nargs="+")
    parser.add_argument("--attn-pool", action="store_true")
    args = parser.parse_args()

    if args.pk and args.batch is not None:
        parser.error("--pk and --batch are mutually exclusive")
    rank = args.rank
    alpha = args.alpha
    num_workers = args.workers

    if args.pk:
        batch = args.pk[0] * args.pk[1]
    elif args.batch is not None:
        batch = args.batch
    elif DEVICE.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        batch = max(1, int(free_bytes / 1e9))
        batch = (batch // 8) * 8 or 1
    else:
        import psutil
        batch = max(1, int(psutil.virtual_memory().available / 1e9))
        batch = (batch // 8) * 8 or 1

    grad_accum = args.grad if args.grad is not None else 1

    tag = f"r{rank}"
    if args.pk:
        tag += f"_pk{args.pk[0]}x{args.pk[1]}"
    if args.fusion_layers:
        tag += "_fuse"
    if args.attn_pool:
        tag += "_apool"

    prefix = "core_" if args.core else ""
    results_csv = Path(__file__).resolve().parent / f"results_{prefix}{tag}.csv"
    ckpt_dir = Path(__file__).resolve().parent / f"checkpoints_{prefix}{tag}"
    cache_dir = CORE_CACHE_DIR if args.core else CACHE_DIR

    # 加载数据
    if not args.no_cache:
        if not (cache_dir / "train_cache.pkl").exists():
            print("Cache not found, preprocessing...")
            preprocess.main(core=args.core)
        print("Loading from cache...")
        if args.core:
            train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_cached_core_data()
        else:
            train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_cached_data()
        collate = cached_collate_fn
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if args.core:
            train_ds, val_acc_ds, val_ds, all_train_ds, num_train_speakers, info = load_core_data()
        else:
            train_ds, val_acc_ds, val_ds, num_train_speakers, info = load_data()
            all_train_ds = TextDataset(
                train_ds.texts + val_acc_ds.texts,
                train_ds.labels + val_acc_ds.labels,
            )
        collate = make_collate_fn(tokenizer, MAX_LEN)

    print(f"num_train_speakers = {num_train_speakers}")

    _pw = num_workers > 0
    if args.pk:
        from data import PKSampler
        pk_sampler = PKSampler(train_ds.labels, p=args.pk[0], k=args.pk[1])
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=False, sampler=pk_sampler, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    val_acc_loader   = DataLoader(val_acc_ds,   batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    val_loader       = DataLoader(val_ds,       batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)
    all_train_loader = DataLoader(all_train_ds, batch_size=batch, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=True, persistent_workers=_pw)

    model = StyleModel(
        num_train_speakers, lora_r=rank, lora_alpha=alpha,
        fusion_layers=args.fusion_layers, attn_pool=args.attn_pool,
    ).to(DEVICE)
    if args.grad is not None:
        model.base.gradient_checkpointing_enable()
    model.base.print_trainable_parameters()
    compiled_model = torch.compile(model) if DEVICE.type == "cuda" else model

    steps_per_epoch = len(train_loader)
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
