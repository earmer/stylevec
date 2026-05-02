"""LoRA 风格向量训练：支持快速测试和完整训练。"""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from shared import Config, DataLoader as SharedDataLoader, EvalData, evaluate_all
from lora.model import StyleModel
from lora.data import make_collate_fn, cached_collate_fn


def save_checkpoint(model: StyleModel, run_ts: str, epoch: int, ckpt_dir: Path):
    """保存模型检查点。"""
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


def collect_embeddings(model: StyleModel, loader: DataLoader, device: torch.device):
    """收集所有样本的嵌入向量。"""
    model.eval()
    vecs, labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, lbl in loader:
            v = model.encode(
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
            )
            vecs.append(v.cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(vecs), np.concatenate(labels)


def compute_acc(model: StyleModel, loader: DataLoader, device: torch.device) -> float:
    """计算 ArcFace 准确率。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, attention_mask, lbl in loader:
            lbl = lbl.to(device, non_blocking=True)
            style_norm = model.encode(
                input_ids.to(device, non_blocking=True),
                attention_mask.to(device, non_blocking=True),
            )
            logits = model.arcface_head(style_norm, lbl)
            correct += (logits.argmax(dim=-1) == lbl).sum()
            total += len(lbl)
    return float(correct.item() / total) if total > 0 else 0.0


def main():
    """主训练函数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--grad", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--dryrun", action="store_true", help="Quick test: 1 epoch, 1 update")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessed cached data")
    parser.add_argument("--core", action="store_true", help="Train on 48-person core subset")
    parser.add_argument("--pk", type=int, nargs=2, metavar=("P", "K"), help="PK sampler (P speakers, K samples)")
    parser.add_argument("--fusion-layers", type=int, nargs="+", help="Layer indices for fusion")
    parser.add_argument("--attn-pool", action="store_true", help="Use attention pooling")
    args = parser.parse_args()

    # 验证互斥参数
    if args.pk and args.batch is not None:
        parser.error("--pk and --batch are mutually exclusive")

    # 从命令行参数创建配置
    config = Config.from_args(args)

    # 处理 dryrun 和 epochs
    if args.dryrun:
        config.train.epochs = 1
        config.train.grad_accum = 1
        print("DRY RUN MODE: 1 epoch, 1 update")
    elif args.epochs is not None:
        config.train.epochs = args.epochs

    device = config.device.device
    use_amp = config.device.use_amp

    # 自动计算批大小
    batch_size = config.auto_batch_size()

    # 加载数据
    print("Loading data...")
    data_loader = SharedDataLoader(config)
    datasets = data_loader.load()

    # 准备 tokenizer 和 collate 函数
    if config.train.use_cache:
        collate_fn = cached_collate_fn
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(config.model.model_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        collate_fn = make_collate_fn(tokenizer, config.data.max_len)

    # 创建 DataLoader
    from lora.data import PKSampler

    if config.train.pk_p is not None:
        pk_sampler = PKSampler(datasets.train.labels, p=config.train.pk_p, k=config.train.pk_k)
        train_loader = DataLoader(
            datasets.train,
            batch_size=batch_size,
            shuffle=False,
            sampler=pk_sampler,
            collate_fn=collate_fn,
            num_workers=config.train.num_workers,
            pin_memory=True,
            persistent_workers=config.train.num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            datasets.train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.train.num_workers,
            pin_memory=True,
            persistent_workers=config.train.num_workers > 0,
        )

    val_acc_loader = DataLoader(
        datasets.val_acc,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
    )
    all_train_loader = DataLoader(
        datasets.all_train,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
    )

    # 创建模型
    model = StyleModel(
        num_train_speakers=datasets.info.num_train_speakers,
        model_path=config.model.model_path,
        hidden_size=config.model.hidden_size,
        style_dim=config.model.style_dim,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        fusion_layers=config.train.fusion_layers,
        use_attn_pool=config.train.use_attn_pool,
        use_grad_ckpt=config.train.use_grad_ckpt,
    ).to(device)

    model.base.print_trainable_parameters()
    compiled_model = torch.compile(model) if device.type == "cuda" else model

    # 计算训练步数
    steps_per_epoch = len(train_loader)
    opt_steps_per_epoch = math.ceil(steps_per_epoch / config.train.grad_accum)
    total_opt_steps = opt_steps_per_epoch * config.train.epochs
    warmup_steps = math.ceil(total_opt_steps * config.train.warmup_ratio)

    print(f"device={device}  batch={batch_size}  grad_accum={config.train.grad_accum}  effective={batch_size * config.train.grad_accum}")
    print(f"num_workers={config.train.num_workers}  opt_steps/epoch={opt_steps_per_epoch}  total={total_opt_steps}  warmup={warmup_steps}")

    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.train.lr,
        fused=device.type == "cuda",
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    # 创建结果文件
    tag = f"r{config.model.lora_r}"
    if config.train.pk_p:
        tag += f"_pk{config.train.pk_p}x{config.train.pk_k}"
    if config.train.fusion_layers:
        tag += "_fuse"
    if config.train.use_attn_pool:
        tag += "_apool"

    prefix = "core_" if config.train.use_core else ""
    results_csv = Path(__file__).resolve().parent / f"results_{prefix}{tag}.csv"
    ckpt_dir = Path(__file__).resolve().parent / f"checkpoints_{prefix}{tag}"

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

    # 训练循环
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (input_ids, attention_mask, lbl) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, _, loss = compiled_model(input_ids, attention_mask, lbl)

            (loss / config.train.grad_accum).backward()
            total_loss += loss.item()

            if (step + 1) % config.train.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # 评估
        tr_vecs, tr_labels = collect_embeddings(model, all_train_loader, device)
        va_vecs, va_labels = collect_embeddings(model, val_loader, device)

        train_eval = EvalData(
            vecs=torch.from_numpy(tr_vecs),
            labels=torch.from_numpy(tr_labels),
            n_classes=datasets.info.num_train_speakers,
        )
        val_eval = EvalData(
            vecs=torch.from_numpy(va_vecs),
            labels=torch.from_numpy(va_labels),
            n_classes=datasets.info.num_val_speakers,
        )
        metrics = evaluate_all(train_eval, val_eval)

        tr_acc = compute_acc(model, train_loader, device)
        va_acc = compute_acc(model, val_acc_loader, device)

        print(
            f"Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"sil  tr={metrics.train_sil:+.4f}  va={metrics.val_sil:+.4f} | "
            f"acc  tr={tr_acc:.3f}  va={va_acc:.3f}"
        )

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.4f}",
                f"{metrics.train_sil:.4f}", f"{metrics.val_sil:.4f}",
                f"{metrics.train_cons:.4f}", f"{metrics.val_cons:.4f}",
                f"{tr_acc:.4f}", f"{va_acc:.4f}",
            ])

        save_checkpoint(model, run_ts, epoch, ckpt_dir)


if __name__ == "__main__":
    main()
