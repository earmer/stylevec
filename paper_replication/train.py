"""StyleDistance training — exact paper replication.

Usage:
    uv run python train.py --use-local-data --batch 64 --epochs 5
    uv run python train.py --resume                     # resume from checkpoints/latest/
    uv run python train.py --dryrun --use-local-data    # single batch test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

import time

from config import Config
from model import StyleDistance
from triplets import TripletDataset, collate_triplets

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_val_loss(model, loader, criterion, device, max_batches=50):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            a_emb, p_emb, n_emb = model(
                batch["a_ids"].to(device), batch["a_mask"].to(device),
                batch["p_ids"].to(device), batch["p_mask"].to(device),
                batch["n_ids"].to(device), batch["n_mask"].to(device),
            )
            total += criterion(a_emb, p_emb, n_emb).item()
            count += 1
            if count >= max_batches:
                break
    return total / max(count, 1)


def log_lora_histograms(writer: SummaryWriter, model, global_step: int):
    """Log weight and gradient histograms for all LoRA parameters."""
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace(".", "/")
        writer.add_histogram(f"lora/weights/{clean}", param.data, global_step)
        if param.grad is not None:
            writer.add_histogram(f"lora/grads/{clean}", param.grad, global_step)


def build_test_embedding_data(model, test_sentences: list[str], feature_labels: list[str],
                               tokenizer, device) -> tuple[torch.Tensor, list[str], list[str]]:
    """Encode test sentences and return (matrix, labels, hover_texts) for add_embedding."""
    model.eval()
    embs = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(test_sentences), bs):
            batch = test_sentences[i : i + bs]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            emb = model.encode(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            embs.append(emb.cpu())
    mat = torch.cat(embs)  # (N, 768)
    return mat, feature_labels, test_sentences


# ── Checkpoint manager ─────────────────────────────────────────────────────────

class CheckpointManager:
    """Save every N steps, keep top-5 by val_loss, support resume."""

    def __init__(self, base_dir: Path, save_every: int = 500, keep_top: int = 5):
        self.base_dir = Path(base_dir)
        self.save_every = save_every
        self.keep_top = keep_top
        self.ts = int(__import__("time").time())
        self.latest_dir = self.base_dir / "latest"
        self.rankings: list[dict] = []   # sorted by val_loss ascending
        self.history_path = self.base_dir / "history.jsonl"
        self.lock_path = self.base_dir / ".lock"

    def _step_dir(self, global_step: int) -> Path:
        return self.base_dir / f"{self.ts}-step-{global_step:06d}"

    def _read_rankings(self):
        path = self.base_dir / "top5.json"
        if path.exists():
            self.rankings = json.loads(path.read_text())

    def _write_rankings(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "top5.json").write_text(json.dumps(self.rankings, indent=2))

    def save(self, model, optimizer, scheduler, scaler, global_step, epoch,
             train_loss, val_loss, lr, device_type, use_amp):
        """Save checkpoint, manage top-5 rotation."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        step_dir = self._step_dir(global_step)
        step_dir.mkdir(parents=True, exist_ok=True)

        # PEFT adapter
        model.encoder.save_pretrained(str(step_dir))

        # Training state
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "global_step": global_step,
            "epoch": epoch,
            "rng_state": torch.get_rng_state(),
        }, step_dir / "training_state.pt")

        # Latest (always overwrite, for resume)
        if self.latest_dir.exists():
            shutil.rmtree(self.latest_dir)
        shutil.copytree(step_dir, self.latest_dir)

        # Append to history
        record = {"global_step": global_step, "epoch": epoch, "train_loss": round(train_loss, 6),
                  "val_loss": round(val_loss, 6), "lr": round(lr, 10)}
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Top-5 rotation
        self._read_rankings()
        self.rankings.append({"step": global_step, "val_loss": val_loss})
        self.rankings.sort(key=lambda r: r["val_loss"])
        while len(self.rankings) > self.keep_top:
            evicted = self.rankings.pop()
            evict_dir = self._step_dir(evicted["step"])
            if evict_dir.exists() and evict_dir != self.latest_dir:
                shutil.rmtree(evict_dir)
        self._write_rankings()
        return step_dir

    def load_latest(self) -> dict | None:
        """Return training state dict if latest checkpoint exists, else None."""
        state_path = self.latest_dir / "training_state.pt"
        if not state_path.exists():
            return None
        return torch.load(state_path, map_location="cpu", weights_only=False)

    def top5_best(self) -> float:
        self._read_rankings()
        return self.rankings[0]["val_loss"] if self.rankings else float("inf")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    parser = argparse.ArgumentParser(description="StyleDistance training")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-top", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use-local-data", action="store_true")
    parser.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory")
    args = parser.parse_args()

    batch_size = args.batch or config.batch_size
    grad_accum = args.grad or 1
    max_epochs = args.epochs or config.max_epochs
    lr = args.lr or config.lr

    if args.dryrun:
        batch_size = 8; max_epochs = 1; grad_accum = 1
        print("DRY RUN: batch=8 epochs=1 grad_accum=1")

    device = torch.device("cpu") if args.cpu else get_device()
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if use_amp else torch.float32

    # CUDA default: batch=512 (paper used 512 on 4x A6000)
    if args.batch is None and device.type == "cuda":
        batch_size = 512

    print(f"device={device}  batch={batch_size}  grad_accum={grad_accum}  "
          f"effective={batch_size * grad_accum}  dtype={dtype}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    if args.use_local_data:
        import pandas as pd
        local = Path(__file__).resolve().parent.parent / "datasets" / "msynthstel" / "data"
        train_pairs = pd.read_parquet(local / "train-00000-of-00001.parquet").to_dict("records")
    else:
        from datasets import load_dataset
        train_pairs = [dict(row) for row in load_dataset(config.dataset_name)["train"]]

    by_feat = defaultdict(list)
    for row in train_pairs:
        by_feat[row["feature"]].append(row)
    train_split, val_split = [], []
    for feat, group in by_feat.items():
        n_train = int(len(group) * config.train_val_split)
        train_split.extend(group[:n_train])
        val_split.extend(group[n_train:])
    print(f"Train pairs: {len(train_split)}  Val pairs: {len(val_split)}")

    # Load test data for TensorBoard embedding visualization
    if not args.dryrun:
        if args.use_local_data:
            test_pairs = pd.read_parquet(local / "test-00000-of-00001.parquet").to_dict("records")
        else:
            from datasets import load_dataset
            test_pairs = [dict(row) for row in load_dataset(config.dataset_name)["test"]]
        test_sentences = [r["positive"] for r in test_pairs]
        test_labels = [r["feature"] for r in test_pairs]
        print(f"Test sentences for embedding viz: {len(test_sentences)}")
    else:
        test_pairs = []
        test_sentences, test_labels = [], []

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    train_ds = TripletDataset(train_split, tokenizer, config.max_seq_len, "train", config.cache_dir)
    val_ds = TripletDataset(val_split, tokenizer, config.max_seq_len, "val", config.cache_dir)
    print(f"Train triplets: {len(train_ds)}  Val triplets: {len(val_ds)}")

    if args.dryrun:
        train_ds = Subset(train_ds, range(batch_size))
        val_ds = Subset(val_ds, range(batch_size))

    collate_fn = partial(collate_triplets, tokenizer=tokenizer, max_len=config.max_seq_len)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=config.num_workers,
                              pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=config.num_workers,
                            pin_memory=pin_memory)

    # ── Model, optimizer, scheduler ────────────────────────────────────────
    print("Building model...")
    model = StyleDistance(config.model_name).to(device)
    model.encoder.print_trainable_parameters()

    criterion = nn.TripletMarginLoss(margin=config.triplet_margin, p=2)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=config.weight_decay, fused=(device.type == "cuda"),
    )
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * max_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"steps/epoch={steps_per_epoch}  total={total_steps}  save_every={args.save_every}")

    # ── TensorBoard ────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir) if args.log_dir else (
        Path("/root/tf-logs") if device.type == "cuda" else config.output_dir / "tf-logs")
    log_dir = log_dir / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard: {log_dir}")

    # ── Checkpoint manager ─────────────────────────────────────────────────
    ckpt = CheckpointManager(config.output_dir, save_every=args.save_every, keep_top=args.keep_top)
    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume:
        state = ckpt.load_latest()
        if state is None:
            print("No checkpoint found to resume from. Starting fresh.")
        else:
            from peft import PeftModel
            from transformers import AutoModel
            base = AutoModel.from_pretrained(config.model_name)
            model.encoder = PeftModel.from_pretrained(base, str(ckpt.latest_dir))
            model.encoder.enable_input_require_grads()
            model.to(device)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scaler and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"]  # restart this epoch (reshuffled, at most save_every steps lost)
            torch.set_rng_state(state["rng_state"])
            best_val_loss = ckpt.top5_best()
            print(f"Resumed step={global_step} epoch={start_epoch} best_val_loss={best_val_loss:.6f}")

    # Graceful shutdown: save latest on SIGINT
    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, on_interrupt)

    # ── Training ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                     desc=f"Epoch {epoch}/{max_epochs}", unit="step",
                     dynamic_ncols=True, disable=args.dryrun)
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in pbar:
            with torch.autocast(device.type, dtype=dtype, enabled=use_amp):
                a_emb, p_emb, n_emb = model(
                    batch["a_ids"].to(device), batch["a_mask"].to(device),
                    batch["p_ids"].to(device), batch["p_mask"].to(device),
                    batch["n_ids"].to(device), batch["n_mask"].to(device),
                )
                loss = criterion(a_emb, p_emb, n_emb) / grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # TensorBoard: per-step scalars
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("loss/train", loss.item() * grad_accum, global_step)

                # Checkpoint every save_every optimizer steps
                if global_step % args.save_every == 0:
                    val_loss = evaluate_val_loss(model, val_loader, criterion, device)
                    avg_train = epoch_loss / (step + 1) if (step + 1) > 0 else 0.0
                    saved = ckpt.save(model, optimizer, scheduler, scaler,
                                      global_step, epoch, avg_train, val_loss,
                                      scheduler.get_last_lr()[0], device.type, use_amp)
                    was_best = val_loss < best_val_loss
                    if was_best:
                        best_val_loss = val_loss

                    # TensorBoard: checkpoint-level logging
                    writer.add_scalar("loss/val", val_loss, global_step)
                    log_lora_histograms(writer, model, global_step)
                    if not args.dryrun:
                        emb_mat, emb_labels, emb_texts = build_test_embedding_data(
                            model, test_sentences, test_labels, tokenizer, device)
                        writer.add_embedding(emb_mat, metadata=emb_labels,
                                             tag="style/test", global_step=global_step)

            # Live postfix
            cur_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{epoch_loss / max(step + 1, 1):.4f}",
                              "best_val": f"{best_val_loss:.4f}",
                              "lr": f"{cur_lr:.2e}",
                              "top5": f"{len(ckpt.rankings)}"})

            if interrupted:
                print("\nInterrupted — saving latest checkpoint...")
                # Quick val for the interrupted step
                try:
                    vl = evaluate_val_loss(model, val_loader, criterion, device)
                    ckpt.save(model, optimizer, scheduler, scaler, global_step, epoch,
                              epoch_loss / max(step + 1, 1), vl, scheduler.get_last_lr()[0],
                              device.type, use_amp)
                except Exception:
                    pass
                print(f"Saved at global_step={global_step}. Resume with --resume.")
                sys.exit(0)

        # End-of-epoch validation (if not already checkpointed this step)
        if global_step % args.save_every != 0:
            val_loss = evaluate_val_loss(model, val_loader, criterion, device)
            avg_train = epoch_loss / len(train_loader)
            ckpt.save(model, optimizer, scheduler, scaler, global_step, epoch,
                      avg_train, val_loss, scheduler.get_last_lr()[0], device.type, use_amp)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        avg_train_loss = epoch_loss / len(train_loader)
        pbar.set_postfix({"loss": f"{avg_train_loss:.4f}", "best_val": f"{best_val_loss:.4f}",
                          "top5": f"{len(ckpt.rankings)}"})

        # TensorBoard: config text once after epoch 1
        if epoch == 1:
            cfg_text = json.dumps({
                "model": config.model_name,
                "lora_r": config.lora_r, "lora_alpha": config.lora_alpha,
                "batch_size": batch_size, "grad_accum": grad_accum,
                "lr": lr, "weight_decay": config.weight_decay,
                "margin": config.triplet_margin, "max_epochs": max_epochs,
                "train_pairs": len(train_split), "val_pairs": len(val_split),
                "test_pairs": len(test_pairs),
                "train_triplets": len(train_ds), "val_triplets": len(val_ds),
                "device": str(device), "dtype": str(dtype),
            }, indent=2, ensure_ascii=False)
            writer.add_text("config", cfg_text, global_step)

    # ── Training end ───────────────────────────────────────────────────────
    writer.add_hparams(
        {"lr": lr, "batch_size": batch_size, "lora_r": config.lora_r,
         "lora_alpha": config.lora_alpha, "weight_decay": config.weight_decay,
         "margin": config.triplet_margin},
        {"hparam/best_val_loss": best_val_loss},
    )
    writer.close()
    print(f"Training complete.  Best val_loss={best_val_loss:.6f}  "
          f"Checkpoints in {config.output_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"Top 5: {json.dumps(ckpt.rankings, indent=2)}")


if __name__ == "__main__":
    main()
