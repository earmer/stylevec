"""StyleDistance training — exact paper replication.

Usage:
    uv run python train.py --use-local-data --batch 64 --epochs 5
    uv run python train.py --resume                     # resume from checkpoints/latest/
    uv run python train.py --dryrun --use-local-data    # single batch test
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
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

from config import Config, load_multilingual_pairs
from model import StyleDistance
from triplets import MultilingualTripletDataset, collate_triplets

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


def latest_tensorboard_step(log_dir: Path) -> int | None:
    """Return the latest scalar step in a TensorBoard log directory."""
    if not log_dir.exists():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        max_step = None
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                max_step = event.step if max_step is None else max(max_step, event.step)
        return max_step
    except Exception as exc:
        print(f"TensorBoard step detection skipped: {exc}")
        return None


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
             train_loss, val_loss, lr, device_type, use_amp, total_steps=None):
        """Save checkpoint, manage top-5 rotation."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        step_dir = self._step_dir(global_step)
        step_dir.mkdir(parents=True, exist_ok=True)

        # PEFT adapter weights; resume opts back into trainability on load.
        model.train()
        model.encoder.save_pretrained(str(step_dir))

        # Training state
        training_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "global_step": global_step,
            "epoch": epoch,
            "rng_state": torch.get_rng_state(),
        }
        if total_steps is not None:
            training_state["total_steps"] = total_steps
        torch.save(training_state, step_dir / "training_state.pt")

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
        return self.load_from_dir(self.latest_dir)

    def resolve_dir(self, path_or_name: str) -> Path:
        """Resolve a checkpoint path or a bare checkpoint directory name."""
        path = Path(path_or_name).expanduser()
        if path.is_absolute():
            return path
        under_base = self.base_dir / path
        if under_base.exists():
            return under_base
        if path.exists():
            return path
        if re.fullmatch(r"\d{8}-\d{6}", path_or_name):
            dt = _dt.datetime.strptime(path_or_name, "%Y%m%d-%H%M%S")
            prefix = str(int(dt.timestamp()))
            matches = sorted(
                self.base_dir.glob(f"{prefix}-step-*"),
                key=lambda p: int(p.name.rsplit("-step-", 1)[1]) if "-step-" in p.name else -1,
            )
            if matches:
                return matches[-1]
        return under_base

    def load_from_dir(self, ckpt_dir: Path) -> dict | None:
        """Return training state dict if the checkpoint dir is loadable."""
        state_path = Path(ckpt_dir) / "training_state.pt"
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
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--max-pairs-per-feature", type=int, default=0)
    parser.add_argument("--keep-top", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint directory path or name under checkpoints/")
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

    # CUDA default: batch=256
    if args.batch is None and device.type == "cuda":
        batch_size = 256

    print(f"device={device}  batch={batch_size}  grad_accum={grad_accum}  "
          f"effective={batch_size * grad_accum}  dtype={dtype}")

    # ── Data ──────────────────────────────────────────────────────────────
    import random as _random
    print("Loading dataset...")
    if args.use_local_data:
        all_pairs = load_multilingual_pairs(config, "train")
    else:
        from datasets import load_dataset
        all_pairs = [dict(row) for row in load_dataset(config.dataset_name)["train"]]

    by_feat = defaultdict(list)
    for row in all_pairs:
        by_feat[row["feature"]].append(row)
    _random.seed(42)
    train_split, val_split = [], []
    for feat, group in by_feat.items():
        shuffled = list(group)
        _random.shuffle(shuffled)
        n_train = int(len(shuffled) * config.train_val_split)
        train_split.extend(shuffled[:n_train])
        val_split.extend(shuffled[n_train:])
    print(f"Train pairs: {len(train_split)}  Val pairs: {len(val_split)}")

    # Load test data for TensorBoard embedding visualization
    if not args.dryrun:
        if args.use_local_data:
            test_pairs = load_multilingual_pairs(config, "test")
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
    mpp = args.max_pairs_per_feature or config.max_pairs_per_feature
    train_ds = MultilingualTripletDataset(train_split, tokenizer, config.max_seq_len, "train", config.cache_dir, mpp)
    val_ds = MultilingualTripletDataset(val_split, tokenizer, config.max_seq_len, "val", config.cache_dir, mpp)
    print(f"Train triplets: {len(train_ds)}  Val triplets: {len(val_ds)}")

    collate_fn = partial(collate_triplets, tokenizer=tokenizer, max_len=config.max_seq_len)
    pin_memory = device.type == "cuda"

    def _make_train_loader():
        """Re-create train DataLoader with current dataset (call after resample)."""
        ds = train_ds
        if args.dryrun:
            ds = Subset(train_ds, range(batch_size))
        return DataLoader(ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn, num_workers=config.num_workers,
                          pin_memory=pin_memory, drop_last=True)

    if args.dryrun:
        val_ds = Subset(val_ds, range(batch_size))

    train_loader = _make_train_loader()
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=config.num_workers,
                            pin_memory=pin_memory)

    # ── Model, optimizer, scheduler ────────────────────────────────────────
    print("Building model...")
    model = StyleDistance(config.model_name, lora_dropout=config.lora_dropout).to(device)
    model.encoder.print_trainable_parameters()

    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: (x - y).pow(2).sum(dim=1),
        margin=config.triplet_margin,
    )
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
    log_root = Path(args.log_dir) if args.log_dir else (
        Path("/root/tf-logs") if device.type == "cuda" else config.output_dir / "tf-logs")
    if args.resume_from and re.fullmatch(r"\d{8}-\d{6}", args.resume_from):
        log_dir = log_root / args.resume_from
    else:
        log_dir = log_root / time.strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    print(f"TensorBoard: {log_dir}")

    # ── Checkpoint manager ─────────────────────────────────────────────────
    ckpt = CheckpointManager(config.output_dir, save_every=args.save_every, keep_top=args.keep_top)
    start_epoch = 1
    global_step = 0
    log_step_offset = 0
    best_val_loss = float("inf")
    # best_epoch_val_loss = float("inf")
    # epochs_no_improve = 0

    # Resume
    if args.resume or args.resume_from:
        resume_dir = ckpt.resolve_dir(args.resume_from) if args.resume_from else ckpt.latest_dir
        state = ckpt.load_from_dir(resume_dir)
        if state is None:
            if args.resume_from:
                raise FileNotFoundError(f"No training_state.pt found in resume checkpoint: {resume_dir}")
            print("No checkpoint found to resume from. Starting fresh.")
        else:
            from peft import PeftModel
            from transformers import AutoModel
            base = AutoModel.from_pretrained(config.model_name)
            model.encoder = PeftModel.from_pretrained(base, str(resume_dir), is_trainable=True)
            model.encoder.enable_input_require_grads()
            model.to(device)
            # Refresh optimizer against the loaded encoder params
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if not trainable_params:
                lora_count = sum(1 for n, _ in model.encoder.named_parameters() if "lora" in n.lower())
                raise RuntimeError(
                    "Resume checkpoint loaded no trainable parameters "
                    f"({lora_count} LoRA tensors found)."
                )
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr, weight_decay=config.weight_decay, fused=(device.type == "cuda"),
            )
            optimizer.load_state_dict(state["optimizer"])
            if scaler and state.get("scaler"):
                scaler.load_state_dict(state["scaler"])
            global_step = state["global_step"]
            start_epoch = state["epoch"]  # restart this epoch (reshuffled, at most save_every steps lost)
            torch.set_rng_state(state["rng_state"])
            # Rebuild scheduler, then restore the exact saved scheduler state.
            saved_total = state.get("total_steps", state.get("scheduler", {}).get("total_iters", total_steps))
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=saved_total,
            )
            scheduler.load_state_dict(state["scheduler"])
            best_val_loss = ckpt.top5_best()
            # best_epoch_val_loss = best_val_loss
            tb_step = latest_tensorboard_step(log_dir)
            if tb_step is not None and tb_step > global_step:
                log_step_offset = tb_step - global_step
                print("TensorBoard log is ahead of checkpoint: "
                      f"checkpoint_step={global_step} tb_step={tb_step}; "
                      f"next_log_step={global_step + log_step_offset + 1}")
            print(f"Resumed from {resume_dir} step={global_step} epoch={start_epoch} "
                  f"best_val_loss={best_val_loss:.6f}")

    # Graceful shutdown: save latest on SIGINT
    interrupted = False

    def on_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, on_interrupt)

    # ── Training ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        train_ds.resample()
        train_loader = _make_train_loader()
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
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                log_step = global_step + log_step_offset

                # TensorBoard: per-step scalars
                writer.add_scalar("lr", scheduler.get_last_lr()[0], log_step)
                writer.add_scalar("loss/train", loss.item() * grad_accum, log_step)

                # Validate every val_every optimizer steps
                if global_step % args.val_every == 0:
                    val_loss = evaluate_val_loss(model, val_loader, criterion, device)
                    writer.add_scalar("loss/val", val_loss, log_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                # Checkpoint every save_every optimizer steps
                if global_step % args.save_every == 0:
                    val_loss = evaluate_val_loss(model, val_loader, criterion, device)
                    avg_train = epoch_loss / (step + 1) if (step + 1) > 0 else 0.0
                    saved = ckpt.save(model, optimizer, scheduler, scaler,
                                      global_step, epoch, avg_train, val_loss,
                                      scheduler.get_last_lr()[0], device.type, use_amp,
                                      total_steps)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                    # TensorBoard: checkpoint-level logging
                    writer.add_scalar("loss/val", val_loss, log_step)
                    log_lora_histograms(writer, model, log_step)
                    if not args.dryrun:
                        emb_mat, emb_labels, emb_texts = build_test_embedding_data(
                            model, test_sentences, test_labels, tokenizer, device)
                        writer.add_embedding(emb_mat, metadata=emb_labels,
                                             tag="style/test", global_step=log_step)

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
                              device.type, use_amp, total_steps)
                except Exception:
                    pass
                print(f"Saved at global_step={global_step}. Resume with --resume.")
                sys.exit(0)

        # End-of-epoch: ensure we have a fresh val_loss for early stopping.
        # If the last optimizer step landed on a checkpoint boundary we already
        # evaluated; otherwise run an evaluation now.
        if global_step % args.save_every != 0:
            val_loss = evaluate_val_loss(model, val_loader, criterion, device)
            avg_train = epoch_loss / len(train_loader)
            ckpt.save(model, optimizer, scheduler, scaler, global_step, epoch,
                      avg_train, val_loss, scheduler.get_last_lr()[0], device.type, use_amp,
                      total_steps)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # # Early stopping: compare epoch-level val_loss (not mid-epoch snapshots)
        # if val_loss < best_epoch_val_loss - config.early_stopping_threshold:
        #     best_epoch_val_loss = val_loss
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     print(f"  val_loss={val_loss:.6f} no better than best_epoch={best_epoch_val_loss:.6f} "
        #           f"(no-improve={epochs_no_improve}/{config.early_stopping_patience})")
        # if epochs_no_improve >= config.early_stopping_patience:
        #     print(f"Early stopping at epoch {epoch} (no improvement for "
        #           f"{epochs_no_improve} epoch(s), best_epoch_val_loss={best_epoch_val_loss:.6f})")
        #     break

        avg_train_loss = epoch_loss / len(train_loader)
        pbar.set_postfix({"loss": f"{avg_train_loss:.4f}", "best_val": f"{best_val_loss:.4f}",
                          "top5": f"{len(ckpt.rankings)}"})

        # TensorBoard: config text once after epoch 1
        if epoch == 1:
            cfg_text = json.dumps({
                "model": config.model_name,
                "languages": config.language_list,
                "lora_r": config.lora_r, "lora_alpha": config.lora_alpha,
                "batch_size": batch_size, "grad_accum": grad_accum,
                "lr": lr, "weight_decay": config.weight_decay,
                "margin": config.triplet_margin, "max_epochs": max_epochs,
                "train_pairs": len(train_split), "val_pairs": len(val_split),
                "test_pairs": len(test_pairs),
                "train_triplets": len(train_ds), "val_triplets": len(val_ds),
                "device": str(device), "dtype": str(dtype),
            }, indent=2, ensure_ascii=False)
            writer.add_text("config", cfg_text, global_step + log_step_offset)

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
