from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from build_triplets import build_training_triplets
from config import config_hash, config_snapshot, load_config, workspace_dir
from dataset import TripletTextDataset, make_triplet_collate
from losses import embedding_spread_loss, weighted_triplet_margin_loss
from model import (
    build_qwen_style_encoder,
    encode_triplet,
    load_trainable_state_dict,
    parameter_summary,
    trainable_state_dict,
)
from validate import evaluate_triplet_loader, move_batch_to_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requested CUDA, but CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("config requested MPS, but MPS is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported runtime.device: {requested}")
    return torch.device(requested)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16" and device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    raise ValueError(f"precision {precision} is not supported on {device.type}")


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_ratio: float, total_steps: int):
    warmup_steps = max(1, int(total_steps * warmup_ratio)) if warmup_ratio > 0 else 0

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def tensorboard_dir(config: dict[str, Any], device: torch.device, run_name: str) -> Path:
    configured = config.get("logging", {}).get("tensorboard_dir")
    if configured:
        base = Path(configured)
    elif device.type == "cuda":
        base = Path("/root/tf-logs")
    else:
        base = workspace_dir() / "tf-logs"
    return base / run_name


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    out_dir: Path,
    global_step: int,
    epoch: int,
    manifest_hash: str,
    accelerator: Any | None = None,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    adapter_dir = out_dir / "adapter"
    if hasattr(unwrapped.backbone, "save_pretrained"):
        unwrapped.backbone.save_pretrained(adapter_dir)
    else:
        adapter_dir.mkdir()
        (adapter_dir / "NO_PEFT_ADAPTER.txt").write_text("LoRA disabled; trainable state is in training_state.pt\n")
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "manifest_hash": manifest_hash,
        "model_trainable": trainable_state_dict(unwrapped),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(state, out_dir / "training_state.pt")


def load_resume_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resume_dir: Path,
    manifest_hash: str,
    allow_manifest_mismatch: bool = False,
) -> tuple[int, int]:
    state_path = resume_dir / "training_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"resume checkpoint missing training_state.pt: {resume_dir}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("manifest_hash") != manifest_hash and not allow_manifest_mismatch:
        raise ValueError(
            "resume manifest hash mismatch; start a new run or rebuild with the same data/config"
        )
    if state.get("manifest_hash") != manifest_hash:
        print("warning: resume manifest hash differs; continuing because config allows it", flush=True)
    load_trainable_state_dict(model, state["model_trainable"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.random.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    return int(state["global_step"]), int(state["epoch"])


def balanced_triplet_order(triplets: list[Any], seed: int) -> list[Any]:
    buckets: dict[str, list[Any]] = {}
    generator = random.Random(seed)
    for row in triplets:
        buckets.setdefault(row.rule, []).append(row)
    for rows in buckets.values():
        generator.shuffle(rows)
    ordered: list[Any] = []
    rules = sorted(buckets)
    while rules:
        next_rules: list[str] = []
        for rule in rules:
            rows = buckets[rule]
            if rows:
                ordered.append(rows.pop())
            if rows:
                next_rules.append(rule)
        rules = next_rules
    return ordered


def build_loaders(config: dict[str, Any], tokenizer: Any, triplets: list[Any]) -> tuple[DataLoader, DataLoader]:
    collate = make_triplet_collate(tokenizer, int(config["model"]["max_length"]))
    train_triplets = [row for row in triplets if row.split == "train"]
    if bool(config["train"].get("rule_balanced_batches", False)):
        train_triplets = balanced_triplet_order(train_triplets, int(config["runtime"]["seed"]))
    train_dataset = TripletTextDataset(train_triplets, "train")
    val_dataset = TripletTextDataset(triplets, "val")
    generator = torch.Generator()
    generator.manual_seed(int(config["runtime"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=not bool(config["train"].get("rule_balanced_batches", False)),
        generator=generator,
        num_workers=int(config["train"].get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=bool(config["train"].get("pin_memory", False)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["validation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    return train_loader, val_loader


def train_from_config(config_path: str, run_name: str, resume: str | None = None) -> Path:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    seed_everything(int(config["runtime"]["seed"]))
    run_dir = workspace_dir() / "outputs" / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    triplets, manifest = build_training_triplets(config)
    manifest_hash = manifest["manifest_hash"]
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    train_loader, val_loader = build_loaders(config, tokenizer, triplets)
    eval_val_loader = val_loader

    requested_device = str(config["runtime"]["device"])
    use_accelerate = str(config["runtime"].get("distributed", "none")) == "accelerate" or int(os.environ.get("WORLD_SIZE", "1")) > 1
    accelerator = None
    if use_accelerate:
        from accelerate import Accelerator

        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config requested CUDA distributed training, but CUDA is unavailable")
        mixed_precision = str(config["runtime"]["precision"])
        accelerator = Accelerator(
            mixed_precision=mixed_precision if mixed_precision in {"bf16", "fp16"} else "no",
            gradient_accumulation_steps=int(config["train"].get("grad_accum", 1)),
        )
        device = accelerator.device
    else:
        device = select_device(requested_device)

    model = build_qwen_style_encoder(config)
    if not use_accelerate:
        model = model.to(device)
    summary = parameter_summary(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    total_steps = int(config["train"]["max_steps"])
    scheduler = make_scheduler(optimizer, float(config["train"].get("warmup_ratio", 0.0)), total_steps)

    global_step = 0
    start_epoch = 0
    if resume:
        global_step, start_epoch = load_resume_state(
            model,
            optimizer,
            scheduler,
            Path(resume),
            manifest_hash,
            bool(config["train"].get("allow_manifest_mismatch_on_resume", False)),
        )

    if use_accelerate:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )
        is_main = accelerator.is_main_process
    else:
        is_main = True

    writer = SummaryWriter(str(tensorboard_dir(config, device, run_name))) if is_main else None
    if writer is not None:
        writer.add_text("config/snapshot", config_snapshot(config), global_step)
        writer.add_scalar("params/total", summary["total"], global_step)
        writer.add_scalar("params/trainable", summary["trainable"], global_step)
        writer.add_scalar("params/trainable_ratio", summary["ratio"], global_step)
        for key, value in manifest["counts"].items():
            writer.add_scalar(f"data/{key}", float(value), global_step)

    precision = str(config["runtime"]["precision"])
    grad_accum = int(config["train"].get("grad_accum", 1))
    max_grad_norm = float(config["train"].get("max_grad_norm", 1.0))
    max_seconds = float(config["train"].get("max_seconds", 0.0))
    regularization = config.get("regularization", {})
    spread_variance_weight = float(regularization.get("variance_weight", 0.0))
    spread_covariance_weight = float(regularization.get("covariance_weight", 0.0))
    spread_min_dim_std = float(regularization.get("min_dim_std", 0.0))
    stop_time = time.monotonic() + max_seconds if max_seconds > 0.0 else 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    last_epoch = start_epoch
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        last_epoch = epoch
        for batch_index, batch in enumerate(train_loader):
            if should_stop(global_step, total_steps, stop_time):
                break
            batch = move_batch_to_device(batch, device)
            if use_accelerate:
                with accelerator.accumulate(model):
                    anchor, positive, negative, norms = encode_triplet(model, batch)
                    loss, stats = weighted_triplet_margin_loss(
                        anchor,
                        positive,
                        negative,
                        batch["margin"],
                        batch["weight"],
                    )
                    embeddings = torch.cat([anchor, positive, negative], dim=0)
                    spread_loss, spread_stats = embedding_spread_loss(
                        embeddings,
                        spread_min_dim_std,
                        spread_variance_weight,
                        spread_covariance_weight,
                    )
                    loss = loss + spread_loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if is_main and writer is not None:
                            log_train_step(writer, loss, stats, spread_stats, norms, optimizer, grad_norm, global_step)
            else:
                with autocast_context(device, precision):
                    anchor, positive, negative, norms = encode_triplet(model, batch)
                    loss, stats = weighted_triplet_margin_loss(
                        anchor,
                        positive,
                        negative,
                        batch["margin"],
                        batch["weight"],
                    )
                    embeddings = torch.cat([anchor, positive, negative], dim=0)
                    spread_loss, spread_stats = embedding_spread_loss(
                        embeddings,
                        spread_min_dim_std,
                        spread_variance_weight,
                        spread_covariance_weight,
                    )
                    loss = loss + spread_loss
                    scaled_loss = loss / grad_accum
                scaled_loss.backward()
                if (batch_index + 1) % grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if writer is not None:
                        log_train_step(writer, loss, stats, spread_stats, norms, optimizer, grad_norm, global_step)

            if should_stop(global_step, total_steps, stop_time):
                break

        if should_stop(global_step, total_steps, stop_time):
            break

    if accelerator is not None:
        accelerator.wait_for_everyone()

    if is_main:
        eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
        val_metrics = evaluate_triplet_loader(
            eval_model,
            eval_val_loader,
            device,
            max_batches=int(config["validation"].get("max_batches", 4)),
        )
        if writer is not None:
            for key, value in val_metrics.items():
                writer.add_scalar(f"validation/{key}", value, global_step)
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_dir / "latest",
            global_step,
            last_epoch,
            manifest_hash,
            accelerator,
        )
        if writer is not None:
            writer.flush()
            writer.close()
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.end_training()
    return run_dir


def should_stop(global_step: int, total_steps: int, stop_time: float) -> bool:
    if global_step >= total_steps:
        return True
    return stop_time > 0.0 and time.monotonic() >= stop_time


def log_train_step(
    writer: SummaryWriter,
    loss: torch.Tensor,
    stats: dict[str, torch.Tensor],
    spread_stats: dict[str, torch.Tensor],
    norms: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_norm: torch.Tensor | float,
    global_step: int,
) -> None:
    writer.add_scalar("loss/train", float(loss.detach().float().cpu().item()), global_step)
    writer.add_scalar("cosine/positive_mean", float(stats["pos_cos"].mean().cpu().item()), global_step)
    writer.add_scalar("cosine/negative_mean", float(stats["neg_cos"].mean().cpu().item()), global_step)
    writer.add_scalar("triplet/violation_ratio", float(stats["violations"].mean().cpu().item()), global_step)
    writer.add_scalar("spread/dim_std_mean", float(spread_stats["dim_std_mean"].float().cpu().item()), global_step)
    writer.add_scalar("spread/variance_loss", float(spread_stats["variance_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/covariance_loss", float(spread_stats["covariance_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/top_eigen_ratio", float(spread_stats["top_eigen_ratio"].float().cpu().item()), global_step)
    writer.add_scalar("spread/effective_rank", float(spread_stats["effective_rank"].float().cpu().item()), global_step)
    writer.add_scalar("embedding/pre_norm_mean", float(norms.detach().float().mean().cpu().item()), global_step)
    writer.add_scalar("embedding/pre_norm_std", float(norms.detach().float().std(unbiased=False).cpu().item()), global_step)
    writer.add_scalar("optim/lr", float(optimizer.param_groups[0]["lr"]), global_step)
    grad_value = grad_norm.detach().float().cpu().item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    writer.add_scalar("optim/grad_norm", grad_value, global_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    out_dir = train_from_config(args.config, args.run_name, args.resume)
    print(f"run outputs: {out_dir}")


if __name__ == "__main__":
    main()
