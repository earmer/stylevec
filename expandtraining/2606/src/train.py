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

from prebuilt import load_or_build
import tracking
from config import (
    config_snapshot,
    configured_matryoshka_dims,
    load_config,
    validation_matryoshka_dims,
    workspace_dir,
)
from dataset import TripletTextDataset, make_triplet_collate
from curriculum import curriculum_order, schedule_summary
from losses import (
    content_pair_loss,
    cross_correlation_penalty,
    embedding_spread_loss,
    info_nce_loss,
    matryoshka_prefix_loss,
    uniformity_loss,
    weighted_triplet_margin_loss,
)
from model import (
    build_qwen_style_encoder,
    encode_triplet_outputs,
    load_trainable_state_dict,
    parameter_summary,
    trainable_state_dict,
)
from validate import evaluate_triplet_loader, move_batch_to_device, select_validation_triplets


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
    trainable = trainable_state_dict(unwrapped)
    save_weights(trainable, out_dir, global_step, manifest_hash)
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "manifest_hash": manifest_hash,
        "model_trainable": trainable,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(state, out_dir / "training_state.pt")


def save_weights(trainable: dict[str, torch.Tensor], out_dir: Path, global_step: int, manifest_hash: str) -> None:
    """Every trained parameter (LoRA, projection head, fusion) without optimizer state: what validation
    needs, and what survives `run_queue --prune`. adapter/ alone lacks the projection head."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"global_step": global_step, "manifest_hash": manifest_hash, "model_trainable": trainable}
    torch.save(payload, out_dir / "weights.pt.partial")
    (out_dir / "weights.pt.partial").replace(out_dir / "weights.pt")


def snapshot_steps(config: dict[str, Any], total_steps: int) -> set[int]:
    """Steps (from `train.keep_weights_at`, fractions of max_steps) whose weights are kept in
    checkpoints/step_NNNNN/, so intermediate points of a run can be validated later."""
    fractions = config["train"].get("keep_weights_at", []) or []
    steps = {int(round(float(f) * total_steps)) for f in fractions}
    return {step for step in steps if 0 < step < total_steps}


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


def balanced_triplet_order(
    triplets: list[Any],
    seed: int,
    epoch: int = 0,
) -> list[Any]:
    buckets: dict[str, list[Any]] = {}
    generator = random.Random(seed + epoch)
    for row in triplets:
        buckets.setdefault(row.rule, []).append(row)
    for rows in buckets.values():
        generator.shuffle(rows)
    ordered: list[Any] = []
    rules = _rotated(sorted(buckets), epoch)
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


def _rotated(values: list[str], offset: int) -> list[str]:
    if not values:
        return []
    offset = offset % len(values)
    return values[offset:] + values[:offset]


def curriculum_row_budget(config: dict[str, Any]) -> int:
    train = config["train"]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return int(train["max_steps"]) * int(train["batch_size"]) * int(train.get("grad_accum", 1)) * world


def curriculum_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("curriculum", {}).get("enabled", False))


def train_loader_shuffles(config: dict[str, Any]) -> bool:
    # Curriculum and rule-balanced orders are positional; a shuffling loader would erase them.
    return not (curriculum_enabled(config) or bool(config["train"].get("rule_balanced_batches", False)))


def order_train_triplets(
    config: dict[str, Any],
    train_triplets: list[Any],
    epoch: int,
    start_step: int = 0,
) -> list[Any]:
    if curriculum_enabled(config):
        # One pass covers all max_steps; stage follows position, so `epochs` is irrelevant here.
        ordered = curriculum_order(
            train_triplets, config, int(config["runtime"]["seed"]), curriculum_row_budget(config)
        )
        per_step = int(config["train"]["batch_size"]) * int(config["train"].get("grad_accum", 1)) * int(
            os.environ.get("WORLD_SIZE", "1")
        )
        return ordered[start_step * per_step :] or ordered[-per_step:]
    if not bool(config["train"].get("rule_balanced_batches", False)):
        rows = list(train_triplets)
        generator = random.Random(int(config["runtime"]["seed"]) + epoch)
        generator.shuffle(rows)
        return rows
    return balanced_triplet_order(
        train_triplets,
        int(config["runtime"]["seed"]),
        epoch=epoch if bool(config["train"].get("rule_balance_rotate_each_epoch", True)) else 0,
    )


def build_loaders(
    config: dict[str, Any],
    tokenizer: Any,
    triplets: list[Any],
) -> tuple[TripletTextDataset, DataLoader, DataLoader]:
    collate = make_triplet_collate(tokenizer, int(config["model"]["max_length"]))
    train_triplets = [row for row in triplets if row.split == "train"]
    train_dataset = TripletTextDataset(order_train_triplets(config, train_triplets, epoch=0), "train")
    val_dataset = TripletTextDataset(
        select_validation_triplets(triplets, int(config["validation"].get("per_rule", 64))), "val"
    )
    generator = torch.Generator()
    generator.manual_seed(int(config["runtime"]["seed"]))
    train_loader_kwargs = dataloader_worker_kwargs(config["train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=train_loader_shuffles(config),
        generator=generator,
        collate_fn=collate,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["validation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    return train_dataset, train_loader, val_loader


def dataloader_worker_kwargs(section: dict[str, Any]) -> dict[str, Any]:
    worker_count = int(section.get("num_workers", 0))
    kwargs: dict[str, Any] = {
        "num_workers": worker_count,
        "pin_memory": bool(section.get("pin_memory", False)),
    }
    if worker_count > 0:
        kwargs["persistent_workers"] = bool(section.get("persistent_workers", False)) and not bool(
            section.get("rule_balance_rotate_each_epoch", False)
        )
        kwargs["prefetch_factor"] = int(section.get("prefetch_factor", 2))
    return kwargs


HEAD_MODULES = ("projection.", "fusion.", "content_head.")


def optimizer_parameter_groups(model: torch.nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    """One group at train.lr, or two when train.head_lr is set: the randomly initialised projection head
    (and fusion weights) at head_lr, the LoRA adapters at lr. The cosine schedule scales both alike."""
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("model has no trainable parameters")
    head_lr = config["train"].get("head_lr")
    if head_lr is None:
        return [{"params": [parameter for _, parameter in named]}]
    head = [parameter for name, parameter in named if name.startswith(HEAD_MODULES)]
    body = [parameter for name, parameter in named if not name.startswith(HEAD_MODULES)]
    if not head:
        raise ValueError("train.head_lr is set but the model has no projection/fusion parameters")
    groups = [{"params": head, "lr": float(head_lr), "name": "head"}]
    if body:
        groups.insert(0, {"params": body, "name": "body"})
    return groups


def build_optimizer(
    parameters: list[torch.nn.Parameter] | list[dict[str, Any]],
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {
        "lr": float(config["train"]["lr"]),
        "weight_decay": float(config["train"]["weight_decay"]),
    }
    if bool(config["train"].get("fused_adamw", False)) and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def train_from_config(config_path: str, run_name: str, resume: str | None = None, dataset: str | None = None) -> Path:
    from transformers import AutoTokenizer

    config = load_config(config_path)
    seed_everything(int(config["runtime"]["seed"]))
    run_dir = workspace_dir() / "outputs" / "runs" / run_name
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    triplets, manifest = load_or_build(config, dataset)
    manifest_hash = manifest["manifest_hash"]
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_path"], local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer has neither pad_token nor eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, train_loader, val_loader = build_loaders(config, tokenizer, triplets)
    eval_val_loader = val_loader
    raw_train_triplets = [row for row in triplets if row.split == "train"]
    write_run_records(config, run_dir, train_dataset.triplets, raw_train_triplets)

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
            # The schedule is in optimizer steps and is stepped once per optimizer step below. Linked to the
            # optimizer, accelerate would step it num_processes times each, ending a 2-GPU run's schedule
            # (lr 0) at max_steps / 2.
            step_scheduler_with_optimizer=False,
        )
        device = accelerator.device
    else:
        device = select_device(requested_device)

    model = build_qwen_style_encoder(config)
    if not use_accelerate:
        model = model.to(device)
    summary = parameter_summary(model)
    optimizer = build_optimizer(optimizer_parameter_groups(model, config), config)
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
        eval_val_loader = val_loader
        is_main = accelerator.is_main_process
    else:
        is_main = True

    tracker = tracking.start(config, run_name, run_dir, "train") if is_main else None
    tracking.mirror_tensorboard(tracker)
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
    checkpoint_every_steps = int(config["train"].get("checkpoint_every_steps", 0))
    keep_steps = snapshot_steps(config, total_steps)
    regularization = config.get("regularization", {})
    spread_settings = spread_settings_from_config(config)
    matryoshka_weight = float(regularization.get("matryoshka_weight", 0.0))
    matryoshka_dims = configured_matryoshka_dims(config)
    objective = objective_settings_from_config(config)
    content_settings = content_settings_from_config(config)
    combine_forward = bool(config["train"].get("combine_triplet_forward", False))
    non_blocking = device.type == "cuda" and bool(config["train"].get("pin_memory", False))
    log_every_steps = max(1, int(config.get("logging", {}).get("train_every_steps", 1)))
    total_batches = len(train_loader)
    stop_time = time.monotonic() + max_seconds if max_seconds > 0.0 else 0.0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    last_epoch = start_epoch
    for epoch in range(start_epoch, int(config["train"]["epochs"])):
        last_epoch = epoch
        train_dataset.set_triplets(order_train_triplets(config, raw_train_triplets, epoch, global_step))
        for batch_index, batch in enumerate(train_loader):
            if should_stop(global_step, total_steps, stop_time):
                break
            batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
            if use_accelerate:
                with accelerator.accumulate(model):
                    outputs = encode_triplet_outputs(model, batch, combine_forward=combine_forward)
                    anchor, positive, negative = outputs["embeddings"]
                    norms = outputs["pre_norm"]
                    groups = style_group_ids(batch, anchor.device) if objective["loss"] == "infonce" else None
                    loss, stats = objective_loss(anchor, positive, negative, batch, objective, groups)
                    matryoshka_loss, matryoshka_stats = matryoshka_step_loss(
                        anchor,
                        positive,
                        negative,
                        batch,
                        matryoshka_dims,
                        matryoshka_weight,
                        objective,
                        groups,
                    )
                    spread_loss, spread_stats = spread_step_loss(anchor, positive, negative, spread_settings)
                    content_loss, content_stats = content_step_loss(outputs, batch, content_settings)
                    loss = loss + spread_loss + matryoshka_loss + content_loss
                    if grad_accum > 1:
                        window_size = accumulation_window(batch_index, total_batches, grad_accum)
                        loss = loss * (float(grad_accum) / float(window_size))
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        if is_main and global_step in keep_steps:
                            save_weights(
                                trainable_state_dict(accelerator.unwrap_model(model)),
                                checkpoint_dir / f"step_{global_step:05d}",
                                global_step,
                                manifest_hash,
                            )
                        if is_main and checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
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
                        if is_main and writer is not None and global_step % log_every_steps == 0:
                            log_train_step(
                                writer,
                                loss,
                                stats,
                                spread_stats,
                                matryoshka_stats,
                                content_stats,
                                norms,
                                optimizer,
                                grad_norm,
                                global_step,
                                batch,
                            )
            else:
                with autocast_context(device, precision):
                    outputs = encode_triplet_outputs(model, batch, combine_forward=combine_forward)
                    anchor, positive, negative = outputs["embeddings"]
                    norms = outputs["pre_norm"]
                    groups = style_group_ids(batch, anchor.device) if objective["loss"] == "infonce" else None
                    loss, stats = objective_loss(anchor, positive, negative, batch, objective, groups)
                    matryoshka_loss, matryoshka_stats = matryoshka_step_loss(
                        anchor,
                        positive,
                        negative,
                        batch,
                        matryoshka_dims,
                        matryoshka_weight,
                        objective,
                        groups,
                    )
                    spread_loss, spread_stats = spread_step_loss(anchor, positive, negative, spread_settings)
                    content_loss, content_stats = content_step_loss(outputs, batch, content_settings)
                    loss = loss + spread_loss + matryoshka_loss + content_loss
                    window_size = accumulation_window(batch_index, total_batches, grad_accum)
                    scaled_loss = loss / float(window_size)
                scaled_loss.backward()
                if (batch_index + 1) % grad_accum == 0 or (batch_index + 1) == total_batches:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if global_step in keep_steps:
                        save_weights(
                            trainable_state_dict(model), checkpoint_dir / f"step_{global_step:05d}", global_step, manifest_hash
                        )
                    if is_main and checkpoint_every_steps > 0 and global_step % checkpoint_every_steps == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            checkpoint_dir / "latest",
                            global_step,
                            last_epoch,
                            manifest_hash,
                            None,
                        )
                    if writer is not None and global_step % log_every_steps == 0:
                        log_train_step(
                            writer,
                            loss,
                            stats,
                            spread_stats,
                            matryoshka_stats,
                            content_stats,
                            norms,
                            optimizer,
                            grad_norm,
                            global_step,
                            batch,
                        )

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
            prefix_dims=validation_matryoshka_dims(config),
            combine_forward=combine_forward,
            non_blocking=non_blocking,
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
        tracking.finish(tracker)
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.end_training()
    return run_dir


def write_run_records(
    config: dict[str, Any],
    run_dir: Path,
    ordered: list[Any],
    train_triplets: list[Any],
) -> None:
    """The exact config and, under a curriculum, what the schedule actually sampled."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    (run_dir / "config.yaml").write_text(config_snapshot(config), encoding="utf-8")
    if not curriculum_enabled(config):
        return
    if float(config["train"].get("max_seconds", 0.0)) > 0.0:
        print(
            "warning: curriculum stages follow max_steps; train.max_seconds can stop the run before the late "
            "stages, so a time-capped staged arm is not comparable to its mixed control",
            flush=True,
        )
    available: dict[str, int] = {}
    for row in train_triplets:
        available[row.rule] = available.get(row.rule, 0) + 1
    summary = schedule_summary(ordered, config, available)
    (run_dir / "curriculum_schedule.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def should_stop(global_step: int, total_steps: int, stop_time: float) -> bool:
    if global_step >= total_steps:
        return True
    return stop_time > 0.0 and time.monotonic() >= stop_time


def accumulation_window(batch_index: int, total_batches: int, accumulation: int) -> int:
    if accumulation <= 1:
        return 1
    window_start = (batch_index // accumulation) * accumulation
    return max(1, min(accumulation, total_batches - window_start))


def spread_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    regularization = config.get("regularization", {})
    return {
        "min_dim_std": float(regularization.get("min_dim_std", 0.0)),
        "variance_weight": float(regularization.get("variance_weight", 0.0)),
        "covariance_weight": float(regularization.get("covariance_weight", 0.0)),
        "decorrelation": str(regularization.get("decorrelation", "covariance")),
        "uniformity_weight": float(regularization.get("uniformity_weight", 0.0)),
        "uniformity_t": float(regularization.get("uniformity_t", 2.0)),
    }


def spread_step_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Variance floor and decorrelation on the whole batch, plus uniformity within anchors and positives.

    Not within negatives: 74% of same_speaker_topic_matched negatives are Paimon or Traveler, so
    uniformity there would push one speaker's lines apart.
    """
    total, stats = embedding_spread_loss(
        torch.cat([anchor, positive, negative], dim=0),
        settings["min_dim_std"],
        settings["variance_weight"],
        settings["covariance_weight"],
        settings["decorrelation"],
    )
    uniformity = uniformity_loss([anchor, positive], settings["uniformity_t"])
    stats["uniformity_loss"] = uniformity.detach()
    if settings["uniformity_weight"] > 0.0:
        total = total + uniformity * settings["uniformity_weight"]
    return total, stats


def objective_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """train.loss picks the main objective: `triplet` (the hinge on 1 - cos, default) or `infonce`.
    Under infonce, train.infonce_triplet_weight adds the hinge on top (0 = InfoNCE alone)."""
    train = config["train"]
    return {
        "loss": str(train.get("loss", "triplet")),
        "temperature": float(train.get("infonce_temperature", 0.05)),
        "triplet_weight": float(train.get("infonce_triplet_weight", 0.0)),
    }


def style_group_ids(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Integer ids for the collate's anchor_style/negative_style labels. Each unknown (None) gets its own
    negative id, so it never matches anything and is never masked."""
    ids: dict[str, int] = {}

    def encode(labels: list[str | None], offset: int) -> torch.Tensor:
        values = [-(2 * index + offset) if label is None else ids.setdefault(label, len(ids))
                  for index, label in enumerate(labels)]
        return torch.tensor(values, dtype=torch.long, device=device)

    return encode(batch["anchor_style"], 1), encode(batch["negative_style"], 2)


def objective_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    batch: dict[str, Any],
    settings: dict[str, Any],
    groups: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if settings["loss"] == "triplet":
        return weighted_triplet_margin_loss(anchor, positive, negative, batch["margin"], batch["weight"])
    if settings["loss"] != "infonce":
        raise ValueError(f"unknown train.loss: {settings['loss']!r}")
    anchor_groups, negative_groups = groups if groups is not None else style_group_ids(batch, anchor.device)
    loss, stats = info_nce_loss(
        anchor,
        positive,
        negative,
        batch["margin"],
        batch["weight"],
        anchor_groups,
        negative_groups,
        settings["temperature"],
    )
    stats["infonce_loss"] = loss.detach()
    if settings["triplet_weight"] > 0.0:
        triplet, _ = weighted_triplet_margin_loss(anchor, positive, negative, batch["margin"], batch["weight"])
        stats["triplet_loss"] = triplet.detach()
        loss = loss + triplet * settings["triplet_weight"]
    return loss, stats


def matryoshka_step_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    batch: dict[str, Any],
    prefix_dims: list[int],
    loss_weight: float,
    settings: dict[str, Any],
    groups: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The main objective again on each training prefix (re-normalised), weighted by matryoshka_weight."""
    if loss_weight <= 0.0 or not prefix_dims:
        zero = anchor.new_tensor(0.0)
        return zero, {"loss": zero.detach()}
    loss, stats = matryoshka_prefix_loss(
        anchor,
        positive,
        negative,
        prefix_dims,
        lambda a, p, n: objective_loss(a, p, n, batch, settings, groups),
    )
    weighted = loss * float(loss_weight)
    stats["weighted_loss"] = weighted.detach()
    return weighted, stats


# Rules whose anchor and negative share content: a line and its paraphrase, or a rewrite and its source.
CONTENT_PAIR_RULES = frozenset({"origin_vs_para", "paratarget_original_push", "same_content_diff_style"})


def content_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """model.content_dim > 0 adds the content head. regularization.content_weight trains it on content pairs;
    content_leak_weight pushes the style embedding away from linear correlation with it."""
    regularization = config.get("regularization", {})
    return {
        "enabled": int(config["model"].get("content_dim", 0)) > 0,
        "weight": float(regularization.get("content_weight", 0.5)),
        "temperature": float(regularization.get("content_temperature", 0.1)),
        "leak_weight": float(regularization.get("content_leak_weight", 0.0)),
    }


def content_duplicates(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    """[batch, 3 * batch]: candidate text j (negatives, positives, anchors) equals row i's anchor or negative."""
    candidates = list(batch["negative_text"]) + list(batch["positive_text"]) + list(batch["anchor_text"])
    rows = [
        [text in (anchor_text, negative_text) for text in candidates]
        for anchor_text, negative_text in zip(batch["anchor_text"], batch["negative_text"])
    ]
    return torch.tensor(rows, dtype=torch.bool, device=device)


def content_step_loss(
    outputs: dict[str, Any],
    batch: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Content-head InfoNCE on CONTENT_PAIR_RULES rows, plus the style/content leakage penalty."""
    if not settings["enabled"]:
        return outputs["embeddings"][0].new_tensor(0.0), {}
    anchor, positive, negative = outputs["content"]
    device = anchor.device
    is_pair = torch.tensor([rule in CONTENT_PAIR_RULES for rule in batch["rule"]], dtype=torch.bool, device=device)
    loss, stats = content_pair_loss(
        anchor, positive, negative, is_pair, content_duplicates(batch, device), settings["temperature"]
    )
    style = torch.cat(outputs["embeddings"], dim=0)
    leak = cross_correlation_penalty(style, torch.cat([anchor, positive, negative], dim=0).detach())
    stats["pair_loss"] = loss.detach()
    stats["leak"] = leak.detach()
    stats["leak_noise_floor"] = style.new_tensor(style.shape[1] / style.shape[0]).detach()
    total = loss * settings["weight"]
    if settings["leak_weight"] > 0.0:
        total = total + leak * settings["leak_weight"]
    return total, stats


def log_train_step(
    writer: SummaryWriter,
    loss: torch.Tensor,
    stats: dict[str, torch.Tensor],
    spread_stats: dict[str, torch.Tensor],
    matryoshka_stats: dict[str, torch.Tensor],
    content_stats: dict[str, torch.Tensor],
    norms: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    grad_norm: torch.Tensor | float,
    global_step: int,
    batch: dict[str, Any],
) -> None:
    writer.add_scalar("loss/train", float(loss.detach().float().cpu().item()), global_step)
    writer.add_scalar("cosine/positive_mean", float(stats["pos_cos"].mean().cpu().item()), global_step)
    writer.add_scalar("cosine/negative_mean", float(stats["neg_cos"].mean().cpu().item()), global_step)
    writer.add_scalar("triplet/violation_ratio", float(stats["violations"].mean().cpu().item()), global_step)
    writer.add_scalar("spread/dim_std_mean", float(spread_stats["dim_std_mean"].float().cpu().item()), global_step)
    writer.add_scalar("spread/variance_loss", float(spread_stats["variance_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/covariance_loss", float(spread_stats["covariance_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/correlation_loss", float(spread_stats["correlation_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/uniformity_loss", float(spread_stats["uniformity_loss"].float().cpu().item()), global_step)
    writer.add_scalar("spread/top_eigen_ratio", float(spread_stats["top_eigen_ratio"].float().cpu().item()), global_step)
    writer.add_scalar("spread/effective_rank", float(spread_stats["effective_rank"].float().cpu().item()), global_step)
    for key in ("masked_fraction", "accuracy", "infonce_loss", "triplet_loss"):
        if key in stats:
            writer.add_scalar(f"infonce/{key}", float(stats[key].float().cpu().item()), global_step)
    for key, value in matryoshka_stats.items():
        writer.add_scalar(f"matryoshka/{key}", float(value.float().cpu().item()), global_step)
    for key, value in content_stats.items():
        writer.add_scalar(f"content/{key}", float(value.float().cpu().item()), global_step)
    writer.add_scalar("embedding/pre_norm_mean", float(norms.detach().float().mean().cpu().item()), global_step)
    writer.add_scalar("embedding/pre_norm_std", float(norms.detach().float().std(unbiased=False).cpu().item()), global_step)
    writer.add_scalar("optim/lr", float(optimizer.param_groups[0]["lr"]), global_step)
    for group in optimizer.param_groups[1:]:
        writer.add_scalar(f"optim/lr_{group.get('name', 'group')}", float(group["lr"]), global_step)
    grad_value = grad_norm.detach().float().cpu().item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
    writer.add_scalar("optim/grad_norm", grad_value, global_step)
    log_rule_batch_stats(writer, stats, batch, global_step)


def log_rule_batch_stats(
    writer: SummaryWriter,
    stats: dict[str, torch.Tensor],
    batch: dict[str, Any],
    global_step: int,
) -> None:
    rules = list(batch["rule"])
    weighted_loss = (stats["raw_loss"].detach() * batch["weight"].detach()).float().cpu()
    violations = stats["violations"].detach().float().cpu()
    pos_cos = stats["pos_cos"].detach().float().cpu()
    neg_cos = stats["neg_cos"].detach().float().cpu()
    for rule in sorted(set(rules)):
        mask = torch.tensor([value == rule for value in rules], dtype=torch.bool)
        writer.add_scalar(f"loss_by_rule/{rule}", float(weighted_loss[mask].mean().item()), global_step)
        writer.add_scalar(f"triplet_by_rule/{rule}_violation_ratio", float(violations[mask].mean().item()), global_step)
        writer.add_scalar(f"cosine_by_rule/{rule}_positive_mean", float(pos_cos[mask].mean().item()), global_step)
        writer.add_scalar(f"cosine_by_rule/{rule}_negative_mean", float(neg_cos[mask].mean().item()), global_step)
        writer.add_scalar(f"rule_mix/{rule}", float(mask.float().mean().item()), global_step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dataset", default=None, help="prebuilt dataset name or path; default: build from raw")
    args = parser.parse_args()
    out_dir = train_from_config(args.config, args.run_name, args.resume, args.dataset)
    print(f"run outputs: {out_dir}")


if __name__ == "__main__":
    main()
