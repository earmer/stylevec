"""测试不同 batch size 对显存的影响。"""

import gc
import os
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hidden"))

import torch
from model import StyleModel, MODEL_PATH

MAX_LEN = 128
NUM_SPEAKERS = 100
RANK = 16
BATCH_SIZES = [8, 16, 24, 32, 40, 48]
WARMUP_STEPS = 3
STABLE_STEPS = 5


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_proc = psutil.Process(os.getpid())


def get_mem_mb(device):
    """CUDA 用 max_memory_allocated；MPS/CPU 用系统可用内存反推。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        torch.mps.synchronize()
    # macOS 统一内存：用 total - available 作为系统已用量
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1e6


def reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def clear_cache(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def make_batch(batch_size, device):
    input_ids = torch.randint(0, 1000, (batch_size, MAX_LEN), device=device)
    attention_mask = torch.ones(batch_size, MAX_LEN, dtype=torch.long, device=device)
    labels = torch.randint(0, NUM_SPEAKERS, (batch_size,), device=device)
    return input_ids, attention_mask, labels


def bench_one(model, optimizer, batch_size, device):
    """跑若干步 forward+backward，等稳定后测显存。"""
    reset_peak(device)

    # warmup: 让显存分配稳定
    for _ in range(WARMUP_STEPS):
        input_ids, attention_mask, labels = make_batch(batch_size, device)
        _, _, loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 稳定测量
    reset_peak(device)
    measurements = []
    for _ in range(STABLE_STEPS):
        input_ids, attention_mask, labels = make_batch(batch_size, device)
        _, _, loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        measurements.append(get_mem_mb(device))

    return measurements


def main():
    device = detect_device()
    print(f"device: {device}")
    print(f"rank: {RANK}")
    print(f"seq_len: {MAX_LEN}")
    print(f"batch_sizes: {BATCH_SIZES}")
    print()

    model = StyleModel(NUM_SPEAKERS, lora_r=RANK).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    # 测模型加载后的基线显存
    baseline = get_mem_mb(device)
    print(f"baseline (model + optimizer): {baseline:.1f} MB")
    print()
    print(f"{'batch':>6}  {'peak_mb':>10}  {'delta_mb':>10}  {'per_sample_mb':>14}")
    print("-" * 50)

    for bs in BATCH_SIZES:
        clear_cache(device)
        time.sleep(1)

        measurements = bench_one(model, optimizer, bs, device)
        peak = max(measurements)
        delta = peak - baseline
        per_sample = delta / bs

        print(f"{bs:>6}  {peak:>10.1f}  {delta:>10.1f}  {per_sample:>14.1f}")

    print()
    print("done.")


if __name__ == "__main__":
    main()
