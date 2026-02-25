"""隐藏层 + Attention 提取，4 种池化策略，缓存为 .npz。"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_PATH = Path(__file__).resolve().parent.parent / "base-models" / "qwen-3-0.6b"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
DEVICE = torch.device("mps")

N_LAYERS = None  # 动态检测：首次前向传播后设置
POOL_METHODS = ["last", "attn", "rev_comp", "rev_inv"]


def cache_path(layer, pool):
    return CACHE_DIR / f"hidden_L{layer}_{pool}.npz"


def all_cached():
    """检查是否所有缓存文件都存在（从已有缓存推断层数）。"""
    # 从 last pool 的文件推断层数
    layer = 0
    while cache_path(layer, "last").exists():
        layer += 1
    if layer == 0:
        return False
    # 检查所有 pool 是否齐全
    for l in range(layer):
        for pool in POOL_METHODS:
            if not cache_path(l, pool).exists():
                return False
    return True


def detect_n_layers():
    """从已有缓存推断层数。"""
    layer = 0
    while cache_path(layer, "last").exists():
        layer += 1
    return layer


def pool_hidden(hidden_states, attentions):
    """对单句的 hidden_states 和 attentions 计算 4 种池化。

    Args:
        hidden_states: tuple of (N_LAYERS,) tensors, each (1, seq_len, 1024)
        attentions: tuple of (28,) tensors, each (1, 16, seq_len, seq_len)

    Returns:
        dict[pool_method] -> np.array of shape (N_LAYERS, 1024)
    """
    n_layers = len(hidden_states)
    results = {p: [] for p in POOL_METHODS}

    for layer_idx in range(n_layers):
        hs = hidden_states[layer_idx][0]  # (seq_len, 1024)

        # last token pooling — always available
        last_vec = hs[-1].cpu().float().numpy()
        results["last"].append(last_vec)

        # attention-based pooling: layer 0 = embedding, no attention
        if layer_idx == 0:
            # 无 attention，所有 attention 池化退化为 last token
            for p in ["attn", "rev_comp", "rev_inv"]:
                results[p].append(last_vec)
            continue

        # attention weights: last token attending to all positions
        # attentions[layer_idx - 1] shape: (1, 16, seq_len, seq_len)
        attn = attentions[layer_idx - 1][0]  # (16, seq_len, seq_len)
        # 16 heads 均值，last token 的 attention 分布
        attn_weights = attn[:, -1, :].mean(dim=0)  # (seq_len,)

        hs_float = hs.float()  # (seq_len, 1024)

        # attn-weighted pooling
        w = attn_weights / (attn_weights.sum() + 1e-10)
        attn_vec = (w.unsqueeze(1) * hs_float).sum(dim=0).cpu().numpy()
        results["attn"].append(attn_vec)

        # reverse-attention complement: w' = 1 - w
        w_comp = 1.0 - attn_weights
        w_comp = w_comp / (w_comp.sum() + 1e-10)
        rev_comp_vec = (w_comp.unsqueeze(1) * hs_float).sum(dim=0).cpu().numpy()
        results["rev_comp"].append(rev_comp_vec)

        # reverse-attention inverse: w' = 1 / (w + eps)
        eps = 1e-8
        w_inv = 1.0 / (attn_weights + eps)
        w_inv = w_inv / (w_inv.sum() + 1e-10)
        rev_inv_vec = (w_inv.unsqueeze(1) * hs_float).sum(dim=0).cpu().numpy()
        results["rev_inv"].append(rev_inv_vec)

    return {p: np.stack(results[p]) for p in POOL_METHODS}  # each (N_LAYERS, 1024)


def extract_and_cache(all_texts):
    """对所有文本提取隐藏层，按层×池化缓存。

    Args:
        all_texts: list of str, 总共 1200 句（train 640 + val 160 + gen 400）
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if all_cached():
        print("All hidden state caches exist, skipping extraction.")
        return

    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(DEVICE).eval()

    global N_LAYERS
    n = len(all_texts)
    n_layers = None
    collectors = None

    print(f"Extracting hidden states for {n} sentences ...")
    with torch.no_grad():
        for i in tqdm(range(n), desc="extract"):
            inputs = tokenizer(all_texts[i], return_tensors="pt").to(DEVICE)
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

            # 首次前向：检测实际层数
            if n_layers is None:
                n_layers = len(outputs.hidden_states)
                N_LAYERS = n_layers
                collectors = {p: [[] for _ in range(n_layers)] for p in POOL_METHODS}
                print(f"  Detected {n_layers} hidden state layers, {len(outputs.attentions)} attention layers")

            pooled = pool_hidden(outputs.hidden_states, outputs.attentions)
            for p in POOL_METHODS:
                for layer_idx in range(n_layers):
                    collectors[p][layer_idx].append(pooled[p][layer_idx])

            # 释放显存
            del outputs, inputs, pooled
            if i % 100 == 0:
                torch.mps.empty_cache()

    # 保存缓存
    print("Saving caches ...")
    for p in POOL_METHODS:
        for layer_idx in range(n_layers):
            arr = np.stack(collectors[p][layer_idx])  # (n, 1024)
            np.savez_compressed(cache_path(layer_idx, p), embeddings=arr)
    print(f"Saved {n_layers * len(POOL_METHODS)} cache files.")

    del model, tokenizer
    torch.mps.empty_cache()


def load_cached(layer, pool):
    """加载单个缓存文件，返回 (n, 1024) 数组。"""
    return np.load(cache_path(layer, pool))["embeddings"]
