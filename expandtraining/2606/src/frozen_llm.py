"""Training-free style vectors from a frozen causal LM, scored on the validation rows the trained runs use.

Every method gives one vector per validation row, so validate.write_validation_outputs reports it exactly as it
reports a checkpoint.

  raw      mean-pooled hidden state of the text at a layer: the untrained floor.
  diff     raw(text) - raw(neutral rewrite of the text): what the wording adds over its content
           (StyleVector, ACL 2025, built this way for steering; here it is read out as an embedding).
  persona  cosine of a row's diff to each reference speaker's mean diff: coordinates in a space of known voices.
  cond     per reference speaker k, log p(text | its neutral rewrite, k's example lines), centred over k: how much
           k's lines help predict this wording once its content is given.

Reference speakers come from the training split and never include held-out speakers; their example lines are
named only "speaker" so the LM cannot bring in what it knows about the character.
"""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from records import TripletRecord, normalize_text
from validate import cosine_summary, select_validation_triplets, speaker_of

NEUTRAL_PROMPT = (
    "Rewrite the text below in plain, neutral, standard written language: no colloquialisms, no catchphrases, no "
    "emotional particles, no character voice. Keep every piece of information and roughly the same length. Keep "
    "the language of the original (Simplified Chinese stays Simplified Chinese). Reply with the rewrite only."
)
# The English prompt translated 7.6% of Chinese lines into English (frozen_a); Chinese lines get a Chinese prompt.
NEUTRAL_PROMPT_ZH = (
    "把下面这句话改写成平实、中性的规范书面语：去掉口语、口头禅、语气词和人物腔调，保留全部信息，长度大致相同。"
    "必须用简体中文输出，不要翻译成其他语言。只输出改写结果。"
)


def is_chinese(text: str) -> bool:
    return sum("\u4e00" <= ch <= "\u9fff" for ch in text) > 0.3 * max(len(text), 1)


def language_kept(text: str, rewrite: str) -> bool:
    return is_chinese(text) == is_chinese(rewrite)


def validation_rows(triplets: list[TripletRecord], per_rule: int) -> tuple[pd.DataFrame, list[str]]:
    """The rows collect_validation_embeddings writes (same selection and metadata), with their full texts."""
    rows, texts = [], []
    for triplet in select_validation_triplets(triplets, per_rule):
        meta = triplet.meta
        for role in ("anchor", "positive", "negative"):
            text = normalize_text(getattr(triplet, f"{role}_text"))
            rows.append({
                "triplet_id": triplet.triplet_id,
                "role": role,
                "rule": triplet.rule,
                "heldout": bool(meta.get("heldout", False)),
                "speaker": speaker_of(meta, role),
                "db_id": int(meta.get(f"{role}_db_id", -1)),
                "char_length": len(text),
                "length_bucket": len(text) // 40,
                "text_preview": text[:120],
            })
            texts.append(text)
    return pd.DataFrame(rows), texts


def reference_lines(triplets: list[TripletRecord], speakers: int, lines: int) -> dict[str, list[str]]:
    """Training-split lines of the `speakers` training speakers with most lines, `lines` each, evenly spread."""
    by_speaker: defaultdict[str, set[str]] = defaultdict(set)
    for triplet in triplets:
        if triplet.split == "train" and triplet.rule == "same_speaker_real" and not triplet.meta.get("heldout"):
            speaker = speaker_of(triplet.meta, "anchor")
            if speaker:
                by_speaker[speaker].add(normalize_text(triplet.anchor_text))
    ranked = sorted(by_speaker, key=lambda name: (-len(by_speaker[name]), name))[:speakers]
    chosen = {}
    for name in ranked:
        pool = sorted(by_speaker[name], key=lambda text: hashlib.sha1(text.encode()).hexdigest())
        chosen[name] = [text for text in pool if 4 <= len(text) <= 60][:lines]
    return chosen


def load_lm(path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype).to(device).eval()
    return tokenizer, model


def _by_length(texts: list[str]) -> list[int]:
    return sorted(range(len(texts)), key=lambda i: len(texts[i]))


@torch.no_grad()
def neutralize(texts: list[str], tokenizer, model, batch_size: int = 64) -> list[str]:
    """One greedy neutral rewrite per text (Qwen3 chat template, thinking off)."""
    tokenizer.padding_side = "left"
    out: list[str] = [""] * len(texts)
    order = _by_length(texts)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": NEUTRAL_PROMPT_ZH if is_chinese(texts[i]) else NEUTRAL_PROMPT},
                 {"role": "user", "content": texts[i]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for i in index
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        longest = max(len(texts[i]) for i in index)
        generated = model.generate(**encoded, max_new_tokens=min(2 * longest + 32, 512), do_sample=False)
        for i, row in zip(index, generated[:, encoded["input_ids"].shape[1]:]):
            rewrite = normalize_text(tokenizer.decode(row, skip_special_tokens=True))
            out[i] = rewrite or texts[i]
    return out


@torch.no_grad()
def hidden_means(texts: list[str], tokenizer, model, layers: list[int], batch_size: int = 64,
                 max_length: int = 192) -> dict[int, np.ndarray]:
    """Mean-pooled hidden state of each text at each layer (0 = embeddings; the last layer is pre final norm)."""
    tokenizer.padding_side = "right"
    out = {layer: np.zeros((len(texts), model.config.hidden_size), dtype=np.float32) for layer in layers}
    order = _by_length(texts)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        encoded = tokenizer([texts[i] for i in index], return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(model.device)
        hidden = model(**encoded, output_hidden_states=True).hidden_states
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        for layer in layers:
            pooled = (hidden[layer].float() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            out[layer][index] = pooled.cpu().numpy()
    return out


@torch.no_grad()
def neuron_means(texts: list[str], tokenizer, model, layers: list[int], batch_size: int = 64,
                 max_length: int = 192) -> dict[int, np.ndarray]:
    """Mean-pooled MLP neuron activations (the input to down_proj) of decoder layer `layer` - 1, so layer L
    pairs with hidden_states[L]. float16: a 4B model has 9728 neurons a layer."""
    tokenizer.padding_side = "right"
    width = model.config.intermediate_size
    out = {layer: np.zeros((len(texts), width), dtype=np.float16) for layer in layers}
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        model.model.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(
            lambda _module, inputs, layer=layer: captured.__setitem__(layer, inputs[0]))
        for layer in layers
    ]
    try:
        order = _by_length(texts)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            encoded = tokenizer([texts[i] for i in index], return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length).to(model.device)
            model.model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            for layer in layers:
                pooled = (captured[layer].float() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
                out[layer][index] = pooled.cpu().numpy().astype(np.float16)
    finally:
        for hook in hooks:
            hook.remove()
    return out


def fisher_neurons(diff_by_speaker: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per neuron: variance of the speakers' mean shifts over the mean within-speaker variance (how well the
    neuron's line-minus-rewrite shift tells speakers apart), and the pooled within-speaker std. Neurons are
    ranked on training speakers only."""
    means = np.stack([d.mean(0) for d in diff_by_speaker.values()])
    within = np.stack([d.var(0) for d in diff_by_speaker.values()]).mean(0)
    return means.var(0) / np.clip(within, 1e-6, None), np.sqrt(np.clip(within, 1e-6, None))


def persona_coordinates(diff: np.ndarray, reference: dict[str, np.ndarray]) -> np.ndarray:
    """Cosine of each row's diff to each reference speaker's mean diff, after removing the mean over speakers."""
    centres = np.stack([vectors.mean(0) for vectors in reference.values()])
    centres = centres - centres.mean(0, keepdims=True)
    unit = lambda m: m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-8, None)  # noqa: E731
    return unit(diff) @ unit(centres).T


def _example_prefix(lines: list[str]) -> str:
    return "Lines by one speaker:\n" + "\n".join(f"- {line}" for line in lines) + "\n\n"


@torch.no_grad()
def conditional_scores(texts: list[str], neutrals: list[str], examples: dict[str, list[str]], tokenizer, model,
                       batch_size: int = 64) -> np.ndarray:
    """(texts, speakers) mean log-probability per token of each text given its neutral rewrite and one reference
    speaker's lines, centred over speakers. Each speaker's prefix is encoded once and its cache reused."""
    tokenizer.padding_side = "right"
    scores = np.zeros((len(texts), len(examples)), dtype=np.float32)
    suffix_heads = [f"Same content, said plainly: {neutral}\nThe same speaker, in their own words: " for neutral in neutrals]
    order = _by_length(texts)
    for column, lines in enumerate(examples.values()):
        prefix = tokenizer(_example_prefix(lines), return_tensors="pt").to(model.device)
        prefix_cache = model(**prefix, use_cache=True).past_key_values
        prefix_len = prefix["input_ids"].shape[1]
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            heads = [tokenizer(suffix_heads[i], add_special_tokens=False)["input_ids"] for i in index]
            bodies = [tokenizer(texts[i], add_special_tokens=False)["input_ids"] for i in index]
            seqs = [h + b for h, b in zip(heads, bodies)]
            width = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), width), tokenizer.pad_token_id, dtype=torch.long)
            target = torch.zeros((len(seqs), width), dtype=torch.bool)
            mask = torch.zeros((len(seqs), prefix_len + width), dtype=torch.long)
            mask[:, :prefix_len] = 1
            for row, (head, seq) in enumerate(zip(heads, seqs)):
                ids[row, :len(seq)] = torch.tensor(seq)
                target[row, len(head):len(seq)] = True
                mask[row, prefix_len:prefix_len + len(seq)] = 1
            cache = copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(len(seqs))
            ids, target, mask = ids.to(model.device), target.to(model.device), mask.to(model.device)
            hidden = model.model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                                 use_cache=True).last_hidden_state
            # token t is predicted from position t-1; the first suffix token from the prefix's last state, which
            # never belongs to the scored text (every head is non-empty), so it is skipped. The LM head runs only
            # on scored positions: full-vocabulary logits for the whole batch do not fit.
            scored = target[:, 1:]
            states, labels = hidden[:, :-1][scored], ids[:, 1:][scored]
            logp = torch.cat([
                torch.log_softmax(model.lm_head(states[i:i + 1024]).float(), dim=-1)
                .gather(-1, labels[i:i + 1024].unsqueeze(-1)).squeeze(-1)
                for i in range(0, len(labels), 1024)
            ])
            owner = torch.arange(len(seqs), device=ids.device).unsqueeze(1).expand_as(scored)[scored]
            mean = torch.zeros(len(seqs), device=ids.device).index_add_(0, owner, logp)
            mean = mean / scored.sum(1).clamp_min(1)
            scores[index, column] = mean.cpu().numpy()
            del cache
    return scores - scores.mean(1, keepdims=True)


def summary_of(rows: pd.DataFrame, embeddings: np.ndarray) -> dict[str, float]:
    """cosine_summary from embeddings alone (a frozen method has no training loss to read cosines from)."""
    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8, None)
    frame = rows.reset_index(drop=True)
    index = {(t, r): i for i, (t, r) in enumerate(zip(frame["triplet_id"], frame["role"]))}
    ids = frame.loc[frame["role"] == "anchor", "triplet_id"]
    a = [index[(t, "anchor")] for t in ids]
    pos = torch.tensor((unit[a] * unit[[index[(t, "positive")] for t in ids]]).sum(1))
    neg = torch.tensor((unit[a] * unit[[index[(t, "negative")] for t in ids]]).sum(1))
    return cosine_summary(pos, neg, embeddings, frame, ())

