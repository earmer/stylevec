"""Leak regularisers: penalties that keep a line's own content out of its style embedding, judged against its
content twin (the anchor's paraphrase, or a rewrite's source; dataset.twin_columns).

The leak InfoNCE taught is pairwise: z(line) close to z(its paraphrase). A linear decorrelation from a small
content code (infonce_b chead) sat at its noise floor while the leak grew, so these methods all see the
instance-level pair. The origin_vs_para hinge already targets the pair but fires on ~3% of rows; a twin
reaches ~63% of rows, which is the coverage these methods add.

One method per run, `regularization.leak_method` (`none` is off), at `leak_weight` until `leak_until_step`
(0 = every step). The twin is the target, encoded without gradient (encode_triplet_outputs detach_twin), so a
method costs one extra forward and no gradient checkpointing. A row whose twin is its own negative reads the
twin from the (detached) negative. `leak_queue` keeps that many past (style, twin) pairs, detached, so batch
statistics see more than one micro-batch's ~20 pairs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch.func import functional_call

LEAK_METHODS = ("none", "hsic", "adversary", "subspace")
HSIC_KERNELS = ("rbf", "linear")


def leak_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    regularization = config.get("regularization", {})
    return {
        "method": str(regularization.get("leak_method", "none")),
        "weight": float(regularization.get("leak_weight", 0.0)),
        "until_step": int(regularization.get("leak_until_step", 0)),
        "queue": int(regularization.get("leak_queue", 0)),
        "hsic_kernel": str(regularization.get("leak_hsic_kernel", "rbf")),
        "probe_temperature": float(regularization.get("leak_probe_temperature", 0.1)),
        "probe_warmup": int(regularization.get("leak_probe_warmup", 0)),
    }


def leak_enabled(settings: dict[str, Any]) -> bool:
    """subspace is a projection inside the model (model.leak_null_*), not a weighted loss."""
    if settings["method"] == "subspace":
        return True
    return settings["method"] != "none" and settings["weight"] > 0.0


def validate_leak_config(config: dict[str, Any]) -> None:
    settings = leak_settings_from_config(config)
    if settings["method"] not in LEAK_METHODS:
        raise ValueError(f"regularization.leak_method must be one of {LEAK_METHODS}, got {settings['method']!r}")
    if settings["hsic_kernel"] not in HSIC_KERNELS:
        raise ValueError(f"regularization.leak_hsic_kernel must be one of {HSIC_KERNELS}")
    if settings["method"] == "adversary" and int(config["model"].get("leak_probe_hidden", 0)) <= 0:
        raise ValueError("regularization.leak_method adversary needs model.leak_probe_hidden > 0")
    if settings["probe_temperature"] <= 0.0 or settings["probe_warmup"] < 0:
        raise ValueError("regularization.leak_probe_temperature must be positive, leak_probe_warmup >= 0")
    null_dims = int(config["model"].get("leak_null_dims", 0))
    if (settings["method"] == "subspace") != (null_dims > 0):
        raise ValueError("regularization.leak_method subspace and model.leak_null_dims > 0 go together")
    if settings["weight"] < 0.0:
        raise ValueError("regularization.leak_weight must be non-negative")
    if settings["queue"] < 0:
        raise ValueError("regularization.leak_queue must be non-negative (0 = this micro-batch only)")
    max_steps = int(config["train"].get("max_steps", 0))
    if settings["until_step"] < 0 or (settings["until_step"] > 0 and settings["until_step"] > max_steps):
        raise ValueError("regularization.leak_until_step must be in [0, train.max_steps] (0 = off)")


def _gram(x: torch.Tensor, kernel: str) -> torch.Tensor:
    """Kernel matrix with a zero diagonal (the unbiased estimator's form). RBF bandwidth: the median
    pairwise squared distance (detached)."""
    off_diagonal = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    if kernel == "linear":
        gram = x @ x.T
    else:
        squared = x.pow(2).sum(dim=-1)
        distances = (squared[:, None] + squared[None, :] - 2.0 * x @ x.T).clamp_min(0.0)  # cdist: no MPS backward
        gram = torch.exp(-distances / distances[off_diagonal].detach().median().clamp_min(1e-6))
    return gram * off_diagonal


def _unbiased_hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Song et al. (2012) unbiased HSIC of zero-diagonal kernels; needs n >= 4."""
    n = k.shape[0]
    ones_k, ones_l = k.sum(), l.sum()
    trace = (k * l).sum()
    cross = (k.sum(dim=0) * l.sum(dim=0)).sum()
    return (trace + ones_k * ones_l / ((n - 1) * (n - 2)) - 2.0 * cross / (n - 2)) / (n * (n - 3))


def kernel_alignment(style: torch.Tensor, twin: torch.Tensor, kernel: str = "rbf") -> torch.Tensor:
    """Debiased normalised HSIC (CKA) between style embeddings and their twins, row i paired with row i:
    ~0 when the style geometry carries nothing of the twins' (content) geometry, 1 when it reproduces it.
    Debiased because plain CKA of independent vectors sits well above 0 at these sizes (~0.4 for 64 rows),
    the same noise floor that hid the leak from chead's linear penalty. Nonlinear with the RBF kernel."""
    if style.shape != twin.shape or style.ndim != 2 or style.shape[0] < 4:
        raise ValueError(f"style and twin must be matching [n >= 4, dim], got {style.shape}, {twin.shape}")
    with torch.autocast(device_type=style.device.type, enabled=False):
        k = _gram(style.float(), kernel)
        l = _gram(twin.float(), kernel)
        scale = (_unbiased_hsic(k, k).clamp_min(1e-12) * _unbiased_hsic(l, l).clamp_min(1e-12)).sqrt()
        return _unbiased_hsic(k, l) / scale


class PairQueue:
    """The last `size` (style, twin) pairs, detached, oldest dropped first."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.style: torch.Tensor | None = None
        self.twin: torch.Tensor | None = None

    def extend(self, style: torch.Tensor, twin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This batch's pairs followed by the queued ones (gradient only through `style`'s rows); then
        queues this batch's pairs."""
        if self.size <= 0 or self.style is None:
            all_style, all_twin = style, twin
        else:
            all_style, all_twin = torch.cat([style, self.style]), torch.cat([twin, self.twin])
        if self.size > 0:
            self.style = all_style.detach()[: self.size]
            self.twin = all_twin.detach()[: self.size]
        return all_style, all_twin


class LeakState:
    """Per-run state a method keeps across steps: the pair queue, the adversary's probe (a module of the
    encoder, so the optimizer and checkpoints already hold it), and the encoder's content_null (subspace),
    whose covariance that method feeds."""

    def __init__(self, settings: dict[str, Any], probe: torch.nn.Module | None = None,
                 null: torch.nn.Module | None = None) -> None:
        self.settings = settings
        self.queue = PairQueue(settings["queue"])
        self.probe = probe
        self.null = null
        if settings["method"] == "adversary" and leak_enabled(settings) and probe is None:
            raise ValueError("leak_method adversary needs the encoder's leak_probe (model.leak_probe_hidden)")
        if settings["method"] == "subspace" and null is None:
            raise ValueError("leak_method subspace needs the encoder's content_null (model.leak_null_dims)")


def probe_logits(probe: torch.nn.Module, style: torch.Tensor, twins: torch.Tensor, temperature: float,
                 frozen: bool = False) -> torch.Tensor:
    """[rows, candidates]: the probe's guess at each style row's twin among all queued twins. frozen reads
    the probe with detached weights, so only the style side gets that term's gradient."""
    params = {name: value.detach() for name, value in probe.named_parameters()} if frozen else None
    guess = functional_call(probe, params, (style,)) if frozen else probe(style)
    return F.normalize(guess.float(), dim=-1) @ F.normalize(twins.float(), dim=-1).T / temperature


def adversary_loss(probe: torch.nn.Module, style: torch.Tensor, twins: torch.Tensor, current: int,
                   settings: dict[str, Any], global_step: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Two players, no gradient reversal (its unbounded objective tears the embedding apart). The probe
    learns, on detached style rows (this batch and the queue), to pick each row's own twin among every
    queued twin: an instance-level content readout. The encoder, through the frozen probe on this batch's
    rows only, is pushed to make that guess uniform (KL to uniform = log n - entropy), ramped in over
    leak_probe_warmup steps so the probe learns first. Chance accuracy is 1 / n. Returns (probe loss,
    encoder loss, stats): leak_weight scales only the encoder's side."""
    n = twins.shape[0]
    target = torch.arange(n, device=style.device)
    probe_ce = F.cross_entropy(probe_logits(probe, style.detach(), twins, settings["probe_temperature"]), target)
    logits = probe_logits(probe, style[:current], twins, settings["probe_temperature"], frozen=True)
    log_probs = F.log_softmax(logits, dim=-1)
    confusion = (math.log(n) + (log_probs.exp() * log_probs).sum(dim=-1)).mean()
    ramp = min(1.0, (global_step + 1) / settings["probe_warmup"]) if settings["probe_warmup"] > 0 else 1.0
    stats = {
        "probe_ce": probe_ce.detach(),
        "probe_acc": (logits.argmax(dim=-1) == target[:current]).float().mean().detach(),
        "probe_chance": style.new_tensor(1.0 / n),
        "confusion": confusion.detach(),
        "ramp": style.new_tensor(ramp),
    }
    return probe_ce, confusion * ramp, stats


def subspace_update(state: LeakState, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Feeds this batch's unprojected twins to the encoder's content_null; returns how much of the anchors'
    and twins' (unprojected) embeddings sits in the content subspace."""
    anchor, _positive, negative = outputs["unprojected"]
    has_twin, is_negative = batch["has_twin"], batch["twin_is_negative"]
    twin = torch.where(is_negative[:, None], negative.detach(), outputs["twin_unprojected"])
    keep = has_twin | is_negative
    if int(keep.sum()) < 2:
        return {}
    state.null.update(twin[keep])
    return {"null_share_anchor": state.null.share(anchor[keep].detach()).detach(),
            "null_share_twin": state.null.share(twin[keep]).detach(),
            "null_updates": state.null.updates.float()}


def twin_pairs(outputs: dict[str, Any], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """(anchor style embeddings, detached twin embeddings) for every row with a twin."""
    anchor, _positive, negative = outputs["embeddings"]
    has_twin, is_negative = batch["has_twin"], batch["twin_is_negative"]
    twin = torch.where(is_negative[:, None], negative.detach(), outputs["twin"].detach())
    keep = has_twin | is_negative
    return anchor[keep], twin[keep]


def leak_step_loss(
    state: LeakState,
    outputs: dict[str, Any],
    batch: dict[str, Any],
    global_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = state.settings
    zero = outputs["embeddings"][0].new_tensor(0.0)
    if not leak_enabled(settings) or (0 < settings["until_step"] <= global_step):
        return zero, {}
    if settings["method"] == "subspace":
        return zero, subspace_update(state, outputs, batch)
    style, twin = twin_pairs(outputs, batch)
    if style.shape[0] < 4:
        return zero, {}
    current = style.shape[0]
    style, twin = state.queue.extend(style, twin)
    pair_cos = (style * twin).sum(dim=-1)
    stats = {"pairs": style.new_tensor(float(style.shape[0])), "pair_cos": pair_cos.detach().mean()}
    if settings["method"] == "hsic":
        alignment = kernel_alignment(style, twin, settings["hsic_kernel"])
        stats["hsic_cka"] = alignment.detach()
        return alignment * settings["weight"], stats
    if settings["method"] == "adversary":
        probe_ce, confusion, method_stats = adversary_loss(state.probe, style, twin, current, settings, global_step)
        return probe_ce + confusion * settings["weight"], {**stats, **method_stats}
    raise ValueError(f"unknown leak method {settings['method']!r}")
