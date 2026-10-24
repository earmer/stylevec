from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderSettings:
    base_model_path: str
    projection_dim: int
    layer_mode: str
    layers: tuple[int, ...]
    dropout: float
    lora_enabled: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: tuple[str, ...]
    center: str = "none"
    center_momentum: float = 0.01
    content_dim: int = 0


class MeanPool(nn.Module):
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [batch, tokens, dim], got {hidden.shape}")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be [batch, tokens], got {attention_mask.shape}")
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


CENTER_MODES = ("none", "running")
CENTER_BUFFERS = ("center.mean", "center.initialized")


class RunningCenter(nn.Module):
    """Subtracts a running mean of the pooled hidden states before the projection.

    Mean-pooled LLM hidden states share one large common direction, so every embedding starts
    in a narrow cone (cosine ~0.9 between any two lines) and the triplet hinge has little to push on.
    The mean is an EMA updated without gradient in training mode and frozen in eval, so anchors,
    positives and negatives (encoded in separate forwards) and validation all use the same shift. Its
    buffers are saved with the trained weights (model.trainable_state_dict)."""

    def __init__(self, dim: int, momentum: float) -> None:
        super().__init__()
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"center momentum must be in (0, 1], got {momentum}")
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_mean = pooled.detach().float().mean(dim=0)
                if bool(self.initialized):
                    self.mean.lerp_(batch_mean, self.momentum)
                else:
                    self.mean.copy_(batch_mean)
                    self.initialized.fill_(True)
        return pooled - self.mean.to(dtype=pooled.dtype)


class LayerFusion(nn.Module):
    def __init__(self, mode: str, layers: tuple[int, ...]) -> None:
        super().__init__()
        if mode not in {"last", "weighted_sum", "concat"}:
            raise ValueError(f"unsupported layer fusion mode: {mode}")
        if mode != "last" and not layers:
            raise ValueError("layers must be non-empty unless layer mode is last")
        self.mode = mode
        self.layers = layers
        self.layer_logits = nn.Parameter(torch.zeros(len(layers))) if mode == "weighted_sum" else None

    def output_multiplier(self) -> int:
        return len(self.layers) if self.mode == "concat" else 1

    def forward(self, last_hidden: torch.Tensor, hidden_states: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        if self.mode == "last":
            return last_hidden
        if hidden_states is None:
            raise ValueError("hidden_states are required for layer fusion")
        max_layer = max(self.layers)
        if max_layer >= len(hidden_states):
            raise ValueError(f"requested layer {max_layer}, but model returned {len(hidden_states)} hidden states")
        selected = [hidden_states[index] for index in self.layers]
        if self.mode == "concat":
            return torch.cat(selected, dim=-1)
        if self.layer_logits is None:
            raise RuntimeError("weighted_sum layer logits are missing")
        weights = torch.softmax(self.layer_logits, dim=0).to(dtype=selected[0].dtype)
        stacked = torch.stack(selected, dim=0)
        return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)


class QwenStyleEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, settings: EncoderSettings) -> None:
        super().__init__()
        self.backbone = backbone
        self.settings = settings
        self.fusion = LayerFusion(settings.layer_mode, settings.layers)
        projection_in = hidden_size * self.fusion.output_multiplier()
        self.pool = MeanPool()
        self.dropout = nn.Dropout(settings.dropout)
        if settings.center not in CENTER_MODES:
            raise ValueError(f"model.center must be one of {CENTER_MODES}, got {settings.center!r}")
        self.center = RunningCenter(projection_in, settings.center_momentum) if settings.center == "running" else None
        self.projection = nn.Linear(projection_in, settings.projection_dim)
        if settings.content_dim < 0:
            raise ValueError(f"model.content_dim must be non-negative, got {settings.content_dim}")
        # A small separate head trained to hold content (issue #1): the style embedding stays projection_dim
        # wide, and validation, retrieval and export never read this head.
        self.content_head = nn.Linear(projection_in, settings.content_dim) if settings.content_dim > 0 else None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        need_hidden = self.settings.layer_mode != "last"
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
            return_dict=True,
        )
        token_hidden = self.fusion(outputs.last_hidden_state, outputs.hidden_states if need_hidden else None)
        pooled = self.pool(token_hidden, attention_mask)
        pooled = pooled.to(dtype=self.projection.weight.dtype)
        if self.center is not None:
            pooled = self.center(pooled)
        pooled = self.dropout(pooled)
        projected = self.projection(pooled)
        pre_norm = projected.norm(dim=-1)
        embeddings = F.normalize(projected, p=2, dim=-1, eps=1e-8)
        output = {"embeddings": embeddings, "pre_norm": pre_norm}
        if self.content_head is not None:
            output["content"] = F.normalize(self.content_head(pooled), p=2, dim=-1, eps=1e-8)
        return output


def build_encoder_settings(config: dict[str, Any]) -> EncoderSettings:
    model_cfg = config["model"]
    lora_cfg = model_cfg.get("lora", {})
    fusion_cfg = model_cfg.get("layer_fusion", {})
    return EncoderSettings(
        base_model_path=str(model_cfg["base_model_path"]),
        projection_dim=int(model_cfg["projection_dim"]),
        layer_mode=str(fusion_cfg.get("mode", "last")),
        layers=tuple(int(value) for value in fusion_cfg.get("layers", [])),
        dropout=float(model_cfg.get("dropout", 0.0)),
        lora_enabled=bool(lora_cfg.get("enabled", True)),
        lora_rank=int(lora_cfg.get("rank", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        lora_targets=tuple(str(value) for value in lora_cfg.get("targets", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        center=str(model_cfg.get("center", "none")),
        center_momentum=float(model_cfg.get("center_momentum", 0.01)),
        content_dim=int(model_cfg.get("content_dim", 0)),
    )


def build_qwen_style_encoder(config: dict[str, Any]) -> QwenStyleEncoder:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModel

    settings = build_encoder_settings(config)
    precision = str(config.get("runtime", {}).get("precision", "fp32"))
    if precision == "fp32":
        load_dtype = torch.float32
    elif precision == "bf16":
        load_dtype = torch.bfloat16
    elif precision == "fp16":
        load_dtype = torch.float16
    else:
        raise ValueError(f"unsupported runtime.precision: {precision}")
    hf_config = AutoConfig.from_pretrained(settings.base_model_path, local_files_only=True, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        settings.base_model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=load_dtype,
    )
    if bool(config["model"].get("gradient_checkpointing", False)):
        if hasattr(backbone, "gradient_checkpointing_enable"):
            # Non-reentrant: the reentrant default drops gradients to LoRA weights when the checkpointed
            # block's input (frozen embeddings) does not require grad.
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
    if settings.lora_enabled:
        lora_config = LoraConfig(
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            target_modules=list(settings.lora_targets),
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad = False
    hidden_size = int(getattr(hf_config, "hidden_size"))
    return QwenStyleEncoder(backbone=backbone, hidden_size=hidden_size, settings=settings)


TRIPLET_ROLES = ("anchor", "positive", "negative")


def encode_triplet(
    model: QwenStyleEncoder,
    batch: dict[str, Any],
    combine_forward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = encode_triplet_outputs(model, batch, combine_forward)
    anchor, positive, negative = outputs["embeddings"]
    return anchor, positive, negative, outputs["pre_norm"]


def encode_triplet_outputs(
    model: QwenStyleEncoder,
    batch: dict[str, Any],
    combine_forward: bool = False,
    detach_twin: bool = False,
) -> dict[str, Any]:
    """Each output key as an (anchor, positive, negative) tuple, except pre_norm, which is concatenated.
    A batch with a "twin" role (the collate's content twins) also gets outputs["twin"], its embeddings.
    detach_twin encodes the twins in a separate forward without gradient: a target, not a trained input,
    so it needs no activation memory (the leak regularisers)."""
    if detach_twin and "twin" in batch:
        outputs = encode_triplet_outputs(model, {k: v for k, v in batch.items() if k != "twin"}, combine_forward)
        with torch.no_grad():
            twin = model(**batch["twin"])
            outputs["twin"] = twin["embeddings"]
            if "content" in twin:
                outputs["twin_content"] = twin["content"]
        return outputs
    if combine_forward:
        return encode_triplet_combined(model, batch)
    roles = [model(**batch[role]) for role in batch_roles(batch)]
    return split_roles(roles)


def batch_roles(batch: dict[str, Any]) -> tuple[str, ...]:
    return TRIPLET_ROLES + (("twin",) if "twin" in batch else ())


def split_roles(roles: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    outputs: dict[str, Any] = {key: tuple(role[key] for role in roles[:3]) for key in roles[0] if key != "pre_norm"}
    outputs["pre_norm"] = torch.cat([role["pre_norm"] for role in roles[:3]])
    if len(roles) > 3:
        outputs["twin"] = roles[3]["embeddings"]
    return outputs


def encode_triplet_combined(
    model: QwenStyleEncoder,
    batch: dict[str, Any],
) -> dict[str, Any]:
    parts = [batch[role] for role in batch_roles(batch)]
    max_length = max(part["input_ids"].shape[1] for part in parts)
    input_ids = torch.cat([_pad_tokens(part["input_ids"], max_length, pad_value=0) for part in parts], dim=0)
    attention_mask = torch.cat([_pad_tokens(part["attention_mask"], max_length, pad_value=0) for part in parts], dim=0)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    batch_size = parts[0]["input_ids"].shape[0]
    roles = [{key: value[i * batch_size:(i + 1) * batch_size] for key, value in output.items()} for i in range(len(parts))]
    return split_roles(roles)


def _pad_tokens(values: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    missing = length - values.shape[1]
    if missing <= 0:
        return values
    return F.pad(values, (0, missing), value=pad_value)


def parameter_summary(module: nn.Module) -> dict[str, float]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {
        "total": float(total),
        "trainable": float(trainable),
        "ratio": float(trainable / total) if total else 0.0,
    }


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Trained parameters plus the learned (non-gradient) state the forward depends on: the running
    center's buffers."""
    state = {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    state.update({name: buffer.detach().cpu() for name, buffer in module.named_buffers() if name.endswith(CENTER_BUFFERS)})
    return state


def load_trainable_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    missing = [name for name in state if name not in current]
    if missing:
        raise ValueError(f"checkpoint has unknown trainable parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            if current[name].shape != value.shape:
                raise ValueError(f"shape mismatch for {name}: {current[name].shape} != {value.shape}")
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
