"""LoRA 风格模型：Qwen3-0.6B（冻结）+ LoRA + 风格头 + ArcFace 头。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from classifiers import ArcFaceHead  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "base-models" / "qwen-3-0.6b"
HIDDEN_SIZE = 1024
STYLE_DIM = 128
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


class LayerFusion(nn.Module):
    def __init__(self, layer_indices: list[int]):
        super().__init__()
        self.layer_indices = layer_indices
        self.weights = nn.Parameter(torch.zeros(len(layer_indices)))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        w = F.softmax(self.weights, dim=0)
        selected = torch.stack([hidden_states[i] for i in self.layer_indices])
        return torch.einsum("n, n b l h -> b l h", w, selected)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("b l h, h -> b l", h, self.query)
        scores = scores / (h.size(-1) ** 0.5)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("b l, b l h -> b h", weights, h)


class StyleModel(nn.Module):
    def __init__(self, num_train_speakers: int, lora_r: int = LORA_R, lora_alpha: int = LORA_ALPHA,
                 fusion_layers=None, attn_pool=False):
        super().__init__()

        base = AutoModel.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        self.base = get_peft_model(base, lora_cfg)
        self.base.enable_input_require_grads()
        self.base.gradient_checkpointing_enable()

        self.layer_fusion = LayerFusion(fusion_layers) if fusion_layers else None
        self.attn_pool = AttentionPooling(HIDDEN_SIZE) if attn_pool else None

        # float32 头，避免 bfloat16 精度问题影响 ArcFace
        self.style_head = nn.Linear(HIDDEN_SIZE, STYLE_DIM, bias=False)
        self.arcface_head = ArcFaceHead(STYLE_DIM, num_train_speakers, s=30.0, m=0.3)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化的 128 维风格向量（float32）。"""
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(self.layer_fusion is not None),
        )
        h = self.layer_fusion(out.hidden_states) if self.layer_fusion else out.last_hidden_state
        if self.attn_pool:
            pooled = self.attn_pool(h, attention_mask)
        else:
            mask = attention_mask.unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = pooled.float()
        style = self.style_head(pooled)
        return F.normalize(style, dim=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        style_norm = self.encode(input_ids, attention_mask)
        if labels is not None:
            logits = self.arcface_head(style_norm, labels)
            loss = F.cross_entropy(logits, labels)
            return style_norm, logits, loss
        return style_norm, None, None
