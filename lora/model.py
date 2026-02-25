"""LoRA 风格模型：Qwen3-0.6B（冻结）+ LoRA + 风格头 + ArcFace 头。"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hidden"))
from classifiers import ArcFaceHead  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "base-models" / "qwen-3-0.6b"
HIDDEN_SIZE = 1024
STYLE_DIM = 128
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


class StyleModel(nn.Module):
    def __init__(self, num_train_speakers: int):
        super().__init__()

        base = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        self.base = get_peft_model(base, lora_cfg)
        self.base.enable_input_require_grads()
        self.base.gradient_checkpointing_enable()

        # float32 头，避免 bfloat16 精度问题影响 ArcFace
        self.style_head = nn.Linear(HIDDEN_SIZE, STYLE_DIM, bias=False)
        self.arcface_head = ArcFaceHead(STYLE_DIM, num_train_speakers, s=30.0, m=0.3)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """返回 L2 归一化的 128 维风格向量（float32）。"""
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1].float()          # (B, L, 1024)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # (B, 1024)
        style = self.style_head(pooled)            # (B, 128)
        return F.normalize(style, dim=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        style_norm = self.encode(input_ids, attention_mask)
        if labels is not None:
            logits = self.arcface_head(style_norm, labels)
            loss = F.cross_entropy(logits, labels)
            return style_norm, logits, loss
        return style_norm, None, None
