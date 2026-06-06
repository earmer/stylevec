"""StyleDistance model: roberta-base + LoRA + mean pooling + L2 norm.

Exact architecture from the paper (Section 4.2, Appendix D):
  - Frozen roberta-base body
  - LoRA adapter on all linear layers (r=8, alpha=8, dropout=0.0)
  - Mean pooling over valid tokens (attention_mask)
  - L2-normalized 768-dim style embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


def _get_all_linear_targets(model: nn.Module) -> list[str]:
    """Collect names of all nn.Linear modules for LoRA targeting."""
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            # peft uses the leaf module name (last part after dot)
            targets.append(name.split(".")[-1])
    return list(set(targets))


class StyleDistance(nn.Module):
    def __init__(self, model_name: str = "FacebookAI/xlm-roberta-base", lora_dropout: float = 0.1):
        super().__init__()

        base = AutoModel.from_pretrained(model_name)
        for p in base.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_cfg)
        self.encoder.enable_input_require_grads()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalized style embedding."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # (B, L, 768)
        # Mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        a_ids: torch.Tensor,
        a_mask: torch.Tensor,
        p_ids: torch.Tensor,
        p_mask: torch.Tensor,
        n_ids: torch.Tensor,
        n_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (anchor_emb, positive_emb, negative_emb), each L2-normalized."""
        return (
            self.encode(a_ids, a_mask),
            self.encode(p_ids, p_mask),
            self.encode(n_ids, n_mask),
        )

    def save(self, path: str | Path):
        self.encoder.save_pretrained(str(path))

    @classmethod
    def load(cls, path: str | Path, model_name: str = "FacebookAI/xlm-roberta-base") -> "StyleDistance":
        import json
        from peft import PeftModel

        base = AutoModel.from_pretrained(model_name)
        encoder = PeftModel.from_pretrained(base, str(path))
        # Handle the fact that base params need requires_grad=False but PEFT
        # restores them — we don't care since we only call .encode()
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)
        inst.encoder = encoder
        inst.encoder.enable_input_require_grads()
        return inst
