from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model import EncoderSettings, LayerFusion, MeanPool, QwenStyleEncoder, parameter_summary


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states, return_dict):
        hidden = self.embedding(input_ids)
        hidden_states = tuple(hidden + float(index) for index in range(5)) if output_hidden_states else None
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=hidden_states)


def settings(mode: str, layers: tuple[int, ...]) -> EncoderSettings:
    return EncoderSettings(
        base_model_path="unused",
        projection_dim=3,
        layer_mode=mode,
        layers=layers,
        dropout=0.0,
        lora_enabled=False,
        lora_rank=0,
        lora_alpha=0,
        lora_dropout=0.0,
        lora_targets=(),
    )


def test_mean_pool_masks_padding() -> None:
    pool = MeanPool()
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    mask = torch.tensor([[1, 0]])
    assert torch.allclose(pool(hidden, mask), torch.tensor([[1.0, 1.0]]))


def test_layer_fusion_concat_shape() -> None:
    fusion = LayerFusion("concat", (1, 3))
    hidden = torch.zeros(2, 4, 5)
    states = tuple(hidden + float(index) for index in range(4))
    out = fusion(hidden, states)
    assert out.shape == (2, 4, 10)


def test_qwen_style_encoder_fake_backbone_shape() -> None:
    model = QwenStyleEncoder(FakeBackbone(hidden_size=4), hidden_size=4, settings=settings("weighted_sum", (1, 2)))
    batch = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    out = model(**batch)
    assert out["embeddings"].shape == (2, 3)
    assert torch.allclose(out["embeddings"].norm(dim=-1), torch.ones(2), atol=1e-6)
    assert parameter_summary(model)["trainable"] > 0

