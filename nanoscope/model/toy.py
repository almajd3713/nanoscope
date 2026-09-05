from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from nanoscope.model.registry import LMOutput, register_model


class ToyLM(nn.Module):
    """Small fixture model for infrastructure tests, not the M1 baseline."""

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):  # type: ignore[no-untyped-def]
        hidden = self.dropout(self.token_embedding(input_ids))
        return LMOutput(logits=self.projection(hidden))


def _build_toy(params: Mapping[str, Any]) -> ToyLM:
    return ToyLM(
        vocab_size=int(params["vocab_size"]),
        hidden_size=int(params.get("hidden_size", 32)),
        dropout=float(params.get("dropout", 0.0)),
    )


register_model("toy_lm", _build_toy)
