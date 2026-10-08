"""GPT-2-style transformer: learned positions, LayerNorm, GELU MLP, multi-head attention.

This is the control the modern block (modern.py) is measured against. It is a `Decoder`
(nanoscope/blocks/structure.py) built from library blocks: read that file for the loop over
layers, and nanoscope/blocks/ for what each part does.
"""

from __future__ import annotations

from nanoscope.blocks.attention import Attention
from nanoscope.blocks.embedding import LearnedPosition
from nanoscope.blocks.mlp import GELUMLP
from nanoscope.blocks.norm import LayerNorm
from nanoscope.blocks.registry import shipped
from nanoscope.blocks.structure import Block, Decoder


@shipped
class GPT2(Decoder):
    """A GPT-2 style decoder: learned positions, LayerNorm, GELU MLP, multi-head attention."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model,
            n_layers=n_layers,
            block=Block(norm=LayerNorm(), attn=Attention(n_heads=n_heads, bias=True),
                        mlp=GELUMLP(bias=True)),
            final_norm=LayerNorm(), pos_emb=LearnedPosition(), tie_weights=True)
