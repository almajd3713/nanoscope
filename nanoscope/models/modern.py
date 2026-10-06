"""The 2024-26 consensus decoder: RoPE, RMSNorm, SwiGLU, GQA, QK-norm, no biases,
tied embeddings and z-loss.

Every component has a switch, so a leave-one-out ablation is one keyword:
Modern(rope=False) uses learned positions instead, swiglu=False a GELU MLP,
rmsnorm=False LayerNorm, qk_norm=False no QK-norm, n_kv_heads=n_heads plain
multi-head attention, z_loss=0 no z-loss, tie_weights=False a separate output matrix.
ffn_hidden sets the MLP width, e.g. to match another model's parameter count
(see nanoscope.sizing.match_params).

It is a `Decoder` (nanoscope/blocks/structure.py) built from library blocks; each switch
below picks which block goes in its place.
"""

from __future__ import annotations

from nanoscope.blocks.attention import Attention
from nanoscope.blocks.embedding import LearnedPosition
from nanoscope.blocks.mlp import GELUMLP, SwiGLU
from nanoscope.blocks.norm import LayerNorm, RMSNorm
from nanoscope.blocks.positional import RoPE
from nanoscope.blocks.registry import shipped
from nanoscope.blocks.structure import Block, Decoder


@shipped
class Modern(Decoder):
    def __init__(
        self,
        vocab_size: int,
        context_length: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_kv_heads: int = 2,
        rope: bool = True,
        swiglu: bool = True,
        rmsnorm: bool = True,
        qk_norm: bool = True,
        z_loss: float = 1e-4,
        tie_weights: bool = True,
        ffn_hidden: int | None = None,  # default: 8d/3 for SwiGLU, 4d for GELU
    ) -> None:
        norm = RMSNorm() if rmsnorm else LayerNorm(bias=False)
        super().__init__(
            vocab_size=vocab_size, context_length=context_length, d_model=d_model,
            n_layers=n_layers,
            block=Block(
                norm=norm,
                attn=Attention(n_heads=n_heads, n_kv_heads=n_kv_heads,
                               pos=RoPE() if rope else None, qk_norm=qk_norm),
                mlp=SwiGLU(hidden=ffn_hidden) if swiglu else GELUMLP(hidden=ffn_hidden)),
            final_norm=norm, pos_emb=None if rope else LearnedPosition(),
            tie_weights=tie_weights, z_loss=z_loss)
