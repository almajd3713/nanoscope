"""A transformer layer: attention and an MLP, each wrapped in a residual and a norm.

Submodule names follow the shipped Modern model: `norm1`, `attn`, `norm2`, `mlp`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from nanoscope.blocks.embedding import TokenEmbedding
from nanoscope.blocks.head import Head
from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule, BlockSpec, build_option

ORDERS = ("pre", "post")


@block("structure", "composite", reference="residual", features=("post_norm",))
class Block(BlockModule):
    """pre:  x = x + attn(norm1(x)); x = x + mlp(norm2(x))
    post: x = norm1(x + attn(x));   x = norm2(x + mlp(x))

    `norm` is one spec, built twice so the two norms have their own weights."""

    def __init__(self, d_model: int, context_length: int, norm: BlockSpec, attn: BlockSpec,
                 mlp: BlockSpec, order: str = "pre") -> None:
        super().__init__()
        if order not in ORDERS:
            raise ValueError(f"Block order must be one of {', '.join(ORDERS)}, got {order!r}")
        self.order = order
        self.norm1 = build_option(norm, d_model, context_length)
        self.attn = build_option(attn, d_model, context_length)
        self.norm2 = build_option(norm, d_model, context_length)
        self.mlp = build_option(mlp, d_model, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "pre":
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        x = self.norm1(x + self.attn(x))
        return self.norm2(x + self.mlp(x))

    def flops_per_token(self, context_length: int) -> int:
        return sum(m.flops_per_token(context_length)
                   for m in (self.norm1, self.attn, self.norm2, self.mlp))


@block("structure", "composite", features=("tie_weights", "z_loss"))
class Decoder(nn.Module):
    """Token embedding, `n_layers` copies of `block`, a final norm and the output head.

    `pattern=[a, b]` instead of `block` repeats a list of block specs through the layers
    (a, b, a, b, ...), e.g. sliding-window and global attention layers.
    `pos_emb` (e.g. LearnedPosition()) is added after the token embedding; leave it out when
    attention carries the positions (RoPE). Like every model it returns logits, or
    `(logits, aux_loss)` when `z_loss` is set: the z-loss keeps the softmax normaliser near 1.
    Matrices start at N(0, 0.02), and each layer's output projections at
    0.02 / sqrt(2 * n_layers) so the residual stream does not grow with depth.
    """

    def __init__(self, vocab_size: int, context_length: int, d_model: int, n_layers: int,
                 block: BlockSpec | None = None, final_norm: BlockSpec | None = None,
                 pos_emb: BlockSpec | None = None, tie_weights: bool = True,
                 z_loss: float = 0.0, pattern: list[BlockSpec] | None = None) -> None:
        super().__init__()
        if (block is None) == (pattern is None) or pattern == []:
            raise TypeError("Decoder needs exactly one of block (every layer the same) or "
                            "pattern (a non-empty list of blocks repeated through the layers)")
        layers = [block] if pattern is None else list(pattern)
        self.context_length, self.d_model, self.z_loss = context_length, d_model, z_loss
        self.tok_emb = TokenEmbedding(vocab_size=vocab_size).build(d_model, context_length)
        self.pos_emb = build_option(pos_emb, d_model, context_length)
        self.blocks = nn.ModuleList(
            build_option(layers[i % len(layers)], d_model, context_length)
            for i in range(n_layers))
        self.norm = (nn.Identity() if final_norm is None
                     else build_option(final_norm, d_model, context_length))
        self.head = Head(vocab_size=vocab_size).build(d_model, context_length)
        if tie_weights:
            self.head.weight = self.tok_emb.weight
        for name, p in self.named_parameters():
            if p.ndim == 2:
                std = 0.02 / math.sqrt(2 * n_layers) if name.endswith("proj.weight") else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        T = idx.size(1)
        assert T <= self.context_length, f"sequence of {T} tokens > context_length"  # noqa: SIM300
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        for layer in self.blocks:
            x = layer(x)
        logits = self.head(self.norm(x))
        if not self.z_loss:
            return logits
        z = torch.logsumexp(logits.float(), dim=-1)
        return logits, self.z_loss * z.pow(2).mean()

    def flops_per_token(self, context_length: int) -> int:
        """Training FLOPs per token: the sum of every block's own formula."""
        parts = [self.tok_emb, self.pos_emb, *self.blocks, self.norm, self.head]
        return sum(m.flops_per_token(context_length)
                   for m in parts if hasattr(m, "flops_per_token"))
