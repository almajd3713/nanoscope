"""Mixture of experts: a router sends each token to its `top_k` best experts and mixes their
outputs. Only those experts run for the token, so the parameters grow with `experts` while the
work per token grows with `top_k`.

Each expert is a SwiGLU MLP (`w1`, `w3`, `proj`). The router is `gate`. A load-balancing term
keeps the router from sending everything to one expert; the block leaves it in `aux_loss_value`
after every forward pass, and `Decoder` adds it to its output as `(logits, aux)`, the same way
it returns a z-loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoscope.blocks.registry import block
from nanoscope.blocks.spec import BlockModule


class Expert(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.silu(self.w1(x)) * self.w3(x))


@block("mlp", "primitive", reference="naive_moe", features=("moe",))
class MoE(BlockModule):
    """`experts` SwiGLU experts, `top_k` of them per token. `aux_loss` weights the
    load-balancing term E * sum_e (share of assignments to e) * (mean router probability of e),
    which is 1 when the load is even and grows as it skews; 0 turns the term off.

    Primitive tier (never locked) until a lesson unlocks it as `block:MoE`."""

    aux_loss_value: torch.Tensor | None

    def __init__(self, d_model: int, context_length: int, experts: int = 4, top_k: int = 2,
                 hidden: int | None = None, aux_loss: float = 0.01) -> None:
        super().__init__()
        if experts < 1 or not 1 <= top_k <= experts:
            raise ValueError(f"MoE needs 1 <= top_k <= experts, got top_k={top_k}, "
                             f"experts={experts}")
        if aux_loss < 0:
            raise ValueError(f"MoE aux_loss must be at least 0, got {aux_loss}")
        hidden = hidden or 8 * ((8 * d_model // 3 + 7) // 8)
        self.experts_count, self.top_k, self.aux_loss = experts, top_k, aux_loss
        self.gate = nn.Linear(d_model, experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, hidden) for _ in range(experts))
        self.aux_loss_value = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])  # one row per token
        probs = self.gate(flat).float().softmax(-1)  # (N, E)
        weight, chosen = probs.topk(self.top_k, dim=-1)  # (N, k)
        weight = (weight / weight.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(flat)
        if flat.device.type == "meta":
            # shape tracing (`describe`) has no data to pick rows by: run every expert, keep
            # each token's chosen weights
            dense = torch.zeros_like(probs, dtype=flat.dtype).scatter(1, chosen, weight)
            for e, expert in enumerate(self.experts):
                out = out + expert(flat) * dense[:, e : e + 1]
        else:
            for e, expert in enumerate(self.experts):
                rows, slot = (chosen == e).nonzero(as_tuple=True)
                if rows.numel():
                    out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot].unsqueeze(-1))
        if self.aux_loss and self.training:
            share = F.one_hot(chosen, self.experts_count).float().sum(1).mean(0) / self.top_k
            self.aux_loss_value = self.aux_loss * self.experts_count * (share * probs.mean(0)).sum()
        else:
            self.aux_loss_value = None
        return out.reshape(shape)

    def flops_per_token(self, context_length: int) -> int:
        """Training FLOPs per token count the parameters a token uses: the router and `top_k`
        experts."""
        expert = sum(p.numel() for p in self.experts[0].parameters())
        return 6 * (self.gate.weight.numel() + self.top_k * expert)
