"""Muon: momentum SGD whose update for each weight matrix is orthogonalized.

Matrices (2-D parameters) get the orthogonalized momentum; everything else (norm gains,
biases) gets plain AdamW, as in the original recipe. Embedding and head matrices are 2-D too,
so they take the Muon path here; split them off with your own factory if you want the usual
AdamW treatment for them.

    run(Modern, "tinystories-5min", optimizer=nanoscope.optim.muon)
"""

from __future__ import annotations

from typing import Any

import torch

from nanoscope.presets import Preset

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz coefficients (Jordan et al.)


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Push g's singular values toward 1 with a few Newton-Schulz iterations."""
    a, b, c = NS_COEFFS
    x = g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        m = x @ x.T
        x = a * x + (b * m + c * m @ m) @ x
    return (x.T if transposed else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups: list[dict[str, Any]], lr: float, momentum: float = 0.95,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, betas=betas, eps=eps,
                        weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if wd:
                    p.mul_(1 - lr * wd)
                if p.ndim == 2:
                    buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                    buf.mul_(group["momentum"]).add_(p.grad)
                    update = orthogonalize(p.grad.add(buf, alpha=group["momentum"]),
                                           group["ns_steps"])
                    # scale so the update's RMS is comparable to AdamW's
                    p.add_(update, alpha=-lr * 0.2 * max(p.size(0), p.size(1)) ** 0.5)
                else:
                    beta1, beta2 = group["betas"]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    state["exp_avg"].lerp_(p.grad, 1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    denom = (state["exp_avg_sq"] / (1 - beta2**t)).sqrt().add_(group["eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-lr / (1 - beta1**t))
        return loss


def muon(param_groups: list[dict[str, Any]], preset: Preset) -> torch.optim.Optimizer:
    """Optimizer factory: Muon for matrices, AdamW for the rest, at the preset's learning rate."""
    return Muon(param_groups, lr=preset.learning_rate, betas=preset.betas, eps=preset.eps)
