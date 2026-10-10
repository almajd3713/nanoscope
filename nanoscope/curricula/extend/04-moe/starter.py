"""Extend 4: a mixture of experts.   Check: nanoscope learn check extend/04-moe"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A SwiGLU MLP, as in modern-block/03-swiglu: proj(silu(w1(x)) * w3(x)). No biases."""

    def __init__(self, d_model, hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w3(x))


class MyMoE(nn.Module):
    def __init__(self, d_model, context_length, experts=4, top_k=2, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d_model
        self.top_k = top_k
        self.gate = nn.Linear(d_model, experts, bias=False)  # the router
        self.experts = nn.ModuleList(Expert(d_model, hidden) for _ in range(experts))

    def forward(self, x):
        # TODO, in four steps:
        # 1. flatten x to one row per token, and get the router probabilities (softmax over experts)
        # 2. keep each token's top_k probabilities (torch.topk) and rescale them to sum to 1
        # 3. for each expert, run it only on the tokens that chose it, and add its output to
        #    those tokens' rows, multiplied by that token's weight for the expert
        # 4. reshape back to x's shape
        raise NotImplementedError
