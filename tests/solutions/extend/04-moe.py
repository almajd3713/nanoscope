import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
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
        self.gate = nn.Linear(d_model, experts, bias=False)
        self.experts = nn.ModuleList(Expert(d_model, hidden) for _ in range(experts))

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = self.gate(flat).softmax(-1)
        weight, chosen = probs.topk(self.top_k, dim=-1)
        weight = weight / weight.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, expert(flat[rows]) * weight[rows, slot].unsqueeze(-1))
        return out.reshape(shape)
