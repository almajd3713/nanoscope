import torch.nn as nn
import torch.nn.functional as F


class MySwiGLU(nn.Module):
    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or 8 * ((8 * d_model // 3 + 7) // 8)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w3(x))
