import torch
import torch.nn as nn

from nanoscope.blocks import register_block


def squared_relu_mlp(x, w_up, w_down):
    h = torch.relu(x @ w_up)
    return (h * h) @ w_down


@register_block(reference=squared_relu_mlp, family="mlp")
class SquaredReLUMLP(nn.Module):
    def __init__(self, d_model, context_length, hidden=None):
        super().__init__()
        hidden = hidden or 4 * d_model
        self.w_up = nn.Parameter(torch.randn(d_model, hidden) * 0.02)
        self.w_down = nn.Parameter(torch.randn(hidden, d_model) * 0.02)

    def forward(self, x):
        h = torch.relu(x @ self.w_up)
        return (h * h) @ self.w_down
