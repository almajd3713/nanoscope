import torch
import torch.nn as nn


class ZLoss(nn.Module):
    def __init__(self, coefficient=1e-4):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, logits):
        return self.coefficient * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
