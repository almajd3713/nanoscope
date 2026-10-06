"""Modern lesson 1: RMSNorm.   Check with: nanoscope learn check modern-block/01-rmsnorm"""

import torch
import torch.nn as nn


class MyRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # TODO: self.weight, a learned scale of d_model ones   (nn.Parameter(torch.ones(d_model)))

    def forward(self, x):
        # x / sqrt(mean(x ** 2 over the last dimension) + eps) * self.weight
        raise NotImplementedError("divide by the root mean square, then scale")
