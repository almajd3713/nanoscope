"""Modern lesson 6: z-loss.   Check with: nanoscope learn check modern-block/06-z-loss"""

import torch
import torch.nn as nn


class ZLoss(nn.Module):
    def __init__(self, coefficient=1e-4):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, logits):
        # logits: (batch, time, vocab).  Return one number:
        #   coefficient * mean( logsumexp(logits, dim=-1) ** 2 )
        raise NotImplementedError("penalise the squared log of the softmax normaliser")
