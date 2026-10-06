"""Modern lesson 2: RoPE.   Check with: nanoscope learn check modern-block/02-rope"""

import torch
import torch.nn as nn


class MyRoPE(nn.Module):
    def __init__(self, d_model, context_length, base=10000.0):
        super().__init__()
        # d_model here is the head size D.  For each position m (0 .. context_length - 1)
        # and pair i (0 .. D/2 - 1) the angle is  m * base ** (-2 i / D).
        # TODO: compute the angles, shape (context_length, D/2), then register_buffer
        #       "cos" and "sin" (persistent=False) so they move with the module.
        #   theta = base ** (-torch.arange(0, d_model, 2).float() / d_model)
        #   angles = torch.outer(torch.arange(context_length).float(), theta)

    def forward(self, x):
        # x: (batch, heads, time, D).  Split the last dimension in halves, x1 and x2
        # (x.chunk(2, dim=-1)), take cos and sin for the first `time` positions and rotate:
        #   x1 * cos - x2 * sin,   x1 * sin + x2 * cos     then concatenate the halves.
        raise NotImplementedError("rotate each pair by its position's angle")
