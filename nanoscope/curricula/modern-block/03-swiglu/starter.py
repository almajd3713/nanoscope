"""Modern lesson 3: SwiGLU.   Check with: nanoscope learn check modern-block/03-swiglu"""

import torch.nn as nn
import torch.nn.functional as F


class MySwiGLU(nn.Module):
    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or 8 * ((8 * d_model // 3 + 7) // 8)
        # TODO: three linear layers, no bias, named exactly:
        #   self.w1    d_model -> hidden
        #   self.w3    d_model -> hidden
        #   self.proj  hidden -> d_model

    def forward(self, x):
        # proj( silu(w1(x)) * w3(x) )      F.silu is x * sigmoid(x)
        raise NotImplementedError("gate one branch with the other")
