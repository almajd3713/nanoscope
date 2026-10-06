from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU

import mylib
from mylib.layers import Gated


class WithCustom(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 128):
        super().__init__(
            vocab_size, context_length, d_model=64, n_layers=2,
            block=Block(norm=RMSNorm(), attn=Gated(n_heads=2), mlp=mylib.FancyMLP(hidden=256)),
            final_norm=RMSNorm(),
            z_loss=2 * 1e-4 + 0,
        )
