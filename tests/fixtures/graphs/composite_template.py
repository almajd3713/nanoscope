from nanoscope.blocks import Attention, Block, Composite, Decoder, RMSNorm, SwiGLU


class PreNormAttention(Composite):
    """A lesson template: fill the slots."""

    SLOTS = ("norm", "attn")

    def forward(self, x):
        return x + self.attn(self.norm(x))


class UsesTemplate(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(
            vocab_size, 64, d_model=32, n_layers=2,
            block=PreNormAttention(norm=RMSNorm(), attn=Attention(n_heads=2)),
        )
