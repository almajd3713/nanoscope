from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, SwiGLU


class Fine(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size, 64, d_model=32, n_layers=1,
                         block=Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU()))


class Dynamic(Decoder):
    def __init__(self, vocab_size: int, wide: bool = False):
        d_model = 256 if wide else 64
        super().__init__(vocab_size, 64, d_model=d_model, n_layers=1,
                         block=Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU()))


class Overrides(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(vocab_size, 64, d_model=32, n_layers=1,
                         block=Block(norm=RMSNorm(), attn=Attention(n_heads=2), mlp=SwiGLU()))

    def forward(self, idx):
        return super().forward(idx)
