import nanoscope.blocks as nb


def _unused_helper():
    return 1


class SlidingGlobal(nb.Decoder):
    def __init__(self, vocab_size: int, context_length: int = 512, window: int = 64):
        super().__init__(
            vocab_size=vocab_size, context_length=context_length, d_model=256, n_layers=6,
            pattern=[
                nb.Block(norm=nb.RMSNorm(), attn=nb.Attention(n_heads=4, window=window,
                                                              pos=nb.RoPE()), mlp=nb.SwiGLU()),
                nb.Block(norm=nb.RMSNorm(), attn=nb.Attention(n_heads=4, pos=nb.RoPE()),
                         mlp=nb.SwiGLU()),
            ],
            final_norm=nb.RMSNorm(),
        )
