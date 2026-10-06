from nanoscope.blocks import Attention, Block, Decoder, GELUMLP, LayerNorm, LearnedPosition


class TinyGPT(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256, n_layers: int = 4):
        super().__init__(
            vocab_size, context_length, d_model=128, n_layers=n_layers,
            block=Block(norm=LayerNorm(), attn=Attention(n_heads=4, bias=True),
                        mlp=GELUMLP(bias=True)),
            final_norm=LayerNorm(), pos_emb=LearnedPosition(), tie_weights=True,
        )
