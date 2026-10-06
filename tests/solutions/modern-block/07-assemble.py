from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU


class MyModern(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 128,
                 n_layers: int = 4):
        super().__init__(
            vocab_size, context_length, d_model=d_model, n_layers=n_layers,
            block=Block(
                norm=RMSNorm(),
                attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True),
                mlp=SwiGLU(),
            ),
            final_norm=RMSNorm(),
            z_loss=1e-4,
        )
