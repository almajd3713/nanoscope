from nanoscope.blocks import Attention, Block, Decoder, RMSNorm, RoPE, SwiGLU


class MyLM(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256):
        super().__init__(
            vocab_size, context_length, d_model=128, n_layers=4,
            block=Block(norm=RMSNorm(), mlp=SwiGLU(hidden=344),
                        attn=Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)),
            final_norm=RMSNorm(),
            tie_weights=True, z_loss=1e-4,
        )
