"""Modern lesson 7: assemble the modern block.   Check: nanoscope learn check modern-block/07-assemble

This is a GPT-2-style model built from library blocks. Swap in the parts you built.
"""

from nanoscope.blocks import Attention, Block, Decoder, GELUMLP, LayerNorm, LearnedPosition


class MyModern(Decoder):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 128,
                 n_layers: int = 4):
        super().__init__(
            vocab_size, context_length, d_model=d_model, n_layers=n_layers,
            block=Block(
                norm=LayerNorm(),            # TODO: RMSNorm()
                attn=Attention(n_heads=4),   # TODO: n_kv_heads=2, pos=RoPE(), qk_norm=True
                mlp=GELUMLP(),               # TODO: SwiGLU()
            ),
            final_norm=LayerNorm(),          # TODO: RMSNorm()
            pos_emb=LearnedPosition(),       # TODO: remove this line once attention uses RoPE
            # TODO: z_loss=1e-4
        )
