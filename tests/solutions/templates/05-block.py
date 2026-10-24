from nanoscope.blocks import GELUMLP, Attention, BlockTemplate, LayerNorm


class MyBlock(BlockTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(
            d_model, context_length,
            norm1=LayerNorm(), attn=Attention(n_heads=4), norm2=LayerNorm(), mlp=GELUMLP(),
        )
