"""A model with comments, odd spacing and trailing commas in awkward places."""
from nanoscope.blocks import (  # the palette
    Attention,
    Block,
    Decoder,
    RMSNorm,
    SwiGLU,
)


class Odd(Decoder):
    """Docstring."""

    def __init__(
        self,
        vocab_size: int,   # from the tokenizer
        context_length: int = 256,
    ):
        super().__init__(
            vocab_size,context_length,   # positional, squeezed
            d_model = 96 ,  # spaces around =
            n_layers=3,
            block = Block(
                norm=RMSNorm( ),
                # the attention
                attn=Attention(n_heads=6,
                               n_kv_heads=3),
                mlp=SwiGLU(hidden=256,),
            ),
            final_norm=RMSNorm(),
        )
