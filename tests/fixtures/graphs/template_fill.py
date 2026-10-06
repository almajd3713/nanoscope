from nanoscope.blocks import AttentionTemplate, Block, BlockTemplate, Decoder, RMSNorm, SwiGLU
from nanoscope.blocks import CausalMask, Linear, ScaledDotScores, Softmax, WeightedSum


class FromPrimitives(Decoder):
    def __init__(self, vocab_size: int):
        super().__init__(
            vocab_size, 16, d_model=32, n_layers=2,
            block=BlockTemplate(
                norm1=RMSNorm(),
                attn=AttentionTemplate(
                    q=Linear(), k=Linear(), v=Linear(),
                    scores=ScaledDotScores(), mask=None, normalize=Softmax(),
                    mix=WeightedSum(), out=Linear(),
                ),
                norm2=RMSNorm(),
                mlp=SwiGLU(),
            ),
        )
