from nanoscope.blocks import AttentionTemplate, CausalMask, Linear, ScaledDotScores, Softmax


class OneHead(AttentionTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(
            d_model, context_length,
            q=Linear(), k=Linear(), v=Linear(),
            scores=ScaledDotScores(), mask=None, normalize=Softmax(),
            mix=None, out=Linear(),
        )
