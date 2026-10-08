from nanoscope.blocks import (
    AttentionTemplate,
    CausalMask,
    Linear,
    ScaledDotScores,
    Softmax,
    WeightedSum,
)


class OneHead(AttentionTemplate):
    def __init__(self, d_model, context_length):
        super().__init__(
            d_model, context_length,
            q=Linear(), k=Linear(), v=Linear(),
            scores=ScaledDotScores(), mask=CausalMask(), normalize=Softmax(),
            mix=WeightedSum(), out=Linear(),
        )
