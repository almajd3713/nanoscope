"""Honest ablations 3: a fair fight.   Check: nanoscope learn check honest-ablations/03-matching

The question: does SwiGLU beat the GELU feed-forward layer? The control is the shipped Modern
with `swiglu=False` (a GELU layer 4 times as wide as the model, 64 wide and 2 layers deep).
MySwiGLU is the same model with SwiGLU instead. Make the comparison fair.
"""

from nanoscope.models import Modern


class MySwiGLU(Modern):
    """Modern with the SwiGLU feed-forward layer, 64 wide and 2 layers deep."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        # TODO: the hidden width is the knob. GELU uses 256 (4 x 64) with two weight matrices.
        # SwiGLU has three, so the same hidden width is a bigger model. Choose `ffn_hidden` so
        # that the non-embedding parameters (and the FLOPs) match the GELU control within 3%.
        # `nanoscope describe` and the check's message show both counts.
        super().__init__(vocab_size, context_length, d_model, n_layers, swiglu=True,
                         ffn_hidden=256)
