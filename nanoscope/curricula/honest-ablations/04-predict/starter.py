"""Honest ablations 4: predict, then run.   Check: nanoscope learn check honest-ablations/04-predict

STOP before you run any check. First write your prediction (see lesson.md):

    nanoscope learn predict honest-ablations/04-predict \\
        --verdict <better|worse|within noise> --low <number> --high <number> --note "why"

The command records the time and a fingerprint of what you wrote. A prediction written after
the experiment ran is refused: it would only be a description of the result.

The experiment: take RoPE away from the modern model. Is the model with RoPE better, and by how
much on val_bpb (negative means the first model is better)?
"""

from nanoscope.models import Modern


class WithRoPE(Modern):
    """Modern as shipped: rotary position embeddings, 64 wide, 2 layers."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        super().__init__(vocab_size, context_length, d_model, n_layers, rope=True)


class NoRoPE(Modern):
    """The same model with the rotary position embeddings removed."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        # TODO: remove RoPE. Until you do, this is the same model as WithRoPE and the
        # experiment compares a model with itself.
        super().__init__(vocab_size, context_length, d_model, n_layers, rope=True)
