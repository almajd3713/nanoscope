"""Honest ablations 1: the same model against itself.   Check: nanoscope learn check honest-ablations/01-seed-noise

An A/A test: two models that are the same on purpose. Any difference between them is noise,
so the size of that difference tells you how big a real difference has to be before you can
believe it.
"""

from nanoscope.models import GPT2


class MyGPT(GPT2):
    """GPT-2 again. The check compares it with a shipped GPT2 that is 64 wide and 2 layers deep."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 32,
                 n_layers: int = 1, n_heads: int = 4):
        # TODO: this is 32 wide and 1 layer. `same-size` fails until it is the same size as
        # that GPT2, because a model that is not the same is no longer an A/A test. Change the
        # defaults above.
        super().__init__(vocab_size, context_length, d_model, n_layers, n_heads)
