"""Extend 5: speedrun.   Check: nanoscope learn check extend/05-speedrun

The baseline is the shipped GPT-2 at 64 wide and 2 layers, trained for 100 steps. Make `Speedrun`
reach a lower val_bpb in the same 100 steps, at about the same size (within 10% of its
non-embedding parameters), and do it so that three seeds agree: the check uses `compare()`, so a
lucky seed does not pass.

A model that learns faster early is the point: what in a modern block helps in the first hundred
steps? Which of the things you built in modern-block/ and extend/ are worth their parameters?
"""
from nanoscope.models import Modern


class Speedrun(Modern):
    """Modern at the baseline's width and depth. This is your starting point, not your answer."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        # TODO: change something, check, and look at the interval before the next change.
        super().__init__(vocab_size, context_length, d_model, n_layers)
