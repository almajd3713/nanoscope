"""Extend 3: Muon versus AdamW.   Check: nanoscope learn check extend/03-muon

The experiment: the same small model trained with AdamW (the default) and with Muon, 200 steps.
Muon takes its own learning rate, so the comparison is only fair once Muon has a sensible one.
"""
from nanoscope.models import Modern
from nanoscope.optim.muon import Muon

# Muon's learning rate as a multiple of the preset's (which is tuned for AdamW).
LR_SCALE = 1.0  # TODO: choose it, then say why in NOTES below
NOTES = """
TODO: how did you choose LR_SCALE? What would make the comparison unfair to AdamW?
"""


class Tiny(Modern):
    """A small modern model: 64 wide, 2 layers."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        super().__init__(vocab_size, context_length, d_model, n_layers)


def make_muon(param_groups, preset):
    """An optimizer factory: Muon on matrices, AdamW on the rest, at LR_SCALE x the preset's rate."""
    return Muon(param_groups, lr=preset.learning_rate * LR_SCALE, betas=preset.betas,
                eps=preset.eps)
