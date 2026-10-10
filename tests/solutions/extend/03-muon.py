from nanoscope.models import Modern
from nanoscope.optim.muon import Muon

LR_SCALE = 2.0
NOTES = """
Muon's update is scaled differently from AdamW's, so the preset's rate (tuned for AdamW) is not
its rate. I tried 1x, 2x and 3x; 1x was within noise and 2x was clearly ahead. AdamW got the
preset's rate only, so this comparison is tilted toward Muon.
"""


class Tiny(Modern):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2):
        super().__init__(vocab_size, context_length, d_model, n_layers)


def make_muon(param_groups, preset):
    return Muon(param_groups, lr=preset.learning_rate * LR_SCALE, betas=preset.betas,
                eps=preset.eps)
