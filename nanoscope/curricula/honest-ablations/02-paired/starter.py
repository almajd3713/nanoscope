"""Honest ablations 2: paired and unpaired.   Check: nanoscope learn check honest-ablations/02-paired

Two functions to write, and a model that is already written for you. Both functions take the
final val_bpb of each seed for model A and for model B (lists of floats) and return
(mean, low, high): the average of A minus B and its 95% confidence interval.
"""

from scipy.stats import t

from nanoscope.models import GPT2


def paired_ci(a, b):
    """A[i] and B[i] came from the same seed i. Take the difference seed by seed first, then
    put an interval around the average of those differences (n - 1 degrees of freedom)."""
    # TODO: differences, their mean and standard deviation, then
    #       margin = t.ppf(0.975, df) * sd / sqrt(n)
    raise NotImplementedError


def unpaired_ci(a, b):
    """A and B have no seed in common. Compare the two averages and combine their spreads
    (Welch: the groups may have different spreads, so the degrees of freedom are not n - 2)."""
    # TODO: va = var(a) / len(a),  vb = var(b) / len(b)   (sample variance, n - 1)
    #       df = (va + vb)**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    #       margin = t.ppf(0.975, df) * sqrt(va + vb)
    raise NotImplementedError


class TinyGPT(GPT2):
    """A small GPT-2, 64 wide and 2 layers. The check trains it twice: 205 steps and 200
    steps. Five more steps is a small, real improvement to look for. Nothing to change here."""

    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__(vocab_size, context_length, d_model, n_layers, n_heads)
