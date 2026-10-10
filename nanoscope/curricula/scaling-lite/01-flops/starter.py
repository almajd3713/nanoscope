"""Scaling 1: FLOP budgets.   Check: nanoscope learn check scaling-lite/01-flops

Three small functions. The lesson text uses them on the models in this tool.
"""

import math


def training_flops(n_params, tokens):
    """About 6 FLOPs per parameter per token: 2 going forward, 4 for the backward pass."""
    # TODO
    raise NotImplementedError


def tokens_for_budget(flops, n_params):
    """How many tokens a model of n_params can train on within a budget of `flops`."""
    # TODO
    raise NotImplementedError


def chinchilla_split(flops, tokens_per_param=20):
    """Return (parameters, tokens) that spend `flops` with about 20 training tokens per
    parameter. Solve flops = 6 * N * (tokens_per_param * N) for N, then D = tokens_per_param * N."""
    # TODO
    _ = math
    raise NotImplementedError
