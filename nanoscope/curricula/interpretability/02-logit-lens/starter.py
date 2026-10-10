"""Interpretability 2: the logit lens.   Check: nanoscope learn check interpretability/02-logit-lens"""


def top_token(logits):
    """Index of the largest logit."""
    # TODO
    raise NotImplementedError


def rank_of(logits, target):
    """0 if `target` has the largest logit, 1 if it is second, and so on."""
    # TODO
    raise NotImplementedError


def first_settled_layer(layer_logits, target):
    """`layer_logits[k]` is the lens reading after layer k. Return the first k whose top token is
    `target` and stays so at every later layer, or -1."""
    # TODO
    raise NotImplementedError
