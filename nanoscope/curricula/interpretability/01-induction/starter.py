"""Interpretability 1: induction heads.   Check: nanoscope learn check interpretability/01-induction"""


def induction_score(attn, half):
    """Mean attention from each position in the second copy to the token AFTER its earlier copy.

    `attn[i]` is the list of weights position i puts on positions 0..i (a causal row, so row i
    has i + 1 numbers). The sequence is `half` tokens followed by the same `half` tokens again."""
    # TODO
    raise NotImplementedError


def previous_token_score(attn):
    """Mean attention from each position to the position just before it."""
    # TODO
    raise NotImplementedError


def first_step_above(steps, scores, threshold):
    """The first step whose score is at or above `threshold` and stays there at every later
    step. Return -1 if there is none."""
    # TODO
    raise NotImplementedError
