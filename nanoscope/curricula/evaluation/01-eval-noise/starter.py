"""Evaluation 1: eval noise.   Check: nanoscope learn check evaluation/01-eval-noise

Two small functions. The lesson text shows how to use them on a real trained model.
"""

import math


def standard_error(values):
    """How far the average of `values` is likely to be from the true average: the sample
    standard deviation (divide by n - 1) over sqrt(n)."""
    # TODO
    raise NotImplementedError


def chunks_needed(sd, half_width):
    """How many measurements (each with spread `sd`) are needed for the 95% interval of their
    average to reach no more than `half_width` either side. A 95% interval is 1.96 standard
    errors. Return a whole number, rounded up."""
    # TODO: solve half_width = 1.96 * sd / sqrt(n) for n
    _ = math
    raise NotImplementedError
