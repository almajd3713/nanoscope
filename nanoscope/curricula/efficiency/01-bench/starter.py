"""Efficiency 1: bench.   Check: nanoscope learn check efficiency/01-bench

Three small functions that `nanoscope bench` is built from.
"""


def tokens_per_second(step_ms, batch_size, context_length):
    """One step trains on batch_size sequences of context_length tokens."""
    # TODO
    raise NotImplementedError


def gpu_busy(kernel_ms, step_ms):
    """The fraction of a step the GPU spends running kernels. It cannot exceed 1."""
    # TODO
    raise NotImplementedError


def cpu_bound(gpu_busy_fraction, threshold=0.7):
    """True when the GPU waits for Python for more than 30% of each step."""
    # TODO
    raise NotImplementedError
