"""Efficiency 2: torch.compile.   Check: nanoscope learn check efficiency/02-compile"""


def speedup(eager_ms, compiled_ms):
    """How many times faster a compiled step is."""
    # TODO
    raise NotImplementedError


def break_even_steps(compile_seconds, eager_ms, compiled_ms):
    """Steps after which the time saved per step has paid back the compile time."""
    # TODO
    raise NotImplementedError


def worth_compiling(steps, compile_seconds, eager_ms, compiled_ms):
    """True when a run of `steps` steps finishes sooner with compile, compile time included."""
    # TODO
    raise NotImplementedError
