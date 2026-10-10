def speedup(eager_ms, compiled_ms):
    return eager_ms / compiled_ms


def break_even_steps(compile_seconds, eager_ms, compiled_ms):
    return compile_seconds * 1000 / (eager_ms - compiled_ms)


def worth_compiling(steps, compile_seconds, eager_ms, compiled_ms):
    return compile_seconds + steps * compiled_ms / 1000 < steps * eager_ms / 1000
