def tokens_per_second(step_ms, batch_size, context_length):
    return batch_size * context_length / (step_ms / 1000)


def gpu_busy(kernel_ms, step_ms):
    return min(kernel_ms / step_ms, 1.0)


def cpu_bound(gpu_busy_fraction, threshold=0.7):
    return gpu_busy_fraction < threshold
