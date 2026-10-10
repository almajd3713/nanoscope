def mfu(tflops, peak_tflops):
    return tflops / peak_tflops


def arithmetic_intensity(flops, bytes_moved):
    return flops / bytes_moved


def attainable_tflops(intensity, peak_tflops, bandwidth_gbs):
    return min(peak_tflops, intensity * bandwidth_gbs / 1000)


def memory_bound(intensity, peak_tflops, bandwidth_gbs):
    return intensity * bandwidth_gbs / 1000 < peak_tflops
