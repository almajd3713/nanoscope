"""Efficiency 4: the roofline.   Check: nanoscope learn check efficiency/04-roofline"""


def mfu(tflops, peak_tflops):
    """Model FLOPs utilisation: the share of the chip's peak that training achieves."""
    # TODO
    raise NotImplementedError


def arithmetic_intensity(flops, bytes_moved):
    """FLOPs done per byte read from or written to memory."""
    # TODO
    raise NotImplementedError


def attainable_tflops(intensity, peak_tflops, bandwidth_gbs):
    """The roofline: memory can feed `intensity * bandwidth` FLOP/s; the chip caps at its peak.
    Bandwidth is in GB/s, the result in TFLOP/s."""
    # TODO
    raise NotImplementedError


def memory_bound(intensity, peak_tflops, bandwidth_gbs):
    """True when memory, not arithmetic, is what limits the kernel."""
    # TODO
    raise NotImplementedError
