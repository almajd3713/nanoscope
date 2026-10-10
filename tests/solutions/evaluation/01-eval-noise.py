import math


def standard_error(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    return sd / math.sqrt(n)


def chunks_needed(sd, half_width):
    return math.ceil((1.96 * sd / half_width) ** 2)
