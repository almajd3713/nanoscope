import math

from scipy.stats import t

from nanoscope.models import GPT2


def paired_ci(a, b):
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    margin = t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - margin, mean + margin


def unpaired_ci(a, b):
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1) / na
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1) / nb
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    margin = t.ppf(0.975, df) * math.sqrt(va + vb)
    diff = ma - mb
    return diff, diff - margin, diff + margin


class TinyGPT(GPT2):
    def __init__(self, vocab_size: int, context_length: int = 256, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__(vocab_size, context_length, d_model, n_layers, n_heads)
