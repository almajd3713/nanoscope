import math


def training_flops(n_params, tokens):
    return 6 * n_params * tokens


def tokens_for_budget(flops, n_params):
    return flops / (6 * n_params)


def chinchilla_split(flops, tokens_per_param=20):
    n = math.sqrt(flops / (6 * tokens_per_param))
    return n, tokens_per_param * n
