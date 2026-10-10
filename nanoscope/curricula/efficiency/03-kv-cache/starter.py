"""Efficiency 3: the KV cache.   Check: nanoscope learn check efficiency/03-kv-cache"""
import math


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_value):
    """Memory of the cache: keys AND values, for every layer, kv head and position."""
    # TODO
    raise NotImplementedError


def tokens_processed(prompt_len, new_tokens, cached):
    """Token positions pushed through the layers to generate `new_tokens` tokens after a prompt of
    `prompt_len`. Cached: the prompt once, then one position per further token. Not cached: every
    step runs the whole sequence so far again."""
    # TODO
    raise NotImplementedError


def max_context(memory_gb, weights_gb, n_layers, n_kv_heads, head_dim, bytes_per_value):
    """The longest sequence (batch 1) whose cache fits in the memory left after the weights.
    1 GB is 1e9 bytes. Round down to a whole number of tokens."""
    # TODO
    raise NotImplementedError


def cached_step(q, new_k, new_v, cache_k, cache_v):
    """One decoding step of one attention head, as a list of numbers.

    `cache_k` and `cache_v` are lists of the earlier positions' key and value vectors (empty on the
    first step). The new token's own key and value join them, then `q` attends over all of them:
    softmax of q . k / sqrt(d), and the weighted sum of the values."""
    # TODO
    raise NotImplementedError
