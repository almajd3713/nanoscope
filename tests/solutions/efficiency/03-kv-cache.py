import math


def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per_value):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_value


def tokens_processed(prompt_len, new_tokens, cached):
    if cached:
        return prompt_len + new_tokens - 1
    return sum(prompt_len + j for j in range(new_tokens))


def max_context(memory_gb, weights_gb, n_layers, n_kv_heads, head_dim, bytes_per_value):
    per_token = 2 * n_layers * n_kv_heads * head_dim * bytes_per_value
    return int((memory_gb - weights_gb) * 1e9 // per_token)


def cached_step(q, new_k, new_v, cache_k, cache_v):
    keys, values = cache_k + [new_k], cache_v + [new_v]
    scores = [sum(a * b for a, b in zip(q, k, strict=True)) / math.sqrt(len(q)) for k in keys]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = sum(weights)
    return [sum(w / total * v[d] for w, v in zip(weights, values, strict=True))
            for d in range(len(new_v))]
