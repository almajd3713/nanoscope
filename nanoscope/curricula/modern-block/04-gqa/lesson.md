When a model generates text it keeps every past token's keys and values in memory (the
**KV cache**), and that cache is what limits how long a context fits. **Grouped-query
attention** shrinks it: keep `n_heads` query heads but only `n_kv_heads` key/value heads, so
each K/V head serves a group of `n_heads / n_kv_heads` queries. `n_kv_heads = n_heads` is plain
multi-head attention; `n_kv_heads = 1` is multi-query attention.

## Surface

Start from your multi-head attention (given in `starter.py`) and change two things:

1. `k` and `v` project to `n_kv_heads * head_dim` instead of `d_model`.
2. After splitting heads, **repeat** each K/V head for its group so the shapes match the
   queries: `k.repeat_interleave(n_heads // n_kv_heads, dim=1)` (query head `h` reads K/V head
   `h // group`).

Everything else is the same. Write `MyGQA(d_model, context_length, n_heads=4, n_kv_heads=2)`
with layers `q`, `k`, `v`, `out`. Passing **unlocks the GQA feature** (`n_kv_heads < n_heads`
in your own `Attention`).

## Deep

Count the K/V cache for a 2048-token context with 32 heads of size 128, at `n_kv_heads` =
32, 8, 1. That ratio is why almost every large model uses GQA, and why quality barely moves.

## Reading

- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
  Checkpoints" (2023).
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019).
