To write token n+1, a transformer attends over the keys and values of tokens 1..n. Without a
cache, nanoscope's `generate` recomputes all of them at every step: the work per token grows with
the sequence, and a 200-token sample costs far more than 200 times one token. A KV cache keeps the
keys and values already computed, so each step only computes the new token's.

## Surface

You will write four functions in `starter.py`, then measure the real thing.

- `kv_cache_bytes(...)`: what the cache costs in memory. Keys *and* values, every layer, every
  **kv** head (grouped-query attention from `modern-block/04-gqa` shrinks this: fewer kv heads,
  smaller cache), every position, every sequence in the batch.
- `tokens_processed(prompt_len, new_tokens, cached)`: the work, counted as token positions pushed
  through the layers. Compare the two cases for a 100-token prompt and 200 new tokens.
- `max_context(...)`: given a GPU's memory and the weights' size, how long a sequence can the cache
  hold? This, not compute, is what limits long contexts in practice.
- `cached_step(q, new_k, new_v, cache_k, cache_v)`: one decoding step of one attention head. It is
  ordinary attention for the last position, with the earlier keys and values taken from the cache
  instead of recomputed.

```bash
nanoscope learn check efficiency/03-kv-cache
```

### Using it

`nanoscope.train_loop.generate` has no cache. Time it yourself at several lengths and watch the time
per token grow:

```python
import time
from nanoscope.models import Modern
from nanoscope.tokenizer import ByteTokenizer
from nanoscope.train_loop import generate

tok = ByteTokenizer()
model = Modern(tok.vocab_size, 512, 128, 4).eval()
for n in (32, 64, 128, 256):
    t = time.time(); generate(model, tok, "a", max_new_tokens=n, context_length=512)
    print(n, f"{1000 * (time.time() - t) / n:.1f} ms per token")
```

Compare the growth with what `tokens_processed(1, n, False) / n` says it should be. A cache turns
that growth into a constant, at the price of `kv_cache_bytes`.

## Deep

1. **GQA.** Take a model with 8 query heads. What do 8, 4, 2 and 1 kv heads do to the cache for a 4096-token
   sequence? What do they cost in quality (`modern-block/04-gqa`)?
2. **Batching.** The cache grows with the batch. At what batch size does a 24 GB GPU running a
   7B-parameter model in 16-bit run out of cache before it runs out of compute?
3. **Positions.** With RoPE, the cached keys were rotated for their own positions. Why does that make
   the cache still valid for the next token?
4. **Build it.** Add a cache to a model you own: its `forward` takes and returns `(k, v)` per layer.
   Check it by generating the same tokens with and without the cache.

## Reading

- **B**: the four functions you wrote.
- **R**: [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et al., 2022), on what limits inference.
- **S**: [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245).
