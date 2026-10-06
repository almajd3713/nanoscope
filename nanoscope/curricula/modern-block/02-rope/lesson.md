GPT-2 adds a learned vector for each position. **RoPE** (rotary position embedding) does
something more elegant: it *rotates* each query and key by an angle proportional to its
position. The dot product between a query at position `m` and a key at position `n` then
depends only on `m - n`, the distance, which is what language cares about.

## Surface

Treat the `D` numbers of a head vector as `D/2` pairs `(x[i], x[i + D/2])`: pair `i` is a point
in a plane. At position `m`, rotate pair `i` by the angle `m * theta_i`, where
`theta_i = base ** (-2i / D)` (so early pairs turn fast, late pairs slowly):

```
x1' = x1 * cos(m * theta) - x2 * sin(m * theta)
x2' = x1 * sin(m * theta) + x2 * cos(m * theta)
```

Write `MyRoPE(d_model, context_length, base=10000.0)` in `starter.py`. Here `d_model` is the
head size `D`. `forward(x)` takes `(batch, heads, time, D)` and returns the rotated tensor of
the same shape. Passing **unlocks `RoPE`** in `nanoscope.blocks`.

## Deep

Take two random vectors, rotate one to position 3 and the other to position 5, and take their
dot product. Now put them at 10 and 12. The same number comes out: only the distance counted.
Check this in a few lines, then explain why that makes the model work at positions it never
saw absolute embeddings for.

## Reading

- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
- EleutherAI blog, "Rotary Embeddings: A Relative Revolution".
