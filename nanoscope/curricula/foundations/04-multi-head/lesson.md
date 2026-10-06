One head learns one way of looking back (the previous word, the subject of the sentence,
a matching bracket). **Multi-head attention** runs several at once and lets each learn its own.
The trick is that it costs the same as one big head: split the vector into `n_heads` slices.

## Surface

With `d_model = 16` and `n_heads = 4`, each head works on `head_dim = 4` numbers. The code is
your single head with two extra moves:

1. After the `q`, `k`, `v` projections, **split heads**: reshape `(batch, time, d_model)` to
   `(batch, n_heads, time, head_dim)` with `view(B, T, n_heads, head_dim).transpose(1, 2)`.
2. Do the same scores, mask, softmax and mix as before. Matrix multiplication works on the last
   two dimensions, so all heads run in one go.
3. **Merge heads** back to `(batch, time, d_model)` with
   `transpose(1, 2).reshape(B, T, d_model)`, then apply `out`.

Name the layers `q`, `k`, `v`, `out` again. The check compares you with a reference that loops
over heads by hand. Passing it **unlocks `Attention`** in `nanoscope.blocks`: the library's
own, optimised version, which you can then use in models and in the graph editor.

## Deep

Count the parameters of one big head and of four small ones with the same `d_model`. Then
count the FLOPs. What did splitting change, and what did it not?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (2017), section 3.2.2.
- **R**: Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html): heads as independent readers and writers of the residual stream.
- **K**: [Multi-Query Attention](https://arxiv.org/pdf/1911.02150) (Shazeer, 2019): the next lesson's cousin.
