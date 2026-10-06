The MLP can only see a fixed window. **Attention** lets each token decide for itself which
earlier tokens matter, and how much. One attention head is five small steps.

## Surface

For input `x` of shape `(batch, time, d_model)`:

1. Three linear layers, no bias, make a **query** `q`, a **key** `k` and a **value** `v` for
   every token. (`q` asks "what am I looking for?", `k` says "what do I contain?", `v` is "what
   I hand over if you pick me".)
2. **Scores**: `q @ k.transpose(-2, -1) / sqrt(d_model)`. Entry `[t, s]` says how much token
   `t` likes token `s`. The division keeps the numbers from growing with `d_model`.
3. **Mask**: a token may only look at itself and earlier tokens, so scores with `s > t` become
   `-inf` (`masked_fill`, with a triangle from `torch.triu`).
4. **Softmax** over the last dimension turns each row of scores into weights that sum to 1.
5. **Mix**: `weights @ v`, a weighted average of the values. Then one more linear layer,
   called `out`, projects the result.

Write `OneHead` in `starter.py` with exactly these four layer names (`q`, `k`, `v`, `out`): the
check copies your weights into a slow, obvious reference (loops over positions) and compares
the outputs on random inputs of several shapes.

**You must write it yourself.** The check refuses `F.scaled_dot_product_attention` and
`torch.nn.MultiheadAttention`; they are the thing you are learning to build. (The primitives in
`nanoscope.blocks.primitives` are fair game, and in the graph editor the
`AttentionTemplate` has an empty slot for each of these steps.)

## Deep

Print the weights matrix for a short sentence. Row `t` has zeros after column `t` (that is the
mask), and each row sums to 1. Try removing the mask: what does the model get to see, and why
would it learn to cheat during training?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (2017), section 3.2: implement it from scratch, then again without looking.
- **B**: The Illustrated / Annotated Transformer, and Karpathy's nanoGPT: read the code line by line.
- **S**: Jay Alammar, "The Illustrated Transformer".
