Attention moves information *between* tokens. A transformer layer pairs it with an MLP that
works on each token *on its own*, and wraps both in two things that make deep stacks trainable:
**LayerNorm** (keep the numbers in a sane range) and **residual connections** (each part adds
to its input instead of replacing it).

## Surface

`starter.py` gives you the `MultiHead` you wrote last lesson. Write `MyBlock`, a *pre-norm*
block:

```
x = x + attn(ln1(x))
x = x + proj(gelu(fc(ln2(x))))
```

- `ln1`, `ln2`: `nn.LayerNorm(d_model)`, two separate layers (they learn separate scales).
- `attn`: your `MultiHead`.
- `fc`: `d_model -> 4 * d_model`, `proj`: `4 * d_model -> d_model`, both without bias.
- `gelu`: use `F.gelu(x, approximate="tanh")`, the version GPT-2 uses.

The check randomises every weight, copies them into a reference block, and compares. Passing
**unlocks `Block`** in `nanoscope.blocks`.

## Deep

Each residual path is a "stream" that every layer reads from and writes to. Try swapping to
*post-norm* (`x = ln(x + attn(x))`) and train a deep stack: why was pre-norm the fix that let
people train 100-layer models?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) (2017): the block is its figure 1.
- **R**: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745) (Xiong et al., 2020): pre-LN versus post-LN.
- **R**: [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019): the exact block you are writing.
- **K**: [BERT](https://arxiv.org/pdf/1810.04805) (2018): the encoder branch, and why it lost for generation.
