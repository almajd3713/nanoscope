LayerNorm subtracts the mean of a vector, divides by its standard deviation, then scales and
shifts. Zhang and Sennrich noticed that the re-centering does little: dividing by the **root
mean square** is enough, and cheaper. Most current models (Llama, Qwen, Gemma, ...) use it.

## Surface

For a vector `x` with `d` entries:

```
rms(x)  = sqrt(mean(x ** 2) + eps)
rmsnorm = x / rms(x) * weight
```

`weight` is a learned scale per entry, starting at 1. There is no mean subtraction and no bias.
Write `MyRMSNorm(d_model, eps=1e-6)` in `starter.py`; the parameter must be called `weight`.
Passing **unlocks `RMSNorm`** in `nanoscope.blocks`.

## Deep

Feed it a vector and its multiple (`x` and `10 * x`): the output is the same, because the norm
divides the scale away. Why does a network want that property between layers?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: [Root Mean Square Layer Normalization](https://arxiv.org/pdf/1910.07467) (Zhang and Sennrich, 2019).
- **R**: [LLaMA](https://arxiv.org/pdf/2302.13971) (2023) and [Mistral 7B](https://arxiv.org/pdf/2310.06825) (2023): the convergent recipe, with RMSNorm in it.
- **R**: [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745) (Xiong et al., 2020): pre-LN versus post-LN.
