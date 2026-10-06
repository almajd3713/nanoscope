The GPT-2 MLP is `proj(gelu(fc(x)))`. Shazeer found a better one: use *two* up-projections,
squash one with **SiLU** (`x * sigmoid(x)`) and multiply it into the other, so one branch
*gates* the other. This is **SwiGLU**, now standard.

## Surface

```
swiglu(x) = proj( silu(w1(x)) * w3(x) )
```

Three linear layers, all without bias: `w1` and `w3` go from `d_model` to `hidden`, `proj` goes
back. Write `MySwiGLU(d_model, hidden=None)`. When `hidden` is not given, use
`8 * ((8 * d_model // 3 + 7) // 8)`: about `8/3 * d_model`, rounded up to a multiple of 8.
Passing **unlocks `SwiGLU`** in `nanoscope.blocks`.

## Deep

Why `8/3`? The GELU MLP has two matrices of size `d x 4d`: `8 d^2` weights. SwiGLU has three of
size `d x h`: `3 d h`. Setting them equal gives `h = 8d/3`. Check it with `sum(p.numel() ...)`:
the two MLPs have (almost) the same number of parameters, so a comparison between them is fair.

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202) (Shazeer, 2020).
- **R**: [LLaMA](https://arxiv.org/pdf/2302.13971) (2023) and [Mistral 7B](https://arxiv.org/pdf/2310.06825) (2023): SwiGLU at the 8/3 width.
