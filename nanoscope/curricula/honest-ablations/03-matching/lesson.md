"Model B beats model A" is only news if B did not simply get more to work with. The cheapest
way to win an ablation is to give the new idea more parameters. This lesson is about not doing
that by accident.

## Surface

**More parameters usually means a lower loss.** A model with twice the weights fits the same
text better, whatever clever idea is inside it. So when you replace a part and the new model
wins, there are two stories: the idea is good, or the new part happens to be bigger. You
cannot tell them apart unless the two models are the same size.

The GELU feed-forward layer in GPT-2 has two weight matrices: up from `d_model` to a hidden
width, then back down. With hidden width `4 * d_model` that is `8 * d_model^2` weights per
layer. SwiGLU (the layer in LLaMA and most modern models) has **three** matrices: two going up
and one coming back. Swap one for the other and keep the hidden width, and the MLP quietly
grows by half. Any win from "SwiGLU" in that comparison has an unfair head start.

This is why papers that introduce SwiGLU shrink its hidden width to about two thirds of the
GELU one: `3 * hidden = 2 * 4 * d_model` gives `hidden = 8/3 * d_model`. The fix is
arithmetic, but the habit is the lesson: **decide what is held equal before you look at any
result.**

### What to match

| Held equal | What it controls | Where it falls short |
|---|---|---|
| Non-embedding parameters | How much the model can store | Two models with equal parameters can do different amounts of work per token |
| Training FLOPs per token | How much compute each token costs | Says nothing about how many tokens you trained on |
| Tokens (training steps and batch) | How much data each model saw | Ignores that one model may be slower to train |

Embeddings are left out of the parameter count because they grow with the vocabulary, which is
a property of the tokenizer, not of the idea. nanoscope's `Study` reports and matches the
non-embedding count for this reason.

Parameters and FLOPs agree for most dense models, because a weight is used once per token.
They split apart for attention (the score computation costs FLOPs without weights), for
mixture-of-experts (many parameters, few active) and for sharing like grouped-query attention.
That is why the library gives each block its own `flops_per_token`, and why the check below
looks at both.

### Make it fair

`MySwiGLU` in `starter.py` is Modern with SwiGLU and a hidden width of 256, which is the width
the GELU control uses. Choose `ffn_hidden` so that its parameters and FLOPs match the control
(a 64-wide, 2-layer Modern with `swiglu=False`) to within 3%.

```bash
nanoscope learn check honest-ablations/03-matching
```

The failing message names both counts and the gap, for example `MySwiGLU has ... non-embedding
parameters, Modern has ... (+36.0%; the lesson allows 3%)`. Do the arithmetic, then confirm
with the check. You can also ask the library:

```python
from nanoscope.sizing import match_params
from nanoscope.models import Modern
match_params(Modern, target, "ffn_hidden", range(64, 512), vocab_size=4096, d_model=64,
             n_layers=2, swiglu=True)
```

where `target` is the control's non-embedding count. `Study(match="params")` does this
for every variant in a study and refuses one that is outside its tolerance (2% by default).

## Deep

1. **Run both comparisons.** Train the control, your matched `MySwiGLU` and an unmatched one
   (`ffn_hidden=256`) for 5 seeds each (a `Study`, or `run(...)` and `compare(...)`). How much
   of the apparent win of the unmatched model disappears when you match? State your answer as
   an interval, not a number.
2. **Match FLOPs, not parameters.** Build a model whose parameters match but whose FLOPs per
   token do not (for example by changing the number of KV heads, or the attention window).
   Which check catches it? When would you prefer to match FLOPs?
3. **The third axis.** Two models matched on parameters and FLOPs can still differ in training
   speed. `Study(budget=FLOPs(...))` gives a slower model fewer tokens; `Tokens(...)` gives
   every model the same data. Which one answers "which idea is better per token" and which
   answers "which is better per hour"?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202) (Shazeer, 2020): introduces SwiGLU and states, in one line, that the hidden width is reduced to keep parameters equal.
- **R**: [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) (Kaplan et al., 2020): loss falls smoothly with parameters, which is why unmatched sizes decide the result.
- **S**: [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) (Hoffmann et al., 2022): what holding compute fixed means.
- **S**: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683) (Raffel et al., 2019): its ablations match parameters and compute and say so.
