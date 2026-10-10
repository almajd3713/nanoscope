Every statement about scaling is a statement about **compute**: how much a training run costs,
and what you get for it. This lesson builds the arithmetic that the rest of the path stands on,
and finds the one place where the textbook rule fools you on a small model.

## Surface

**Training cost.** A weight takes part in one multiply-add per token on the way forward (2
FLOPs) and about twice that on the way back (4 FLOPs). So a model with `N` parameters trained on
`D` tokens costs about

```
C = 6 * N * D        FLOPs
```

This is the **6ND rule**. It is the number behind "this model cost 10^23 FLOPs". With it you can
turn any pair of the three quantities into the third.

Write three functions in `starter.py`:

- `training_flops(n_params, tokens)`: `6 * N * D`.
- `tokens_for_budget(flops, n_params)`: the tokens a model can train on within a budget.
- `chinchilla_split(flops)`: the model size and token count that spend a budget with about **20
  training tokens per parameter**, the ratio the Chinchilla paper found compute-optimal. If
  `D = 20 N` then `C = 6 N (20 N) = 120 N^2`, so `N = sqrt(C / 120)`.

```bash
nanoscope learn check scaling-lite/01-flops
```

### Spend a budget two ways

`tinystories-5min` trains for 500 steps of 8 sequences of 256 tokens: 1,024,000 tokens. A
model with 100,000 non-embedding parameters costs `6 * 1e5 * 1.024e6 = 6.1e11` FLOPs for that
run. Now ask your functions: with the same budget, how many tokens could a model half the size
see? What is the compute-optimal model for it? (About 71,000 parameters on 1.4 million tokens: a smaller model that sees more data than the preset gives it.)
`Study(budget=FLOPs(...))` does this for you: a cheaper variant trains for more tokens until
every variant has spent the same compute.

### Where the rule is wrong on small models

The 6ND rule counts the weights you give it. Ask the library what one token really costs:

```python
from nanoscope.models import GPT2
from nanoscope.sizing import build_on_meta, count_params, flops_per_token

model = build_on_meta(GPT2, vocab_size=4096, d_model=64, n_layers=2)
total, non_embedding = count_params(model)
print(non_embedding, 6 * non_embedding, flops_per_token(model, 256))
```

For this small GPT-2: 100,096 non-embedding parameters, `6N = 600,576`, but
`flops_per_token = 2,566,656`, more than four times as much. Two things are missing from `6N`:

1. **The output layer.** Every token's hidden state is multiplied by a `d_model x vocab`
   matrix to give a score for each of 4,096 words. Its weights are shared with the embedding, so
   they are not in the "non-embedding" count, but the matrix multiply still happens for every
   token. For a small model with a big vocabulary this is most of the cost.
2. **Attention.** The score and mixing steps between tokens use no weights at all, but they
   cost FLOPs that grow with the context length.

For a 100-billion-parameter model both terms are a rounding error. For the models in this tool
they dominate. That is why the library gives every block a `flops_per_token` and why `Study`
can budget in `FLOPs`: it counts what actually happens.

## Deep

1. **Explain the gap.** Compute the head's FLOPs (`6 * vocab * d_model`) and the attention
   FLOPs (`6 * d_model * context * 2` per layer) and check that, with `6N`, they add up to the
   library's number.
2. **Change the vocabulary.** `build_on_meta(GPT2, vocab_size=50257, ...)`: what happens to
   `flops_per_token` and to the fraction that `6N` explains? Why do tokenizer choices change
   what "the same compute" means?
3. **Over-train on purpose.** Chinchilla's 20 tokens per parameter minimises loss for a fixed
   *training* budget. Models that will be used a lot are trained on far more tokens per
   parameter (LLaMA used hundreds to over a thousand), because inference is paid every time. If
   serving cost matters, which side of the optimum do you want to be on?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: the three functions you wrote.
- **R**: [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) (Hoffmann et al., 2022): where the 20-to-1 ratio comes from.
- **R**: [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) (Kaplan et al., 2020): the first large set of fits, and a different conclusion about how to split the budget.
- **S**: [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/pdf/2404.10102) (Besiroglu et al., 2024): a careful refit that disagrees with part of the original, and why.
- **S**: [LLaMA](https://arxiv.org/pdf/2302.13971) (Touvron et al., 2023): training far past the compute-optimal point on purpose.
