The honest-ablations path measured one source of uncertainty: the seed. A score has another,
easier to miss. It is measured on a **finite** set of validation documents, and a different set
would give a slightly different number for the very same model. This lesson measures that,
and explains why nanoscope's comparisons mostly do not need to worry about it.

## Surface

Train a model once. Its validation loss on the whole set is one number. Now cut the validation
tokens into ten equal pieces and measure the loss on each piece. Same model, same weights, ten
different answers: the pieces differ only in which stories they contain, some easier than
others.

```python
import dataclasses
import math

import torch

from nanoscope import run
from nanoscope.models import GPT2
from nanoscope.train_loop import evaluate

result = run(GPT2, "tinystories-5min", device="cpu", d_model=64, n_layers=2, max_steps=200)
data, preset = result.data, result.preset
n_chunks = 10
size = len(data.val) // n_chunks
losses = [
    evaluate(result.model, dataclasses.replace(data, val=data.val[i * size:(i + 1) * size]),
             preset, torch.device("cpu"))
    for i in range(n_chunks)
]
print([round(x, 3) for x in losses])
```

On one run of this, the ten losses were between 4.26 and 4.44 nats per token, with a standard
deviation of 0.062 between pieces. Each piece has only 20 documents, so you expect it to
wobble. The question that matters is how much the **average** wobbles, and that is what the
**standard error** answers:

```
standard error = sample standard deviation / sqrt(n)        (sample: divide by n - 1)
```

For those ten pieces it came to about 0.020. A 95% interval for the average is about 1.96
standard errors either side, so the model's loss on the full 200-document set is known to
roughly ±0.04 nats. Compare that with the seed noise from lesson 1 (about 0.015 bits per byte,
or roughly 0.04 nats): at 200 evaluation documents, the two sources are about the same size.

### Write two functions

Open `starter.py`:

- `standard_error(values)`: the formula above.
- `chunks_needed(sd, half_width)`: how many pieces of that spread you need before the
  interval is no wider than `half_width` either side. Solve `half_width = 1.96 * sd / sqrt(n)`
  for `n`, and round up. It is the planning question behind choosing `eval_docs`: halving the
  interval takes four times the documents.

```bash
nanoscope learn check evaluation/01-eval-noise
```

Try your `chunks_needed` on the numbers from your own run: how many 20-document pieces would
it take to know the loss within ±0.01?

### Why comparisons mostly escape this

`eval_docs` is fixed in a preset: every run, and both sides of every comparison, is scored on
**the same documents**. A hard story is hard for both models. When you subtract one model's
score from the other's, most of the difficulty of the documents cancels, exactly as pairing by
seed cancelled shared batches in lesson 2 of the honest-ablations path. So the interval on a
difference is much narrower than the interval on either score alone.

That is why nanoscope treats **seeds** as the replicates and not the validation tokens (see
`nanoscope/statistics.py`), and why a comparison uses one `eval_docs` for everything. The
exception is when you compare a score from your run with a score someone else computed on
different documents: then the document sampling error does not cancel, and it is as large as
you just measured.

## Deep

1. **Show the cancellation.** Train two different models and evaluate both on the same ten
   pieces. Compute `standard_error` of each model's ten losses, and of the ten *differences*.
   How much narrower is the difference?
2. **Piece size.** Repeat with 5 pieces and with 20. The standard deviation between pieces
   falls as pieces get bigger, but the standard error of the average should not change much.
   Why not?
3. **Choose `eval_docs`.** If you need to detect a 0.01 bpb difference between two models, is
   the validation set or the number of seeds your bottleneck? Use `chunks_needed` for the first
   and the seed spread from lesson 1 for the second.

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: the two functions you wrote. They are the planning arithmetic for any evaluation.
- **R**: [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/pdf/2405.14782) (Biderman et al., 2024): how small choices in evaluation move the numbers.
- **R**: [Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations](https://arxiv.org/pdf/2411.00640) (Miller, 2024): standard errors for benchmark scores, and paired comparisons between models.
- **S**: [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/pdf/2103.03098) (Bouthillier et al., 2021): the sources of variance, including the test set.
