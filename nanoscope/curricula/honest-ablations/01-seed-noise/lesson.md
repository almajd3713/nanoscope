Every experiment in this path ends with a sentence like "model B is better than model A".
Before you can trust a sentence like that, you need to know how often **nothing at all** looks
like an improvement. This lesson measures that, using the cheapest experiment there is: train
one model several times and see how far apart the answers land.

## Surface

Training a network starts from random numbers: the initial weights, and the order in which the
training batches are drawn. Both come from a **seed**. Change the seed and you get a slightly
different model, even though you changed nothing about the model itself. So one number from one
run is one draw from a spread, not the answer.

How wide is the spread? Try it. In a Python session or a notebook:

```python
from nanoscope import run
from nanoscope.models import GPT2

group = run(GPT2, "tinystories-5min", seeds=5, d_model=64, n_layers=2, max_steps=200)
print([round(v, 3) for _, v in (r.val_bpb[-1] for r in group)])
```

(`val_bpb` is the validation loss in bits per byte: lower is better. 200 steps is a short run
on purpose, so five seeds finish in about three minutes on a laptop CPU.)

On a typical CPU run the five numbers land somewhere between 1.55 and 1.60. Nothing differs
between those runs except the seed. The gap between the best and the worst, about 0.04
here, is the **noise floor** for this model at this budget. Your numbers will not match
exactly, but they will have a spread of the same size.

Now the point. Suppose you change something (a different norm, one fewer head) and your
result moves by 0.02. Is that the change, or is that just the seed? You cannot say from one run
of each. The spread tells you what "nothing happened" looks like, so you can tell when
something did.

### The A/A test

A statistician's way to measure this is an **A/A test**: compare a model with an identical copy
of itself. They are the same by construction, so a good procedure should say "no difference".
That is what the check does:

1. `MyGPT` is your copy. It must match the shipped GPT-2 (64 wide, 2 layers) in size. The
   `same-size` check enforces it, because a model that is not the same would no longer be an
   A/A test.
2. Each model trains on 5 seeds. The two sets use **different seeds**, so nothing links a seed
   of A to a seed of B (the next lesson shows why that matters).
3. `compare()` computes the average difference and a 95% **confidence interval** around it. If
   the interval includes zero, the verdict is *within noise*, and that is what the check
   expects.

Fix the sizes in `starter.py` and run

```bash
nanoscope learn check honest-ablations/01-seed-noise
```

Read the line it prints: `MyGPT vs GPT2: within noise (val_bpb difference ... [low, high])`.
The width of `[low, high]` is the size of the spread your five seeds produce. A real
improvement has to push the whole interval past zero.

### Why the interval and not just "the numbers differ"

Two sets of five runs never have exactly the same average. With a 95% interval, roughly one A/A
comparison in twenty will exclude zero by luck alone. That false alarm rate is the price of
using the interval; it is also why a single "significant" result deserves less trust than it
feels like it does, and why this path keeps asking for more seeds when the stakes rise.

## Deep

1. **More seeds, narrower interval.** The interval shrinks roughly with the square root of the
   number of seeds: ten seeds give an interval roughly 60% as wide as five. Run the check with
   `seeds = 10` (the `deep` tier above) and watch the interval narrow. How many seeds would you
   need to reliably detect a difference of 0.01 bpb? Estimate it from the sample standard
   deviation of your seeds: you need roughly `(2 * 1.96 * sd / 0.01) ** 2` runs per model, or a
   fraction of that if you pair them (lesson 2).
2. **Noise depends on the budget.** Repeat with `max_steps = 500` or a wider model. Is the
   spread bigger or smaller than at 200 steps? Why might a bigger model be steadier?
3. **Two kinds of seed.** `run(..., init_seed=0, seeds=5)` keeps the initial weights fixed and
   varies only the batch order; `run(..., data_seed=0, seeds=5)` does the opposite. Which source
   of randomness moves the result more at this size? (The Evaluation path takes this apart
   properly.)

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [Show Your Work: Improved Reporting of Experimental Results](https://arxiv.org/pdf/1909.03004) (Dodge et al., 2019): why one run per model is not evidence, and what to report instead.
- **R**: [Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560) (Henderson et al., 2018): the same lesson learned the hard way in another field, with seeds that flipped the ranking of algorithms.
- **S**: [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/pdf/2103.03098) (Bouthillier et al., 2021): the sources of variance (initialisation, data order, data split) taken apart.
- **K**: `nanoscope.statistics.summarize`: the 95% interval over seeds used by every comparison in this tool.
