In the honest-ablations path a seed was one dial. It is really two. The seed chooses the
**initial weights**, and it chooses the **order of the training batches**. Either can move the
final score. Which one moves it more tells you what to hold fixed when you want a quieter
comparison, and what you are actually averaging over when you run several seeds.

## Surface

`run()` takes `init_seed` and `data_seed`, each defaulting to `seed`. Run a small **grid**: three
initialisations, each trained on three data orders. Nine runs, about five minutes on a laptop CPU.

```python
from nanoscope import run
from nanoscope.models import GPT2

grid = [
    [run(GPT2, "tinystories-5min", init_seed=i, data_seed=j, device="cpu", d_model=64,
         n_layers=2, max_steps=200, progress=False).val_bpb[-1][1]
     for j in range(3)]
    for i in range(3)
]
print(grid)
```

Row `i` is one set of starting weights; column `j` is one data order. If the initial weights
mattered most, the rows would differ and each row would be roughly constant. If the data order
mattered most, the columns would differ. On one run of this grid the nine scores were between
1.556 and 1.595: rows averaged within 0.006 of each other, columns within 0.015. The data order
moved the score more than the initialisation did, at this size and budget.

### Split it with a function

Write `variance_split(grid)` in `starter.py`. It returns three **mean squares**, which are
variances you can compare:

- **init**: how much the row averages differ from the grand average, scaled by the number of
  columns. Large when the starting weights matter.
- **data**: the same for the column averages. Large when the batch order matters.
- **residual**: what neither explains. If it is large, the effects interact (some data orders
  suit some initialisations) or it is plain unexplained noise.

The formulas are in the docstring. For the grid above they came to roughly `init = 0.00015`,
`data = 0.00039`, `residual = 0.00010`: data order carries the most, and a lot is left over.

```bash
nanoscope learn check evaluation/02-variance
```

### Turn mean squares into the spread you care about

A mean square for a factor with `J` replicates per level estimates
`residual + J * (the factor's own variance)`. So the factor's own variance is
`(mean square - residual) / J`. For the grid above: init about 0.000016 (a standard deviation of
0.004), data about 0.000096 (0.010), residual 0.000099 (0.010). Add them and take the root:
0.0145, which is the seed-to-seed standard deviation from lesson 1 of the honest-ablations path.
A grid this small estimates each part roughly: three levels per factor is not many. Treat the
ratios as an order of magnitude, not a measurement.

### What to do with it

- To **compare two models more sharply**, share the source that moves the score most. Pair on
  the data order (`data_seed`) if it dominates: both models then see the same batches, and that
  noise cancels. Vary the other one across your replicates.
- To **ask "is this idea robust?"**, vary both. A result that holds across initialisations and
  data orders is a stronger claim than one that holds across one.
- When you write `seeds=5`, remember what you averaged over: the initial weights and the data
  order together.

## Deep

1. **Make the grid bigger.** Three by three is the minimum. Run 5 by 5 (the GPU variant uses
   `tinystories-30min` and a larger model) and see whether the ranking of init and data holds.
2. **Fix one, vary the other.** Train five runs with `init_seed=0, seeds=5` and five with
   `data_seed=0, seeds=5`. Their spreads are two estimates of the data-order and initialisation
   standard deviations. Do they agree with the grid?
3. **Does size change it?** Repeat the grid with a wider model. Larger networks often depend
   less on the initial draw. Is that what you see?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: `variance_split`. It is a two-way analysis of variance without the statistics package.
- **R**: [Accounting for Variance in Machine Learning Benchmarks](https://arxiv.org/pdf/2103.03098) (Bouthillier et al., 2021): exactly this decomposition, on real benchmarks.
- **R**: [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/pdf/1908.03265) (Liu et al., 2019), for the warm-up story behind early-training variance, if you want a reason the first steps matter so much.
- **S**: [Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping](https://arxiv.org/pdf/2002.06305) (Dodge et al., 2020): the same two seeds, in fine-tuning.
