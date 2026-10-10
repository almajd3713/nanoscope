Lesson 1 showed how far a model's score moves when only the seed changes. This lesson is about
a trick that makes that noise matter less: compare two models **on the same seeds**. You will
write both versions of the interval yourself, then watch the paired one find a small
improvement that the unpaired one cannot.

## Surface

You have five scores for model A and five for model B. There are two honest ways to put an
interval around "A minus B".

**Unpaired.** Treat the two sets as unrelated. Take the average of A and the average of B,
subtract, and combine the two spreads. The spread of the difference is built from the spread of
A *and* the spread of B, so it is wide.

**Paired.** Seed 0 of A and seed 0 of B shared something: the same random numbers drew the
training batches. When the seed gives a hard batch order to one model, it gives it to the other
too, so both scores move together. Subtract **seed by seed first** (`A[0] - B[0]`,
`A[1] - B[1]`, ...), then take the average and the spread of those five differences. The part
that moved both models together cancels, and what is left is small.

An analogy: two runners on the same hilly course. Their finishing times each vary with the
wind on the day. If you only compare each runner's average over different days, the wind adds
to the noise. If you compare them on each day, the wind cancels.

### Write the two functions

Open `starter.py`. Both functions take two lists and return `(mean, low, high)`.

For `paired_ci(a, b)`:

```
d      = [x - y for each seed]
n      = len(d)
mean   = average of d
sd     = sample standard deviation of d     (divide by n - 1)
margin = t.ppf(0.975, n - 1) * sd / sqrt(n)
return mean, mean - margin, mean + margin
```

For `unpaired_ci(a, b)`, Welch's version, where the two groups may spread differently:

```
va = sample variance of a / len(a)       vb = sample variance of b / len(b)
df = (va + vb)**2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
margin = t.ppf(0.975, df) * sqrt(va + vb)
return mean(a) - mean(b), that minus margin, that plus margin
```

`t.ppf(0.975, df)` is the number that turns a standard error into a 95% interval: about 2.78
for 4 degrees of freedom, and it falls toward 1.96 as the degrees of freedom grow. A small
sample pays for its uncertainty here. The checks `paired-formula` and `unpaired-formula` hold
your functions against a naive version on fixed numbers and tell you which part is off.

### See it on real runs

The other two checks run one small experiment twice: `TinyGPT` trained for 205 steps against
the same model trained for 200. Five more steps is a real improvement, but a tiny one (about
0.007 bits per byte), smaller than the seed noise you measured in lesson 1. Both checks use
the same five seeds' worth of runs and ask the same question:

- `paired-sees-it`: with the seeds paired, is the 205-step model **better**? Expect yes. Both
  models see the same batches in the same order for the first 200 steps, so they move together
  and almost all the noise cancels.
- `unpaired-misses-it`: the same experiment, but the 200-step runs get different seeds so
  nothing pairs. Expect **within noise**. The effect did not change; only the way of
  comparing did.

When it passes, read the two lines side by side. They report about the same average difference
with intervals of very different width. That width is what pairing buys you: it is worth far
more seeds than you could afford to run.

### When pairing does not help

Pairing only helps when the shared seed makes the two scores move together, as with two runs
that start from the same weights and see the same batches. Two different architectures share
the batch order but not the initial weights, so they move together less. If A and B were
unrelated, subtracting seed by seed removes nothing and still costs one degree of freedom. nanoscope pairs automatically when both sets
used the same seeds, and the table says `(unpaired)` when it could not. Compare two sets that
ran on different seeds and you are in the wide-interval case.

## Deep

1. **Measure the correlation.** Compute the correlation between `A[i]` and `B[i]` over the
   seeds. Pairing helps in proportion to it: the variance of the difference is
   `var(A) + var(B) - 2 * cov(A, B)`. What would a correlation of 0 give? Of 0.9?
2. **Split the seed.** `run(..., seeds=8, data_seed=0)` varies only the initial weights and
   keeps the batch order the same for every run; `init_seed=0` does the reverse. Pair on one
   of them but not the other. Which shared source makes the scores move together more? (The
   Evaluation path examines this directly.)
3. **Power.** For a true difference of 0.03 bpb, how many seeds does the unpaired comparison
   need to exclude zero reliably, against the paired one? Use the standard deviations you
   observed.

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: the two functions you just wrote. They are what `nanoscope.statistics.paired_difference` and `unpaired_difference` do.
- **R**: "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms" (Dietterich, 1998, *Neural Computation*): why the obvious tests mislead on learning-algorithm comparisons.
- **S**: [Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf) (Demšar, 2006): what to do when there are many datasets, not seeds.
- **K**: Welch (1947), "The generalization of 'Student's' problem when several different population variances are involved" (*Biometrika*): the unequal-variance interval you implemented.
