Scaling laws are the most cited and least trusted result in the field: loss falls smoothly
with size, until it does not. The practical question is small and concrete. If you have
trained three sizes, **how well can you say what the next one up will score?** You will answer
it by predicting first and measuring second.

## Surface

A **ladder** is a set of models that differ only in size, trained the same way. Plot loss
against size on log-log axes and the points fall near a straight line: a **power law**,
`loss = a * size^b` with `b` negative. `nanoscope.scaling.fit_power_law` fits that line to your
runs and returns the exponent `b` with a 95% interval, and `predict(size)` gives the loss it
expects at a new size with a 95% *prediction* interval (the range for one new run).

### Step 1: train the rungs

Save this as `studies/sc02_ladder.py` and run it with `HOLDOUT = False`:

```python
# studies/sc02_ladder.py
from nanoscope import Study
from nanoscope.models import GPT2

HOLDOUT = False  # step 1: the three smaller sizes. Step 2: set True and run again.
ON_GPU = False   # True: the FineWeb-Edu ladder (several GPU-hours)

if ON_GPU:
    preset, widths, holdout, layers, overrides = "fineweb-edu", (128, 192, 256), 384, 4, {}
else:
    preset, widths, holdout, layers, overrides = (
        "tinystories-5min", (32, 48, 64), 96, 2, {"max_steps": 200})

study = Study("sc02-ladder", preset=preset, seeds=3, **overrides)
for width in widths:
    study.add(f"w{width}", GPT2, d_model=width, n_layers=layers)
if HOLDOUT:
    study.add(f"w{holdout}", GPT2, d_model=holdout, n_layers=layers)
```

```bash
nanoscope study studies/sc02_ladder.py     # three sizes, three seeds each
```

### Fit and predict

```python
from nanoscope.models import GPT2
from nanoscope.scaling import fit_power_law
from nanoscope.sizing import build_on_meta, count_params
from nanoscope.studyfiles import finals, study_dir


def params(width, layers=2):
    return count_params(build_on_meta(GPT2, vocab_size=4096, d_model=width, n_layers=layers))[1]


done = finals(study_dir("sc02-ladder"))
points = [(params(int(name[1:])), m["val_bpb"]) for name, seeds in done.items()
          for m in seeds.values()]
fit = fit_power_law(points)
print(fit.exponent, fit.exponent_ci95, fit.status)
print(fit.predict(params(96)))     # (estimate, low, high)
```

Look at the interval on the exponent. With three sizes and nine runs it is wide, and the
`status` says so. That width is the honest answer to "what does a three-size ladder tell
you?": mostly the sign, and not much about the exact slope. Now look at `fit.predict`'s range
for the next size up.

### Step 2: commit, then run the held-out size

Write your prediction in `starter.py` as `PREDICTION = (low, high)`. Use the fit, but also
judge it: is the trend steepening or flattening across the three sizes? Do the small models
have a floor the larger ones will reach? Only after the file is saved, set `HOLDOUT = True` in
the study and run it again (finished runs are skipped, so only the new size trains), then

```bash
nanoscope learn check scaling-lite/02-ladder
```

The check passes when the measured result is inside your range and your range is not far wider
than the fit's own prediction interval. Either way it reports the fit's exponent and interval,
where the fit alone predicted the held-out size to land, and where it really did. Read that
last line before anything else: *did the trend simply continue, or did it not?*

### What a ladder cannot tell you

- **It extrapolates one step.** Predicting a size 10x further out than your largest rung is a
  different claim, and the interval widens to say so.
- **Small and large models can rank ideas differently.** An idea that helps at your sizes may
  not help at 100x the compute. The DataDecide study ran this question across hundreds of small
  models: small experiments predicted the winner at large scale better than chance, but far
  from perfectly, and some choices need larger experiments than others.
- **A fit with a floor is not a pure power law.** Loss cannot fall below the noise in the data.
  A line on log-log axes over-predicts the benefit of size once you approach that floor.

## Deep

1. **Add a rung.** A fourth smaller size narrows the exponent's interval. By how much? What
   happens to the prediction for the held-out size?
2. **Fit FLOPs, not parameters.** Refit by hand against training FLOPs per token
   (`flops_per_token` in `nanoscope.sizing`, from the same `build_on_meta` model) instead of
   parameters. Do the exponents agree? (Lesson 1 explains why they might not.)
3. **Rank two ideas.** Run the ladder for two variants (for example `Modern` and `GPT2`). Do
   their lines cross inside the sizes you ran? What would you conclude from two rungs?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) (Kaplan et al., 2020).
- **R**: [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) (Hoffmann et al., 2022).
- **R**: [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/pdf/2404.10102) (Besiroglu et al., 2024): scaling-law fits are curve fits with error bars, and the error bars matter.
- **R**: [DataDecide: How to Predict Best Pretraining Data with Small Experiments](https://arxiv.org/pdf/2504.11393) (Magnusson et al., 2025): how well small runs predict large ones.
- **S**: [Scaling Data-Constrained Language Models](https://arxiv.org/pdf/2305.16264) (Muennighoff et al., 2023): the trend changes when data runs out.
