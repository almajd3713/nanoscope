# Research guide

A `Study` trains several model variants with several seeds under one budget and reports
differences with confidence intervals. It runs the same `run()` the learner uses.

```python
from nanoscope import Study, Tokens
from nanoscope.models import GPT2, Modern

study = Study("my-study", preset="tinystories-5min", seeds=3, budget=Tokens(4e6),
              match="params", baseline="modern", mode="record")
study.add("gpt2", GPT2)
study.add("modern", Modern)
```

```bash
nanoscope study studies/my_study.py --devices cuda:0,cuda:1
nanoscope report studies/my_study.py
```

See [`studies/m1_ablation.py`](../studies/m1_ablation.py) for a full example.

## Budgets

- `Tokens(n)`: every run sees `n` training tokens.
- `FLOPs(n)`: every run gets `n` FLOPs, so a model that costs more per token sees fewer tokens.
  FLOPs per token use the PaLM formula, including the tied unembedding.

## Matching

`match="params"` requires the variants' **non-embedding** parameter counts to agree within
`tolerance` (default 2%) and refuses to start otherwise. Use `nanoscope.sizing.match_params`
to choose a width, e.g. `ffn_hidden`, that hits a target.

## Explore and record mode

- `mode="explore"` (default): no restrictions, for trying things out.
- `mode="record"`: the git tree must be clean, the commit is stored with each run, and the
  preregistration is frozen in `runs/studies/<name>/study.json` before training. Declare
  expected outcomes with `study.predict("variant", ...)`. Record-mode results refuse
  `resume=False`, so they can't be silently rerun.

## Comparing

`compare` uses a paired t-interval when the seeds match and Welch's otherwise. It refuses to
compare runs scored on different evaluation text, and refuses `val_loss` across tokenizers
(use bits per byte). The `params` column shows non-embedding parameters.

## Several devices

`--devices cuda:0,cuda:1` starts one worker process per device. Each worker takes a share of
the runs listed in `runs/studies/<name>/plan.json`. Distributed data parallel is not
supported yet.

## Seeing progress

Every long operation has a live indicator and on-disk state:

```bash
nanoscope status runs
```

## Kaggle

[`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb) runs any study file. Set `STUDY` in the
first cell, and `RUNS_REPO` to a private Hub repo that keeps runs between sessions. Public token data downloads from the Hub, so no tokenization happens on Kaggle.

## Saving and resuming on the Hub

`Study(..., push_to_hub="user/runs")` or `run(..., push_to_hub="user/runs")` mirrors each run,
checkpoints included, to a private model repo at checkpoints (at most every 20 minutes). A fresh
machine with no local checkpoint pulls it and resumes exactly, including data-order and RNG state.

`wandb=True` on `run()` logs live curves to Weights & Biases.
