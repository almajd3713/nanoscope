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

## Speed and hardware

Small models are usually limited by the CPU launching GPU kernels, not by the GPU. Measure
before tuning:

```bash
nanoscope bench modern            # step time, tokens/s, and whether the GPU or the CPU is the limit
nanoscope bench modern --compile reduce-overhead
```

Close other GPU programs first; anything else running skews the numbers. Then:

- `--compile` (or `compile=True`) uses `torch.compile`. `--compile reduce-overhead` also
  records CUDA graphs, which is the big win for CPU-bound models: on an RTX 4070 laptop GPU
  Modern went from about 25 ms to 10 ms per step and GPT-2 from about 12 ms to 8 ms. The first
  steps pay a compile cost of up to a minute, so it pays off on longer runs. Checkpoints,
  evaluation and sampling use the plain module, so a run can resume with compile on or off.
- `--devices cuda:0,cuda:1` starts one worker process per device.
- `--workers-per-device N` puts N workers on each device. Four small GPT-2 runs on one GPU
  took 1.85 times as long as one run. nanoscope checks one training step's memory first and
  refuses if N workers don't fit in the free GPU memory. Processes share a GPU by time slicing,
  so the gain is smaller once `--compile reduce-overhead` already keeps it busy.
- `--threads N` caps CPU threads per worker. CPU workers split the cores by default.

Each worker takes a share of the runs listed in `runs/studies/<name>/plan.json`. Studies write
a text sample only at each run's last step. Distributed data parallel is not supported yet.

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
