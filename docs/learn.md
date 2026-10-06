# Learning paths

nanoscope teaches by building. A **path** (Foundations, The modern block, ...) is a list of
**lessons**. A lesson gives you a starter file, an experiment to run, and **checks** that say,
in words, whether you got it. Everything works from the command line, and the GUI shows the
same files.

```bash
nanoscope learn list                       # paths, lessons, state, CPU/GPU estimates
nanoscope learn start foundations/01-bigram
nanoscope learn check foundations/01-bigram
nanoscope learn status                     # lessons and what is unlocked
```

`learn start` copies the lesson's `starter.py` (and `notebook.py` if there is one) to
`workspace/lessons/<path>/<lesson>/`. It never overwrites your edits. Edit that file, then run
`learn check`. `learn check --queue` runs the check as a worker job instead.

## How a lesson is checked

Each lesson lists checks in its `lesson.toml`. A check prints one verdict and the reason, pass
or fail, and the whole result is written to `learn/checks/<id>.json`.

| Check | It asks |
|---|---|
| `defines` | Is the class in your file, and does it build (on the `meta` device, so no memory)? |
| `equivalent` | Does your block match a naive reference function on random inputs? The reason gives the largest difference and the input shape that failed. |
| `forbid` | Does your file avoid the shortcuts the lesson forbids (`F.scaled_dot_product_attention`, `torch.nn.MultiheadAttention`) and the blocks you have not unlocked? It reads your file with `ast`, so it reports line numbers. |
| `trains` | Does `run()` of your model reach the target loss on the lesson's preset? |
| `verdict` | Training your model against another with several seeds, is the difference the one the lesson expects (better, worse, within noise)? |
| `predicted` | Did you commit to a prediction *before* the experiment, and does it hold? See below. |
| `reproduces` | Does your result land where the shipped baseline's seeds do? |

If a `defines` check fails, the rest are skipped: they would fail for the same reason.

### Predictions

`nanoscope learn predict <lesson> --verdict better --low -0.2 --high -0.05` writes
`prediction.toml` and records the time and a hash of the file. It must come before the first
check of that lesson: once the experiment has run, the CLI refuses a prediction, and the
`predicted` check refuses one that was recorded late or edited afterwards. A prediction counts
when its interval contains the result and is not far wider than the run's own uncertainty
(an interval that covers everything predicts nothing).

### Compute: a CPU variant and a GPU variant

A lesson that needs GPU hours has two variants. `[compute.cpu]` is a modest version that runs
anywhere and in CI; `[compute.gpu]` is the full one. Each names a preset (or a budget) and an
estimate in minutes. A CPU estimate over 15 minutes requires a GPU variant. `learn list` shows
both estimates, and `estimate_seconds` replaces the declared one with a measurement when you
have run `nanoscope bench --save`.

## Locked components

Bigger blocks (`Attention`, `Block`, `Decoder`, `RoPE`, GQA, ...) are **locked** until you
have built them from simpler parts in a lesson. The lock list is not a separate file: a block
or feature is lockable exactly when some lesson lists it under `unlocks`.

The first `nanoscope learn start` turns on the **guided** policy (`--open` skips it). Until
then, and in every script, notebook and Kaggle session that never ran `learn start`, nothing
is locked. When a lesson's checks all pass, its unlocks are written to `learn/unlocks.json`
with the check result as evidence.

```bash
nanoscope learn unlock --all                          # open everything (recorded as "open")
nanoscope learn unlock block:Attention --reason "I know this"   # skip one lesson (recorded)
nanoscope learn lock --reset                          # guided again; keeps earned and skipped
```

Both ways out are recorded, so `learn status` shows what you earned and what you skipped.

### What is gated, and what never is

Locks limit what you **compose**, never what you **run**. `run(GPT2)`, `run(Modern,
n_kv_heads=1)`, presets, shipped baselines, Study variants of shipped models and generation all
work under `guided`. Three things are gated, by one policy function
(`nanoscope.learn.gating.check`):

1. **Import**: `from nanoscope.blocks import Attention` raises `LockedBlockError` (an
   `ImportError`) naming the lesson and the unlock command.
2. **Build**: constructing your own `Decoder` or `Composite` checks the blocks and features it
   uses (GQA, QK-norm, sliding window, z-loss), which an import cannot see.
3. **Static**: `gating.scan(path)` reads a file without running it. The editor, the graph's
   lock badges, `validate_run_request` and the `forbid` check use it.

`nanoscope run` and a Study refuse a locked model before anything starts, with exit code 2.

### In a raw file versus the editor

A raw `.py` file run with `python`, `nanoscope run`, a Study or a notebook meets points 1 and 2
when it runs. The in-site editor adds point 3 on save, before anything runs, and the graph shows
a locked node as locked.

### Limits: this is a learning aid, not security

Gating is **not security**. Python cannot stop determined code: importing
`nanoscope.blocks.attention` directly, `importlib`, copying the shipped source, or
`torch.nn.MultiheadAttention` all get around it. The `forbid` check catches the obvious cases
at check time; the rest is the honor system. Subclassing a shipped model counts as running it,
a loophole that is documented rather than closed. Gating protects nothing in a shared
deployment and is never used for access control.

## Where things live

| What | Where |
|---|---|
| Lessons | `nanoscope/curricula/<path>/<nn-slug>/` (`lesson.toml`, `lesson.md`, `starter.py`) |
| Your progress | `$NANOSCOPE_HOME/learn/progress.json` |
| Your unlocks | `$NANOSCOPE_HOME/learn/unlocks.json` |
| Check results | `$NANOSCOPE_HOME/learn/checks/<id>.json` |
| Your files | `$NANOSCOPE_HOME/workspace/lessons/<path>/<lesson>/` |

In a hosted deployment each user's learner state moves under `users/<name>/learn/`.
