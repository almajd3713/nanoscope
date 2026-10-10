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

## Paths

`nanoscope learn list` shows these with state. Estimates are the lesson's declared CPU / GPU
minutes; a GPU figure of "measured" is filled in once someone has run it on real hardware.

| Path | Level | Lessons | CPU / GPU estimate | What it teaches |
|---|---|---|---|---|
| `foundations` | 0 | 6 | 14 min / none | A bigram, an MLP, an attention head, multi-head attention, a block, GPT-2. |
| `modern-block` | 1 | 7 | 15 min / 10 min | RMSNorm, RoPE, SwiGLU, GQA, QK-norm, z-loss, then the assembled model. |
| `honest-ablations` | 2 | 5 | 95 min / 60 min | Seed noise, paired and unpaired comparison, matching size, predicting before running, and an ablation study of your own. |
| `evaluation` | 2 | 2 | 18 min / 60 min | How much of a number is the evaluation set, and how much is init versus data order. |
| `efficiency` | 2 | 3 | 6 min / 10 min | Throughput and what bounds it, when `torch.compile` pays, the roofline. KV cache is coming. |
| `scaling-lite` | 2 | 2 | 14 min / 480 min | FLOP budgets, then a three-size ladder with a power-law fit and a prediction written before the largest size is trained. |
| `interpretability` | 2 | 2 | 12 min / 120 min | Induction heads measured over your own checkpoints, and the logit lens. |
| `extend` | 3 | 3 | 14 min / 70 min | A certified block, your own preset, Muon versus AdamW, and a mixture of experts built from primitives. |

## How a lesson is checked

Each lesson lists checks in its `lesson.toml`. A check prints one verdict and the reason, pass
or fail, and the whole result is written to `learn/checks/<id>.json`.

| Check | It asks |
|---|---|
| `defines` | Is the class in your file, and does it build (on the `meta` device, so no memory)? |
| `equivalent` | Does your block match a naive reference function on random inputs? The reason gives the largest difference and the input shape that failed. |
| `forbid` | Does your file avoid the shortcuts the lesson forbids (`F.scaled_dot_product_attention`, `torch.nn.MultiheadAttention`) and the blocks you have not unlocked? It reads your file with `ast`, so it reports line numbers. |
| `trains` | Does `run()` of your model reach the target loss on the lesson's preset? |
| `verdict` | Training your model against another with several seeds, is the difference the one the lesson expects (better, worse, within noise)? `a_optimizer`/`b_optimizer` name an optimizer factory (`muon`, `adamw`, or a function in your file) for each side. `pairing = "unpaired"` gives the second model other seeds, so the comparison cannot be paired. |
| `predicted` | Did you commit to a prediction *before* the experiment, and does it hold? See below. With `must_hit = false` a miss is reported but does not fail the lesson, since a wrong prediction cannot be redone. |
| `reproduces` | Does your result land where the shipped baseline's seeds do? |
| `matches` | Is your model the same size as another (non-embedding parameters or training FLOPs per token, within a tolerance)? Sizing a comparison fairly. |
| `function` | Does a plain function in your file give the same numbers as a naive reference on fixed cases? The reason names the case and the part that is off. |
| `certified` | Is your block registered with `register_block(reference=...)`, does it match its own reference, and does it also match the lesson's? A formula wrong the same way twice passes the first test only. |
| `preset` | Does your file register a preset derived from a shipped one, with the data fields kept (so cached tokens apply) and a schedule that makes sense (warmup inside the run, at least one evaluation)? |
| `ablation` | The capstone of the honest-ablations path: reads your own finished `Study`, compares the full model, a baseline and each removed component, and holds the `GAP` interval and `NULL_COMPONENT` in your file against what it found. It trains nothing. |

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
