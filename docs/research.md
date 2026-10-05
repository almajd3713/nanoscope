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

## Where things live

Without any setup, runs go to `./runs`, reports to `./experiments` and token data to
`~/.nanoscope/data`. Set `NANOSCOPE_HOME` and everything moves under it (`runs/`,
`experiments/`, `data/`, `hardware/`, `learn/`, `workspace/`), read at call time, so a
container or a shared lab machine needs one variable. `NANOSCOPE_DATA_DIR` and
`NANOSCOPE_WORKSPACE` still override their own folder.

A run is addressed by its **ref**, its path under the runs folder:
`tinystories-5min/modern-1a2b3c4d/seed-0` is one run, `tinystories-5min/modern-1a2b3c4d` is
the set of seeds, `studies/m1-ablation/no-rope/seed-2` is a study run, and
`baselines/tinystories-5min/gpt2` is a result shipped with the package. `compare()`,
`nanoscope compare`, `nanoscope stop` and `nanoscope.load_run(ref)` all take refs.

## Budgets

- `Tokens(n)`: every run sees `n` training tokens.
- `FLOPs(n)`: every run gets `n` FLOPs, so a model that costs more per token sees fewer tokens.
  FLOPs per token use the PaLM formula, including the tied unembedding.

## Matching

`match="params"` requires the variants' **non-embedding** parameter counts to agree within
`tolerance` (default 2%) and refuses to start otherwise. Use `nanoscope.sizing.match_params`
to choose a width, e.g. `ffn_hidden`, that hits a target.

## Studies as TOML

A study is data, so it can be written, committed and read by tools. `nanoscope spec` prints
any study file as TOML, and `study`, `report` and record mode accept `.toml` paths:

```bash
nanoscope spec studies/m1_ablation.py > studies/m1_ablation.toml
nanoscope study studies/m1_ablation.toml --devices cuda:0
```

```toml
name = "m1-ablation"
preset = "tinystories-5min"
seeds = [0, 1, 2]
match = "params"
match_knob = "ffn_hidden"   # resize this knob so every variant matches `match_to`
match_to = "gpt2"
range = [64, 1024, 8]
baseline = "modern"

[budget]
tokens = 4000000.0

[[variants]]
name = "gpt2"
model = "nanoscope.models.gpt2:GPT2"

[[variants]]
name = "no-rope"
model = "nanoscope.models.modern:Modern"

[variants.kwargs]
rope = false
```

A model is named by a ref: `module:Class` or `path/to/file.py:Class`. In record mode the TOML
file is the thing that gets preregistered, so it must be committed first. In Python,
`Study.to_spec()` and `Study.from_spec(spec)` do the same conversion.

## Explore and record mode

- `mode="explore"` (default): no restrictions, for trying things out.
- `mode="record"`: the git tree must be clean, the commit is stored with each run, and the
  preregistration is frozen in `runs/studies/<name>/study.json` before training. Declare
  expected outcomes with `study.predict("variant", ...)`. Record-mode results refuse
  `resume=False`, so they can't be silently rerun.

## Comparing

`compare` uses a paired t-interval when the seeds match and Welch's otherwise. It refuses to
compare runs scored on different evaluation text, and refuses `val_loss` across tokenizers
(use bits per byte). The `params` column shows non-embedding parameters. Every row carries a
`verdict` against the baseline: `better` or `worse` when the 95% interval excludes zero,
`within noise` when it doesn't, `no CI` with fewer than three seeds. `comparison.to_dict()`
gives the same table as plain data.

## Speed and hardware

Small models are usually limited by the CPU launching GPU kernels, not by the GPU. Measure
before tuning:

```bash
nanoscope bench modern            # step time, tokens/s, and whether the GPU or the CPU is the limit
nanoscope bench modern --compile reduce-overhead
nanoscope bench modern --save     # also append the result to $NANOSCOPE_HOME/hardware/bench.jsonl
```

Close other GPU programs first; anything else running skews the numbers. Then:

- `--compile` (or `compile=True`) uses `torch.compile`. `--compile reduce-overhead` also
  records CUDA graphs, which is the big win for CPU-bound models: on an RTX 4070 laptop GPU
  Modern went from about 25 ms to 10 ms per step and GPT-2 from about 12 ms to 8 ms. The first
  steps pay a compile cost of up to a minute, so it pays off on longer runs. Checkpoints,
  evaluation and sampling use the plain module, so a run can resume with compile on or off.
- `--devices cuda:0,cuda:1` starts one worker process per device; they take the study's runs
  from the job queue (below).
- `--workers-per-device N` puts N runs at once on each device. Four small GPT-2 runs on one
  GPU took 1.85 times as long as one run. A worker measures one training step's memory first
  (cached in the queue) and leaves a run queued until it fits the free GPU memory. Processes
  share a GPU by time slicing, so the gain is smaller once `--compile reduce-overhead` already
  keeps it busy.
- `--threads N` caps CPU threads per run. Runs sharing a CPU split the cores by default.

Studies write a text sample only at each run's last step. Distributed data parallel is not
supported yet.

## The job queue and workers

Anything long that is not run in your own process goes through one SQLite queue,
`$NANOSCOPE_HOME/queue.db` (WAL mode). Back it up by copying `queue.db` together with its
`-wal` file while no worker runs, or with `sqlite3 queue.db ".backup copy.db"`. Finished
runs live in their folders, so losing the queue loses only what was waiting.

```bash
nanoscope worker --device cuda:0 --slots 2   # claim jobs, one child process per job
nanoscope jobs                               # every job: id, state, lane, kind, ref, error
nanoscope jobs --state queued
nanoscope jobs cancel 7                      # queued: cancelled now; running: STOP, then cancelled
nanoscope status --workers                   # each worker's device, slots, jobs, last heartbeat
```

- **Lanes.** `interactive` jobs (what a person is waiting for) are claimed before `batch`
  jobs (a study's runs); inside a lane the oldest job goes first.
- **Leases.** A claimed job is leased for 60 seconds and the worker renews it while the child
  makes progress. If a worker dies (even `kill -9`), the lease expires, the job returns to the
  queue with `attempts + 1` and the next worker resumes the run from its last checkpoint.
  After 3 attempts it fails.
- **Folders win.** A run whose folder already says `done` is never run twice: enqueueing it
  gives no job, and a queued job for it is marked done without running.
- **Stopping.** `SIGTERM` makes a worker write STOP to its runs, wait for their checkpoints and
  hand the jobs back to the queue. `--timeout N` (or a job's own `timeout`) stops a run after
  N seconds and fails the job with `timeout after Ns`. `--exit-when-idle` ends the worker when
  the queue is empty.
- **Secrets.** `HF_TOKEN` and `WANDB_API_KEY` reach only the jobs that use them (a run with
  `push_to_hub` or `wandb`, `prepare-data`).

`Study.run()` with no devices, or with one, still runs everything in your own process, with no
queue and no subprocess. `--shard` is gone: use `--devices` (and `--workers-per-device`).
`Study.enqueue()` puts a study's unfinished runs on the batch lane without starting workers,
for a worker you run yourself.

## Seeing progress

Every long operation has a live indicator and on-disk state:

```bash
nanoscope status            # every run under the runs folder
nanoscope status --data     # data preparation: downloads, tokenizer, tokenizing
```

Each run keeps a `status.json` that `nanoscope status` reads, written atomically, with one of
these states: `queued`, `preparing`, `running`, `done`, `stopped` (Ctrl-C), `cancelled`
(`nanoscope stop`) or `failed`. A failed run records the error type, message and the last
lines of the traceback, and `nanoscope status` prints them. A run that says `running` but
hasn't refreshed its file for a minute is shown as `running (no heartbeat for 3m)`: its
process is gone. Runs written before `status.json` existed are judged by file times instead.

Data preparation writes `$NANOSCOPE_DATA_DIR/<preset>/prepare.json` while it downloads,
trains a tokenizer or tokenizes, and records the error if it fails. Nothing is written when
the data is already prepared.

All messages go through `logging.getLogger("nanoscope")`; by default they print as
`[nanoscope] ...`. Add your own handler to route them elsewhere.

## Stopping and resuming

```bash
nanoscope stop tinystories-5min/modern-1a2b3c4d/seed-0   # one run
nanoscope stop m1-ablation                               # a study: every running run, and
                                                         # the runs that haven't started
```

`nanoscope stop` writes a `STOP` file; the run finishes its step, saves a checkpoint and ends
as `cancelled`. Running it again resumes where it stopped (a stale `STOP` is cleared). In a
study, cancelling one run lets the study carry on with the next; Ctrl-C or stopping the study
ends it. `checkpoint_steps=[100, 500]` on `run()` additionally keeps full checkpoints at those
steps in `checkpoints/archive/`, never pruned.

## Files on disk

Everything another tool might read is JSON with a `schema` number and the `nanoscope` version
that wrote it. The schemas are in `nanoscope/schemas/` (JSON Schema 2020-12).

| File | Written by | Schema |
|---|---|---|
| `<run>/config.json` | `run()` | `config.v1`: model (class, kwargs, `ref`, `source_sha256`), preset, seed, stats |
| `<run>/status.json` | `run()` | `status.v1` |
| `<run>/metrics.jsonl` | the trainer | one row per step, not versioned |
| `<run>/latest.json`, `checkpoints/` | the trainer | checkpoint pointer and `.pt` files |
| `<run>/STOP` | `nanoscope stop` | empty marker file |
| `studies/<name>/plan.json` | `Study` | `plan.v1` |
| `studies/<name>/study.json` | record mode | `study.v1`: the frozen preregistration |
| `experiments/<name>/results.json` | `StudyReport.write` | `results.v1` |
| `<data>/<preset>/prepare.json` | data preparation | `prepare.v1` |
| `hardware/bench.jsonl` | `nanoscope bench --save` | `bench.v1`, one row per line |

Readers accept the current schema version N and N-1, upgrading older files in memory; a file
with no `schema` key is version 0 (everything written before schemas existed). A file from a
newer nanoscope is refused with a message saying which version wrote it. Shipped baselines are
re-exported whenever the schema changes.

## Kaggle

[`notebooks/kaggle.ipynb`](../notebooks/kaggle.ipynb) runs any study file. Set `STUDY` in the
first cell, and `RUNS_REPO` to a private Hub repo that keeps runs between sessions. Public token data downloads from the Hub, so no tokenization happens on Kaggle.

## Saving and resuming on the Hub

`Study(..., push_to_hub="user/runs")` or `run(..., push_to_hub="user/runs")` mirrors each run,
checkpoints included, to a private model repo at checkpoints (at most every 20 minutes). A fresh
machine with no local checkpoint pulls it and resumes exactly, including data-order and RNG state.

`wandb=True` on `run()` logs live curves to Weights & Biases.
