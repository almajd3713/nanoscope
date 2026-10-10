# nanoscope

A tool for learning and researching small language models. You write an `nn.Module`; nanoscope
handles data, training, evaluation, checkpoints and comparison against baselines, and tells you
whether a difference is bigger than the noise.

The main way to use it is a web app: lessons, a model editor with a drag-and-drop graph, live runs,
and comparisons with confidence intervals. Everything the app does is also available from Python and
the command line.

## Start

```bash
docker compose up
docker compose logs api | grep login    # open this link
```

That starts the API on `127.0.0.1:8765` and one CPU worker. First run asks whether blocks unlock as
you pass lessons (guided) or are all available (open). Pick a lesson, press Start, train, run the check.

Without Docker:

```bash
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu            # http://127.0.0.1:8000
```

Optional pieces: `--profile gpu` (NVIDIA worker), `--profile notebook` (marimo on `:8766`),
`--profile editor-lsp` (language server for the code editor). Remote sign-in, backups and GPU setup
are in [docs/deploy.md](docs/deploy.md). The login token is a remote login: keep the server on
loopback, or behind an SSH tunnel or an HTTPS proxy.

## What the app does

| Screen | Use it to |
|---|---|
| Lessons | Work through paths. Each lesson has a check that says why it fails. Passing unlocks bigger blocks. |
| Model | Edit a model file as a graph, as code, or both. Drag a block onto another to swap it. |
| Run | Watch a run live against the shipped baseline's range, read samples, stop, resume, duplicate. |
| Compare | See a verdict per model with a 95% interval, a forest plot and every seed's curve. |
| Studies | Build a multi-seed study, see its estimated cost, preregister it with a git commit, export an ablation card. |
| Inspect | Attention maps, logit lens and one head across checkpoints, for a trained run. |
| Hardware | Devices, speed benchmarks, and whether the CPU or the GPU is the limit. |
| Workspace, Authoring | Browse your files and check lessons you wrote (Extend level). |

The level switch (Learn, Tinker, Research, Extend) only hides or shows controls. It never changes
what runs. Each screen can show its command-line equivalent; turn that on in Settings.
[docs/gui.md](docs/gui.md) lists every screen.

## Lessons

Paths build models step by step, from a bigram to a modern transformer block (`foundations`,
`modern-block`). Lessons that need hours of GPU
also ship a CPU variant, with estimates. See [docs/learn.md](docs/learn.md).

## The library

The same code, without the app:

```python
import torch.nn as nn
from nanoscope import compare, run

class Bigram(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        return self.head(self.token_embedding(idx))

result = run(Bigram, preset="tinystories-5min")   # about a minute on a laptop CPU
compare(result, "gpt2")                           # against a shipped 3-seed baseline
```

- **Tinker:** `run(Model, seeds=5)`; `init_seed=` and `data_seed=` split the seed into weights and batch order.
- **Research:** `Study` with seeds, budgets, parameter matching, preregistration and record mode.
  See [docs/research.md](docs/research.md).
- **Extend:** hooks, presets, optimizers (`run(optimizer=...)`, Muon included) and your own blocks.
  `GPT2` and `Modern` are short compositions of `nanoscope.blocks`; see [docs/blocks.md](docs/blocks.md).

Four marimo notebooks in `notebooks/marimo/` cover the same ground (`pip install "nanoscope-lab[notebook]"`).

## Command line

```bash
nanoscope run nanoscope/models/gpt2.py:GPT2 --seeds 3
nanoscope compare modern gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --dry-run     # cost estimate first
nanoscope status runs                                # what is running, how far along
nanoscope describe my_model.py:MyLM                  # shapes, params, FLOPs per module
nanoscope inspect <run> --step 200 --prompt "Once"
nanoscope learn list
nanoscope presets
```

## How it is built

Anything that runs your code is a job for a worker; the API never imports it. State lives in plain
files (`status.json`, `metrics.jsonl`) that any tool can read. Details: [docs/server.md](docs/server.md),
[docs/architecture.md](docs/architecture.md).

## Develop

```bash
uv sync --all-extras
make check      # lint, typecheck, tests: what CI runs
make web        # build the app into the package
pnpm -C web dev # app with hot reload, proxying /api
```

The design rules for the UI are in [docs/design-system.md](docs/design-system.md).

## Data credits

Tokens are derived from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By 1.0). Token data is cached under `~/.nanoscope/data`.

MIT licensed.
