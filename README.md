# nanoscope

A workbench for learning and researching small language models. You write a PyTorch `nn.Module`;
nanoscope trains it, evaluates it, and tells you whether a difference between two models is bigger
than seed noise.

It has three parts that share one code path:

- **Lessons** that build a model from a bigram up to a modern transformer block. Each lesson has a
  check that says why it fails, and passing one unlocks the bigger blocks.
- **A model page** where one file is both code and a graph. Drag a block onto another to swap it and
  the file changes by one line.
- **Experiments** with seeds, paired comparisons, 95% intervals, preregistered studies and baselines
  for the shipped models.

The app is the main way in. Everything it does is also a Python call and a CLI command.

## Run it

```bash
docker compose up
docker compose logs api | grep login     # open this link once
```

The API listens on `127.0.0.1:8765` with one CPU worker. Optional profiles: `gpu`, `notebook`
(marimo on `:8766`), `editor-lsp`.

Without Docker (Python 3.10 to 3.12):

```bash
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu             # http://127.0.0.1:8000
```

The login token is a remote login: it can save files and queue runs that execute. Keep the server
on loopback, behind an SSH tunnel, or behind an HTTPS proxy. See [docs/deploy.md](docs/deploy.md).

## Use it from Python

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

`run(Model, seeds=5)` trains several seeds. `Study` adds budgets, parameter matching, preregistration
and record mode ([docs/research.md](docs/research.md)). Hooks, presets, optimizers and your own
blocks are covered in [docs/blocks.md](docs/blocks.md).

## Command line

```bash
nanoscope run nanoscope/models/gpt2.py:GPT2 --seeds 3
nanoscope compare modern gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --dry-run     # estimate the cost first
nanoscope status runs
nanoscope describe my_model.py:MyLM                  # shapes, params, FLOPs per module
nanoscope inspect <run> --step 200 --prompt "Once"
nanoscope learn list
```

## Lessons

Eight paths: `foundations`, `modern-block`, `honest-ablations`, `evaluation`, `efficiency`,
`scaling-lite`, `interpretability`, `extend`. Lessons that need hours of GPU also have a CPU variant,
with a compute estimate for each. See [docs/learn.md](docs/learn.md).

## Levels

The switch in the top bar (Learn, Tinker, Research, Extend) only shows or hides controls. It never
changes what runs. Settings can show the CLI equivalent of every screen. The screens are listed in
[docs/gui.md](docs/gui.md).

## Design

Anything that runs your code is a job for a worker; the API process never imports it. State is
plain files (`status.json`, `metrics.jsonl`) that other tools can read. See
[docs/architecture.md](docs/architecture.md) and [docs/server.md](docs/server.md).

## Develop

```bash
uv sync --all-extras
make check        # lint, typecheck, tests: what CI runs
make web          # build the app into the package
pnpm -C web dev   # app with hot reload
```

UI rules are in [docs/design-system.md](docs/design-system.md).

## Data

Tokens come from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By 1.0), cached under `~/.nanoscope/data`.

MIT licensed.
