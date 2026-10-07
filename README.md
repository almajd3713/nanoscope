# nanoscope

See what your language model learns. You write an `nn.Module`; nanoscope handles the
data, the training loop, evaluation, checkpoints and comparison against baselines.

It has four levels that share one core. The level changes what you see, never which code runs.

0. **Learn**: `run(Model)` with defaults.
1. **Tinker**: change a setting, or train several `seeds=`, and compare.
2. **Research**: a `Study` with seeds, budgets, parameter matching and preregistration.
3. **Extend**: hooks, presets, optimizers and your own blocks.

You can use it as a library, from the command line, through an HTTP API (`nanoscope serve`), or
as a docker-compose stack with lessons and notebooks.

## Install

```bash
pip install nanoscope-lab             # the library
pip install "nanoscope-lab[server]"   # plus the HTTP API (`nanoscope serve`)
```

To run it as a service (API, a worker, and optional notebooks and GPU worker), see
[Run it as a service](#run-it-as-a-service).

From a clone, for development: `uv sync --all-extras`.

## Learn

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
result.plot()
compare(result, "gpt2")                          # against a shipped 3-seed baseline
```

Work through the notebooks in order (marimo notebooks: `pip install "nanoscope-lab[notebook]"`, then
`marimo edit notebooks/marimo/01_first_model.py`, or `docker compose --profile notebook up`):

1. [`01_first_model`](notebooks/marimo/01_first_model.py): write a bigram model and train it.
2. [`02_gpt2`](notebooks/marimo/02_gpt2.py): a real transformer.
3. [`03_modern_block`](notebooks/marimo/03_modern_block.py): RoPE, RMSNorm, SwiGLU, GQA, QK-norm, z-loss.
4. [`04_ablations`](notebooks/marimo/04_ablations.py): which part matters, with seeds and confidence intervals.

Token data downloads from the Hub (`RedhouaneLazib/nanoscope-tokens`) when a preset has
it, and is tokenized locally otherwise. Everything is cached under `~/.nanoscope/data`.

## Research

See the [research guide](docs/research.md). The research program itself is in
[`docs/project-nanoscope.md`](docs/project-nanoscope.md).

## Build models from blocks

`GPT2` and `Modern` are short compositions of the blocks in `nanoscope.blocks`, and so can your
own models. See [docs/blocks.md](docs/blocks.md).

## Guided lessons

Guided paths build the models step by step, with checks that say why. See
[docs/learn.md](docs/learn.md): `nanoscope learn list`, `learn start`, `learn check`.

## Run it as a service

```bash
nanoscope serve --worker cpu      # the HTTP API on 127.0.0.1:8000 and one local worker
docker compose up                 # the same in containers: API on 127.0.0.1:8765, one CPU worker
docker compose --profile gpu up   # plus an NVIDIA GPU worker
docker compose --profile notebook up   # plus marimo notebooks on 127.0.0.1:8766
```

Anything that runs your code is a job for a worker; the API never runs it. See
[docs/server.md](docs/server.md) for the API and [docs/deploy.md](docs/deploy.md) for compose,
remote sign-in, backups and GPUs. The token is a remote login: keep it on loopback or behind an
SSH tunnel or an HTTPS proxy.

## Command line

```bash
nanoscope presets                                   # available presets
nanoscope run nanoscope/models/gpt2.py:GPT2 --seeds 3
nanoscope compare modern gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --devices cuda:0
nanoscope report studies/m1_ablation.py             # writes experiments/<name>/
nanoscope bench modern --compile reduce-overhead    # speed, and whether CPU or GPU is the limit
nanoscope status runs                               # what is running and how far along
nanoscope describe nanoscope/models/modern.py:Modern  # shapes, params, FLOPs, memory per module
nanoscope graph my_model.py                         # a model file's architecture, without running it
nanoscope blocks                                    # the blocks models are composed from
nanoscope prepare-data tinystories-5min             # download or tokenize now
nanoscope publish-data tinystories-5min user/repo   # upload tokens to a Hub dataset
```

## Develop

```bash
uv sync --all-extras
make test       # offline tests
make lint
make typecheck
make check      # lint, typecheck, then test: what CI runs
```

## Data credits

Tokens are derived from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By 1.0).
