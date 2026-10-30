# nanoscope

[![CI](https://github.com/almajd3713/nanoscope/actions/workflows/ci.yml/badge.svg)](https://github.com/almajd3713/nanoscope/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nanoscope-lab)](https://pypi.org/project/nanoscope-lab/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/almajd3713/nanoscope/blob/main/LICENSE)

A web app for learning how small language models work and for testing changes to them properly.
You build a model in lessons, edit it as code or as a graph, train it, and compare it with another
model with confidence intervals instead of a single loss number.

It has three parts that share one code path:

- **Lessons**: 33 of them in 8 paths. Foundations and The modern block build a model from a bigram
  up to a modern transformer block; later paths cover honest ablations, evaluation, scaling,
  interpretability, efficiency and extending the library. Each lesson has a check that says why it
  fails, and passing one unlocks the bigger blocks.
- **A model page** where one file is both code and a graph. Drag a block onto another to swap it and
  the file changes by one line. A model written in plain PyTorch shows the structure its code builds.
- **Experiments** with seeds, paired comparisons, 95% intervals, preregistered studies and baselines
  for the shipped models.

## Installation

With Docker:

```bash
git clone https://github.com/almajd3713/nanoscope
cd nanoscope
./up.sh -d
docker compose logs api | grep login
```

Open the link that grep prints. The app is at `http://127.0.0.1:8765` and trains on one CPU worker.
`./up.sh` also starts a GPU worker when Docker can use an NVIDIA GPU (`./up.sh --cpu` skips it).

Without Docker, on Python 3.10 to 3.13:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: CPU-only torch
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu
```

On Linux, pip's default torch brings about 4 GB of CUDA libraries; the first line skips them when
you have no NVIDIA GPU. `nanoscope serve` serves on `http://127.0.0.1:8000`. The login token lets
whoever has it run code on your machine, so keep the server on localhost or behind an SSH tunnel.
Setup for GPUs, remote use and backups is in
[docs/deploy.md](https://github.com/almajd3713/nanoscope/blob/main/docs/deploy.md).

## Usage

The Learn, Tinker, Research and Extend switch hides controls you don't need yet. It doesn't
change what runs. [docs/gui.md](https://github.com/almajd3713/nanoscope/blob/main/docs/gui.md)
describes each screen. The app calls the same code as Python and the command line:

```python
from nanoscope import compare, run

result = run(MyModel, preset="tinystories-5min")
compare(result, "gpt2")
```

```bash
nanoscope run models/my_model.py:MyModel --seeds 3
nanoscope compare modern gpt2 --preset tinystories-5min
nanoscope --help
```

[docs/research.md](https://github.com/almajd3713/nanoscope/blob/main/docs/research.md) covers
studies and [docs/blocks.md](https://github.com/almajd3713/nanoscope/blob/main/docs/blocks.md)
covers writing your own blocks. All the guides are listed in
[docs/](https://github.com/almajd3713/nanoscope/tree/main/docs).

## Develop

```bash
uv sync --all-extras
make check        # what CI runs
make web          # build the app into the package
pnpm -C web dev   # app with hot reload
```

## Data

Tokens come from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
(CDLA-Sharing-1.0) and [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
(ODC-By 1.0), cached under `~/.nanoscope/data`.

Architecture notes are in
[docs/architecture.md](https://github.com/almajd3713/nanoscope/blob/main/docs/architecture.md).
MIT licensed.
