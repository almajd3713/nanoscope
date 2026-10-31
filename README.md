# nanoscope

[![CI](https://github.com/almajd3713/nanoscope/actions/workflows/ci.yml/badge.svg)](https://github.com/almajd3713/nanoscope/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nanoscope-lab)](https://pypi.org/project/nanoscope-lab/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/almajd3713/nanoscope/blob/main/LICENSE)

Learn how small language models work by building one, then test your changes with seeds and
confidence intervals instead of a single loss number.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/almajd3713/nanoscope/main/docs/images/model-swap-dark.png">
  <img alt="The model page: block palette, code, graph and inspector" src="https://raw.githubusercontent.com/almajd3713/nanoscope/main/docs/images/model-swap-light.png">
</picture>

## Features

- **Lessons** that take you from a bigram model to a modern transformer block, then through
  ablations, evaluation, scaling and interpretability. Each one has a check that says what is wrong.
- **Code and graph in one file.** Edit the model in Python or drag blocks around the graph.
- **Comparisons you can trust.** Several seeds, paired differences, 95% intervals, preregistered
  studies.
- **Runs on a laptop.** Every lesson has a CPU version; a GPU helps with the larger studies.

## Quick start

```bash
git clone https://github.com/almajd3713/nanoscope
cd nanoscope
./up.sh -d
docker compose logs api | grep login   # open this link
```

Or without Docker:

```bash
pip install "nanoscope-lab[server]"
nanoscope serve --worker cpu           # http://127.0.0.1:8000
```

## Python and the command line

The app is a front end to a library you can use directly:

```python
from nanoscope import compare, run

result = run(MyModel, preset="tinystories-5min")
compare(result, "gpt2")
```

```bash
nanoscope run models/my_model.py:MyModel --seeds 3
```

## Documentation

- [Lessons](https://github.com/almajd3713/nanoscope/blob/main/docs/learn.md)
- [Research: studies and comparisons](https://github.com/almajd3713/nanoscope/blob/main/docs/research.md)
- [Writing blocks](https://github.com/almajd3713/nanoscope/blob/main/docs/blocks.md)
- [Deployment: GPUs, remote use, backups](https://github.com/almajd3713/nanoscope/blob/main/docs/deploy.md)
- [All docs](https://github.com/almajd3713/nanoscope/tree/main/docs)

## Development

```bash
uv sync --all-extras
make check
```

## License

MIT. The training data comes from
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
