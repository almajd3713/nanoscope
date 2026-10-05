# nanoscope

See what your language model learns. You write an `nn.Module`; nanoscope handles the
data, the training loop, evaluation, checkpoints and comparison against baselines.

It has two levels that share one core. Learners call `run()`. Researchers write a `Study`
with seeds, budgets, parameter matching and preregistration. The level changes what you
see, never which code runs.

## Learn

```bash
pip install git+https://github.com/almajd3713/nanoscope
```

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

Work through the notebooks in order:

1. [`01-first-model`](notebooks/01-first-model.ipynb): write a bigram model and train it.
2. [`02-gpt2`](notebooks/02-gpt2.ipynb): a real transformer.
3. [`03-modern-block`](notebooks/03-modern-block.ipynb): RoPE, RMSNorm, SwiGLU, GQA, QK-norm, z-loss.
4. [`04-ablations`](notebooks/04-ablations.ipynb): which part matters, with seeds and confidence intervals.

Token data downloads from the Hub (`RedhouaneLazib/nanoscope-tokens`) when a preset has
it, and is tokenized locally otherwise. Everything is cached under `~/.nanoscope/data`.

## Research

See the [research guide](docs/research.md). The research program itself is in
[`docs/project-nanoscope.md`](docs/project-nanoscope.md).

## Command line

```bash
nanoscope presets                                   # available presets
nanoscope run nanoscope/models/gpt2.py:GPT2 --seeds 3
nanoscope compare modern gpt2 --preset tinystories-5min
nanoscope study studies/m1_ablation.py --devices cuda:0
nanoscope report studies/m1_ablation.py             # writes experiments/<name>/
nanoscope bench modern --compile reduce-overhead    # speed, and whether CPU or GPU is the limit
nanoscope status runs                               # what is running and how far along
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
