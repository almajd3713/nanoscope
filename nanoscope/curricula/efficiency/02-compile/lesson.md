`torch.compile` rewrites your model into fewer, fused kernels. That costs time once, and saves
time on every step after. Whether it is worth it depends on how long the run is.

## Surface

Same model, three ways, on an RTX 4070 laptop GPU (`tinystories-5min`, GPT2):

```bash
nanoscope bench nanoscope/models/gpt2.py:GPT2 --preset tinystories-5min
nanoscope bench nanoscope/models/gpt2.py:GPT2 --preset tinystories-5min --compile
nanoscope bench nanoscope/models/gpt2.py:GPT2 --preset tinystories-5min --compile reduce-overhead
```

| mode | step time | GPU busy |
|---|---|---|
| eager | 54.0 ms | 30% |
| `--compile` | 41.2 ms | 33% |
| `--compile reduce-overhead` | 12.8 ms | 46% |

`reduce-overhead` records the whole step as a CUDA graph and replays it with one launch, which is
exactly what a CPU-bound model needs: 4.2x faster. Default compile fuses kernels but still
launches them one by one, so it helps much less here.

Write three functions in `starter.py`:

- `speedup(eager_ms, compiled_ms)`.
- `break_even_steps(compile_seconds, eager_ms, compiled_ms)`: compile time divided by the time
  saved per step.
- `worth_compiling(steps, compile_seconds, eager_ms, compiled_ms)`: does a run of that many steps
  finish sooner, compile time counted?

```bash
nanoscope learn check efficiency/02-compile
```

### Using it

A 500-step run saves 500 x 41 ms = 21 s with `reduce-overhead`. If compiling took 40 s, you lost
time. A 5,000-step run saves 205 s and wins. Short runs and quick edits are the case where compile
hurts; long studies are the case where it pays. Measure `compile_seconds` yourself: it is the
extra wall time of a compiled run over an eager one for the same few steps.

On a CPU, `--compile` works but gains little for models this small, and `reduce-overhead` needs CUDA.
Try the first two commands with `--device cpu` and compare.

## Deep

1. **Graph breaks.** Add a Python `print` inside a model's `forward` and compile it. What happens to
   the speedup? (`TORCH_LOGS=graph_breaks` says where.)
2. **Recompiles.** Change the batch size between steps. How many times does it compile?
3. **Does compile change the result?** Train the same seed eager and compiled and compare the
   final loss. Compare the difference to your seed noise from `honest-ablations/01-seed-noise`.

## Reading

- **B**: the three functions you wrote.
- **R**: [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).
- **S**: [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/): what `reduce-overhead` does.
