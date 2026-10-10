How fast a model trains decides how many experiments you get to run. Before you speed anything
up, measure, and find out what is actually slow.

## Surface

`nanoscope bench` trains a model for a few dozen steps and reports the median step time:

```bash
nanoscope bench nanoscope/models/gpt2.py:GPT2 --preset tinystories-5min
```

On an RTX 4070 laptop GPU it printed:

```
GPT2 on cuda
  step time      54.0 ms
  throughput     37,895 tokens/s  (0.36 TFLOPs)
  CPU-bound: the GPU is busy 30% of each step and waits on Python the rest.
```

Three numbers, three functions to write in `starter.py`:

- `tokens_per_second(step_ms, batch_size, context_length)`: a step trains on `batch_size`
  sequences of `context_length` tokens. At 8 x 256 tokens and 54 ms that is about 38,000
  tokens a second.
- `gpu_busy(kernel_ms, step_ms)`: the share of a step the GPU spends running kernels (the profiler
  reports it). Cap it at 1.
- `cpu_bound(gpu_busy_fraction)`: true when that share is under 0.7. The GPU is then idle for
  a third of the step or more, waiting for Python to launch the next kernel.

```bash
nanoscope learn check efficiency/01-bench
```

### What "CPU-bound" means here

A small model gives the GPU very little to do per kernel. Each layer launches several kernels, and
each launch costs a few microseconds of Python and driver time. When the kernels themselves finish
faster than the next launch arrives, the GPU sits idle and a faster GPU would change nothing.
Making the model bigger, or launching fewer kernels, helps. Buying a faster GPU does not.

On a CPU there is nothing to compare against, so `bench` only reports the step time.

## Deep

1. **Change one thing.** Run `bench` with `--set d_model=256`. Does the busy fraction go up? Why does
   a bigger model use the GPU better?
2. **Batch size.** Double `batch_size` through `--set`. Tokens per second should rise less than 2x.
   Where does the time go?
3. **Noise.** Run `bench` three times. How much does the step time move? How many steps did you
   need before you could trust a 5% difference?

## Reading

- **B**: the three functions you wrote.
- **R**: [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (He, 2022): compute, memory and overhead as the three ways to be slow.
- **S**: [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html): how the kernel time `bench` uses is measured.
