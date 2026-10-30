# nanoscope: a case study

Draft. nanoscope is a tool for learning how small language models work and for testing changes to them with seeds, paired comparisons and confidence intervals.

## The problem

Most ways to learn how a language model works stop at reading code or watching a loss curve go down. Most comparisons between small model variants rest on one run each, so a difference of 0.01 in loss can be noise. I wanted one tool where a learner builds a model piece by piece and a researcher tests a change properly, on the same code.

## What I built

- A Python library, [`nanoscope-lab`](https://pypi.org/project/nanoscope-lab/) on PyPI, with one training path for every level: `run(Model)` for a first model, presets and seeds for tinkering, `Study` with paired comparisons and 95% intervals for research, and hooks and custom blocks for extending it.
- 33 lessons in 8 paths, from a bigram to a modern transformer block, then honest ablations, evaluation, scaling, interpretability, efficiency and extending the library. Each lesson's check says why it fails, and passing one unlocks the bigger blocks.
- A web app over an HTTP API: the lessons, a model page where one file is both code and a graph, runs, compare, studies and hardware.
- A docker compose setup with the API, CPU and GPU workers and a language server for the editor, in hardened containers.

## How it is built

- One code path. The web app draws what the API returns, and the API calls the same library functions as Python and the command line.
- The API never imports user code. It reads models with Python's `ast`, and anything that runs a model is a job for a worker process.
- Files are the source of truth. Every run writes `status.json` and `metrics.jsonl`, and every file another tool reads has a JSON Schema.
- Real data in checks and screens. Lesson checks are calibrated on real TinyStories runs, and the web tests use recorded API responses.

## Results

### Which parts of a modern block matter at small scale

Each variant below removes one component from the modern block and is resized to the GPT-2 control's parameter count. Every variant trained on 4M tokens of TinyStories with 3 seeds, on one GPU. The numbers are bits per byte on held-out text (lower is better), with the difference from the full modern block and its 95% interval.

| Variant | Bits per byte | Difference from modern [95% CI] | Verdict |
|---|---|---|---|
| gpt2 | 0.988 ± 0.011 | +0.103 [+0.095, +0.110] | worse |
| modern | 0.885 ± 0.007 | (baseline) | |
| no-gqa | 0.880 ± 0.005 | −0.005 [−0.010, −0.001] | better |
| no-qk-norm | 0.899 ± 0.010 | +0.014 [−0.003, +0.030] | within noise |
| no-rmsnorm | 0.886 ± 0.009 | +0.001 [−0.001, +0.003] | within noise |
| no-rope | 1.008 ± 0.009 | +0.123 [+0.114, +0.132] | worse |
| no-swiglu | 0.902 ± 0.008 | +0.017 [+0.006, +0.028] | worse |
| no-z-loss | 0.885 ± 0.002 | +0.000 [−0.005, +0.006] | within noise |

Taking out RoPE and SwiGLU made the model worse. Taking out QK-norm, RMSNorm and z-loss made no difference the intervals can detect at this scale. Taking out GQA made the model better. The GPT-2 control was worse than the modern block, +0.103 bits per byte.

The full report, with every seed's curve, is in [experiments/m1-ablation](https://github.com/almajd3713/nanoscope/tree/main/experiments/m1-ablation).

### Beating GPT-2 in 100 steps

The modern block as shipped loses to GPT-2 after 100 training steps on a CPU. The solution to the speedrun lesson (RoPE, RMSNorm, a GELU MLP 240 wide, plain multi-head attention, no QK-norm and no z-loss) wins by 0.077 bits per byte, 95% interval [0.049, 0.105]. What helps after 1,954 steps is not what helps after 100.

## Engineering

The library is 18,368 lines of Python and the web app 17,990 lines of TypeScript, over 629 commits. Every pull request runs 676 Python tests and 471 web unit tests, plus 15 browser test files against a live docker compose stack. The API has 72 paths.

Three bugs the tests caught:

- The browser tests found that the model page of a plain file opened seven event streams, one per hook. A browser allows six HTTP/1.1 connections per host, so the page's own requests waited forever. Streams are now shared per URL.
- The docker compose test found that a 4-CPU container started 16 torch threads, because `os.cpu_count()` reports the host's cores. Workers now read the CPU affinity and the cgroup quota.
- A release tag on the wrong commit nearly published a release candidate to PyPI. Releases now go through one workflow that checks the changelog, bumps the version and tags `main` itself.

## What is next

- Measure the GPU lessons' compute estimates on real hardware; today they are declared, not measured.
- A path of language models from before attention (MLP, RNN, LSTM).
- Use it with students and learn from where they get stuck.

## Links

- Code: https://github.com/almajd3713/nanoscope
- Package: https://pypi.org/project/nanoscope-lab/
