# Study: m1-ablation

- mode: explore
- preset: tinystories-5min, seeds: 0, 1, 2
- budget: Tokens(4e+06) per run

## Results

```
bits per byte on the first 200 validation documents of roneneldan/TinyStories (lower is better)

model       seeds  params  tokens  val_bpb        Δ vs modern
----------  -----  ------  ------  -------------  -----------------------  ------------
gpt2        3      1.35M   4.0M    0.987 ± 0.012  +0.103 [+0.086, +0.120]  worse
modern      3      1.31M   4.0M    0.885 ± 0.009  (baseline)
no-rope     3      1.34M   4.0M    1.011 ± 0.019  +0.126 [+0.105, +0.148]  worse
no-swiglu   3      1.32M   4.0M    0.902 ± 0.008  +0.018 [+0.006, +0.029]  worse
no-rmsnorm  3      1.31M   4.0M    0.886 ± 0.006  +0.002 [−0.002, +0.005]  within noise
no-qk-norm  3      1.31M   4.0M    0.897 ± 0.008  +0.013 [−0.004, +0.030]  within noise
no-gqa      3      1.32M   4.0M    0.882 ± 0.006  −0.003 [−0.012, +0.006]  within noise
no-z-loss   3      1.31M   4.0M    0.886 ± 0.002  +0.001 [−0.007, +0.009]  within noise
```
