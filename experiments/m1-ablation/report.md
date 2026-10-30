# Study: m1-ablation

- mode: explore
- preset: tinystories-5min, seeds: 0, 1, 2
- budget: Tokens(4e+06) per run

## Results

```
bits per byte on the first 200 validation documents of roneneldan/TinyStories (lower is better)

model       seeds  non-emb params  tokens  val_bpb        Δ vs modern
----------  -----  --------------  ------  -------------  -----------------------  ------------
gpt2        3      0.79M           4.0M    0.988 ± 0.011  +0.103 [+0.095, +0.110]  worse
modern      3      0.79M           4.0M    0.885 ± 0.007  (baseline)
no-gqa      3      0.79M           4.0M    0.880 ± 0.005  −0.005 [−0.010, −0.001]  better
no-qk-norm  3      0.79M           4.0M    0.899 ± 0.010  +0.014 [−0.003, +0.030]  within noise
no-rmsnorm  3      0.79M           4.0M    0.886 ± 0.009  +0.001 [−0.001, +0.003]  within noise
no-rope     3      0.79M           4.0M    1.008 ± 0.009  +0.123 [+0.114, +0.132]  worse
no-swiglu   3      0.80M           4.0M    0.902 ± 0.008  +0.017 [+0.006, +0.028]  worse
no-z-loss   3      0.79M           4.0M    0.885 ± 0.002  +0.000 [−0.005, +0.006]  within noise
note: 7 variants are compared with one baseline; each 95% interval is separate, so the chance that at least one excludes zero by luck alone is about 30%. Treat a lone small win with care.
```
