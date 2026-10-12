# Study: toy

- mode: record
- preset: test-tiny, seeds: 0, 1, 2
- code: commit fac364bc5c
- preregistered: studies/toy.toml committed in fac364bc5c at 2026-10-12T14:17:01+01:00; first run started 2026-10-12T13:17:01+00:00
- preregistration committed via nanoscope

## Results

```
bits per byte on the first 10 validation documents of fake/stories (lower is better)

model  seeds  non-emb params  tokens  val_bpb        Δ vs small
-----  -----  --------------  ------  -------------  -----------------------  ------
small  3      0.00M           0.0M    2.732 ± 0.078  (baseline)
wide   3      0.01M           0.0M    1.716 ± 0.158  −1.016 [−1.148, −0.883]  better
```

## Predictions

| variant | metric | predicted | actual (95% CI) | error |
|---|---|---|---|---|
| wide | val_bpb | 1.200 | 1.716 [1.558, 1.873] | +43.0% |
