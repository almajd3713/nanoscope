A speedrun fixes the target and shrinks the budget: here a hundred steps on a CPU, the size of the
shipped GPT-2 (64 wide, 2 layers). The usual way to fail one is to find a change that wins on
seed 0. The check runs three seeds of each model and uses `compare()`, so it only passes when the
difference is outside what seeds alone do.

## Surface

`starter.py` has `Speedrun`, a `Modern` at the baseline's size. Change it:

```bash
nanoscope learn check extend/05-speedrun
```

The check tells you three things in words: whether the size is within 10% of the baseline's, and
how `Speedrun` compares with GPT-2 (the difference and its interval).

### Using it

Change one thing at a time and read the interval each time; two changes at once tell you nothing
about either. Things to try:

- `Modern` already differs from GPT-2 in several ways. Which of them matters in 100 steps? Turn
  them off one at a time (this is an ablation, `honest-ablations/05-capstone`).
- A hundred steps is mostly warmup and the first fast descent. What helps there is not what helps
  at ten thousand.

You may not change the preset or the learning rate: the check trains your class under the
lesson's preset, so the model is the only thing that can win.

## Deep

1. **Does it hold up?** Re-run with five seeds. Is the interval still clear of zero?
2. **Does it hold at 500 steps?** Train both for the preset's full 500. Is the winner the same?
3. **Is it the size?** Measure `nanoscope describe` for both. If you won by 8% more parameters, say so.

## Reading

- **B**: the change you kept and the interval you got.
- **R**: [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), the original speedrun.
- **S**: `docs/research.md` in the repo, on comparing runs.
