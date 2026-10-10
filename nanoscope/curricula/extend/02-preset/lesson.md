A preset is a named bundle of everything a run needs besides the model: dataset, tokenizer,
context length, steps, batch size, learning rate and its schedule, how often to evaluate. A run's
name hashes it, so two runs with the same preset are comparable and two with different ones are not.

## Surface

The quickest way to a new preset is to start from a shipped one and change what you mean to:

```python
register_preset(base.override(name="quick-smoke", max_steps=60))
```

Almost. The schedule fields depend on each other, and `override` changes only the one you name:

- `warmup_steps` is 50 in the base. In a 60-step run that is most of the run spent warming up.
- `eval_interval` is 50. It is fine at 60 steps, but a 30-step run would never evaluate and
  report no loss.
- `sample_interval` and `checkpoint_interval` have the same shape of problem.
- The data fields (`dataset`, `tokenizer`, `vocab_size`, `context_length`, `train_docs`,
  `hub_data`) must stay as they are. Change one and the cached tokens no longer apply, and runs
  are no longer comparable with `tinystories-5min`.

In `starter.py`, register `quick-smoke`: 100 steps at most, warmup at most a quarter of the run,
evaluating at least once.

```bash
nanoscope learn check extend/02-preset
```

### Using it

A preset registered in a file is available to `nanoscope run --preset quick-smoke` when that file
is imported (keep it in the workspace next to your models). Run a bigram with it and read the
eval line.

## Deep

1. **Is it comparable?** A run with `quick-smoke` and one with `tinystories-5min` have different
   names. What would a `compare()` between them mean?
2. **The learning rate.** At 60 steps, is `3e-3` still right? How would you find out without
   guessing?
3. **Where else.** What other fields hide a dependency on `max_steps`?

## Reading

- **B**: the preset you registered.
- **R**: `nanoscope/presets.py`, the shipped presets.
- **S**: `docs/blocks.md`, "registering" parts, for the same pattern with blocks.
