Every new optimizer paper shows a win over AdamW. Whether you see it depends on a choice the
paper's table hides: the learning rate. The comparison here is the honest version of that.

## Surface

Muon updates each weight matrix with an orthogonalised momentum (see `nanoscope/optim/muon.py`);
everything else (norm gains) still uses AdamW. In `starter.py` there is a small model, `Tiny`, and
a factory `make_muon(param_groups, preset)` that builds the optimizer at `LR_SCALE` times the
preset's learning rate. The experiment trains `Tiny` with that factory (A) and with plain AdamW (B)
for 200 steps over several seeds, and asks `compare()` for the verdict on `val_bpb`.

Your job is one number, `LR_SCALE`, and a note on how you chose it:

```bash
nanoscope learn check extend/03-muon
```

At `LR_SCALE = 1.0` the check will probably tell you the two are within noise. That is a true
answer to a question nobody asked: "is Muon, at a rate tuned for a different optimizer, better?"

### Using it

Find a better rate by looking, not guessing: run the same comparison at 0.1, 0.3, 1, 2 and 4, and
read the interval each time. It is a sweep, so it is cheap to do with a `Study`. Two things to notice:

- The curve is not flat. Too small is far worse than AdamW, not just slightly worse.
- You tuned Muon's rate and not AdamW's. Whatever lead you find is partly that.

## Deep

1. **Fair.** Sweep AdamW's rate too (0.3x to 3x of the preset). Does Muon's lead survive?
2. **Size.** This model has about 350k parameters and trains for 200 steps. Papers report Muon
   gains at much larger scales. What would you need to run to say anything about that?
3. **Cost.** The orthogonalisation costs extra FLOPs per step. Compare at equal wall-clock time
   with `nanoscope bench`, not at equal steps.

## Reading

- **B**: your sweep and the interval at the rate you chose.
- **R**: [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) (Keller Jordan).
- **S**: [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02533): why the baseline's tuning matters.
