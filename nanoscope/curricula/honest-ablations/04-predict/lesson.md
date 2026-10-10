Until now you have run an experiment and then read the result. Reading a result after the fact
is easy to fool yourself with: whatever comes out, a reason appears. This lesson reverses the
order. You commit to a prediction in writing first, run the experiment second, and compare.

## Surface

**Why predict at all.** There are two quiet ways to cheat in research, both unintentional.

- *Hindsight*: after the numbers arrive, almost any outcome feels like something you half
  expected. A prediction written before is the only record of what you really expected.
- *The forking paths*: if you look at the result and then choose which comparison, which seeds
  or which metric to report, you will find something. A plan written first removes those
  choices.

Scientists call this **preregistration**. nanoscope's `mode="record"` does it for whole studies
(a git commit of the plan before the first run, `Study.predict(...)` frozen once runs start).
This lesson is the same habit at lesson size, and it is graded on the habit, not on being
right.

### The experiment

`starter.py` has two models. `WithRoPE` is the modern model as shipped. `NoRoPE` is the same
model with its rotary position embeddings taken out; you edit it. The check trains both on 5
paired seeds and reports the difference in `val_bpb` (negative: `WithRoPE` is better).

**Do this before anything else.** Not even `learn check`, because every check counts as having
run the experiment. Think, then write:

```bash
nanoscope learn predict honest-ablations/04-predict \
    --verdict <better|worse|within noise> --low <number> --high <number> \
    --note "why you expect it"
```

Three parts:

- `--verdict`: `better`, `worse` or `within noise` for `WithRoPE` against `NoRoPE`.
- `--low`, `--high`: the range you expect the difference to fall in. This is a bet. A range
  from -5 to +5 is certain to contain the answer and says nothing, so the check refuses a range
  far wider than the experiment's own uncertainty.
- `--note`: why. The reason is the point; you can reread it later.

Then edit `NoRoPE` (change `rope=True` to `rope=False`) and run

```bash
nanoscope learn check honest-ablations/04-predict
```

### How to make a good prediction

1. **Start from the noise floor** (lesson 1). With 5 seeds a difference smaller than a few
   hundredths of a bit is invisible. If you think the effect is small, `within noise` is a real
   prediction.
2. **Reason about the mechanism.** Attention with no position information sees the context as
   a bag of tokens. A causal mask still lets a model infer a little about order, but only a
   little. How much do stories depend on word order? Is this a mild loss or a large one?
3. **Give yourself a range, not a point.** Something like "between a quarter and double my best
   guess" is honest. The check wants the true difference inside your range and your range not
   absurdly wide.
4. **Say what would change your mind** in the note.

### Being wrong counts

`must_hit` is off for this lesson. The check passes if the prediction was recorded first and
is specific. If the result is outside your range the message says so and the lesson still
counts: a wrong prediction cannot be redone, and what you learn from the miss is worth more
than the pass. Read the message. If you missed, ask whether the effect was bigger than you
reasoned or you misjudged the noise.

## Deep

1. **Predict the interval, not just the mean.** The result has its own 95% interval. Write a
   prediction for both the difference and for how wide the interval will be at 5 seeds.
2. **Calibrate.** Make five predictions for five other ablations you have not run (QK-norm,
   z-loss, GQA, SwiGLU, RMSNorm) in one file before running any of them. How often does the
   result fall inside your range? If you pick 80% ranges, you should be right about four times
   in five. Most people are right about half the time.
3. **Record mode.** Put the same experiment in a `Study(mode="record")` with
   `study.predict("with-rope", val_bpb=...)`. What does the record add that a text file does
   not?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [The preregistration revolution](https://www.pnas.org/doi/10.1073/pnas.1708274114) (Nosek et al., 2018, *PNAS*): the argument for writing the plan first, in plain terms.
- **R**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/pdf/2104.09864) (Su et al., 2021): what you are predicting the value of.
- **S**: [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/pdf/2305.19466) (Kazemnejad et al., 2023): a careful comparison, including having no position encoding at all.
- **K**: `Study.predict` and `mode="record"` in `docs/research.md`.
