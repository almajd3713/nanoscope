An induction head completes a pattern: having seen `... A B ... A`, it attends to `B` and the
model predicts `B` next. It is a small circuit (a previous-token head in one layer, feeding a
head in a later layer), and it tends to appear suddenly during training rather than gradually.
You will measure it, then find out when it shows up in a run of your own.

## Surface

Feed the model `half` random tokens and then the same tokens again. In the second copy every
token has an earlier twin, and the token after that twin is the right guess. An induction head
puts its attention there: from position `i` (in the second copy) to position `i - half + 1`.

`nanoscope inspect` returns each head's attention as causal rows (row `i` has `i + 1` numbers).
Write three functions in `starter.py`:

- `induction_score(attn, half)`: the mean attention from positions `half .. 2*half-1` to
  `i - half + 1`.
- `previous_token_score(attn)`: the mean attention from each position to the one before it. The
  first layer's helper head scores high here.
- `first_step_above(steps, scores, threshold)`: the first checkpoint step from which a score
  stays at or above the threshold, or -1.

```bash
nanoscope learn check interpretability/01-induction
```

### Using it

Train a small model (2 layers or more: one layer cannot form the circuit) and keep checkpoints
spaced on a log scale, so the early steps are not skipped:

```bash
nanoscope run nanoscope/models/modern.py:Modern --preset tinystories-5min \
  --checkpoint-steps 10,30,100,300,1000
nanoscope inspect <run> --step 100
```

Pick a repeated random sequence as the prompt, score every head at every saved step, and plot
`first_step_above` for the best head. Compare the step where the loss bends with the step where
the score jumps.

On a GPU, use the `fineweb-edu` preset: a bigger model trained longer has more steps to show
the jump in.

## Deep

1. **Random tokens, not text.** Why does the repeated-random-tokens prompt isolate induction?
   What would a head that only memorised common bigrams score on it?
2. **Which layer?** Does the best induction head sit after the best previous-token head, as the
   circuit says?
3. **Seeds.** Train two seeds. Does the jump happen at the same step? Use your seed noise from
   `honest-ablations/01-seed-noise` to say what counts as "the same".

## Reading

- **B**: the three functions you wrote.
- **R**: [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Olsson et al., 2022).
- **S**: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
