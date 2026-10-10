A transformer's last layer turns the residual stream into logits. The logit lens applies that
same final step to the stream after every earlier layer, which shows what the model would guess
if it stopped there. Early layers guess crudely; the answer sharpens as layers add to it.

## Surface

`nanoscope inspect <run> --prompt "..."` prints the lens: for each layer, the top tokens at the
last position. Write three functions in `starter.py` that turn those readings into numbers:

- `top_token(logits)`: the index of the largest logit.
- `rank_of(logits, target)`: how many tokens outrank `target` (0 means it is the guess).
- `first_settled_layer(layer_logits, target)`: the first layer whose guess is `target` and
  stays `target` at all later layers, or -1.

```bash
nanoscope learn check interpretability/02-logit-lens
```

### Using it

Use the run and checkpoint from `01-induction`. Take a prompt where the model should copy
(the repeated random sequence) and one where it should not, and compare where each settles.
Do it at an early checkpoint and a late one.

## Deep

1. **Rank, not just top-1.** Plot `rank_of` against layer. Does the target climb steadily or jump?
2. **Does the lens lie?** Early layers were never trained to be read this way. What would a
   tuned lens fix?
3. **Settled early.** If a token settles by layer 1 of 4, what do the later layers do?

## Reading

- **B**: the three functions you wrote.
- **R**: [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) (nostalgebraist, 2020).
- **S**: [Eliciting Latent Predictions with the Tuned Lens](https://arxiv.org/abs/2303.08112).
