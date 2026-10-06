The softmax turns scores (logits) into probabilities by dividing by `Z = sum(exp(logits))`.
Only differences between logits matter, so nothing stops all of them drifting upward during
training, which costs precision (bf16 especially) and can destabilise it. The **z-loss**
(from PaLM) adds a tiny penalty on `log Z`, pulling the normaliser toward 1.

## Surface

For logits of shape `(batch, time, vocab)`:

```
log_z  = logsumexp(logits, dim=-1)           one number per position
z_loss = coefficient * mean(log_z ** 2)
```

Write `ZLoss(coefficient=1e-4)`: `forward(logits)` returns that scalar. A model that has a
z-loss returns `(logits, aux_loss)` instead of just `logits`; the trainer adds `aux_loss` to
the cross-entropy. Use `torch.logsumexp` (it is numerically safe; `log(sum(exp(...)))` is
not). Passing **unlocks the z-loss feature** (`z_loss=...` on your own `Decoder`).

## Deep

Compute `log Z` for the logits of an untrained model and of a trained one. Why is a *small*
coefficient (1e-4) enough, and what would a large one do to the model's confidence?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [OLMo 2](https://arxiv.org/pdf/2501.00656) (2024): stability engineering.
- **R**: [ST-MoE](https://arxiv.org/pdf/2202.08906) (Zoph et al., 2022): the z-loss.
- **S**: Chowdhery et al., "PaLM" (2022), where the auxiliary loss on log Z appears.
