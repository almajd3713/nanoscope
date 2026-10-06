Attention scores are dot products of queries and keys. If those vectors grow during training,
the scores grow, the softmax saturates to one-hot, and gradients vanish: training becomes
unstable, especially for large models at high learning rates. **QK-norm** fixes it at the
source: normalise every query and key vector before the dot product.

## Surface

In your attention, after splitting heads and before the scores, apply an RMSNorm over the head
dimension to `q` and to `k`, each with its own learned scale:

```
q = q_norm(q)      k = k_norm(k)      scores = q @ k^T / sqrt(head_dim)
```

`starter.py` gives you `RMSNorm` (your last lesson, finished) and multi-head attention. Add
`self.q_norm = RMSNorm(head_dim)` and `self.k_norm = RMSNorm(head_dim)` and use them. The check
randomises all the weights, including the two scales, and compares you with a reference.
Passing **unlocks the QK-norm feature** (`qk_norm=True` in your own `Attention`).

## Deep

After normalising, every query and key has the same length, so a score can only be large if
the two *point the same way*. What does that do to the largest possible softmax weight, and
what role does the learned scale play?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [OLMo 2](https://arxiv.org/pdf/2501.00656) (2024): read it for stability engineering (QK-norm, z-loss, why runs diverge).
- **S**: Henry et al., "Query-Key Normalization for Transformers" (2020).
- **S**: Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters" (2023): QK-norm at scale.
