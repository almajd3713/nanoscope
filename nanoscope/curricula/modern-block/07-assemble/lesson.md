Six lessons built six parts. This one assembles them and asks the question the whole path is
for: **does it actually help, by more than chance?**

## Surface

`starter.py` is a GPT-2-style model written with the library blocks: LayerNorm, learned
position embeddings, plain attention, a GELU MLP. Turn it into the modern one by swapping each
part for the one you built (the library now lets you import them):

| GPT-2 | Modern |
|---|---|
| `LayerNorm()` | `RMSNorm()` (also as the final norm) |
| learned positions (`pos_emb=LearnedPosition()`) | `RoPE()` inside attention, and drop `pos_emb` |
| `Attention(n_heads=4)` | `Attention(n_heads=4, n_kv_heads=2, pos=RoPE(), qk_norm=True)` |
| `GELUMLP()` | `SwiGLU()` |
| no extra loss | `z_loss=1e-4` on the `Decoder` |

The check then trains `MyModern` and the shipped `GPT2` on your CPU with three seeds each and
compares them. The verdict must be **better**: the 95% interval of the difference has to lie
entirely on the good side of zero. `forbid` also checks you only use blocks you have unlocked.

Read the table that is printed. It shows the difference and its interval, not just which
number is smaller: a difference that the interval cannot tell from zero is "within noise", and
that is a result too.

## Deep

Use five seeds (`seeds=5`) and the GPU variant if you have one (`tinystories-30min`). Then do
what a real ablation does: remove *one* component at a time (`Modern(rope=False)`,
`Modern(qk_norm=False)`, ...) and see which removals you can actually detect. Some will be
within noise at this size. Say so, with the interval.

## Reading

- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023), architecture.
- Dubey et al., "The Llama 3 Herd of Models" (2024).
- The M1 milestone in `docs/project-nanoscope.md`: this is its small version.
