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

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [LLaMA](https://arxiv.org/pdf/2302.13971) (2023) and [Mistral 7B](https://arxiv.org/pdf/2310.06825) (2023): the convergent recipe you just built.
- **R**: [OLMo 2](https://arxiv.org/pdf/2501.00656) (2024).
- **R**: [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) (2024): the most complete public recipe (MLA, MoE, FP8, MTP).
- **R**: [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/pdf/2404.10102) (Besiroglu et al., 2024): scaling laws are curve fits with error bars
- **R**: [T5](https://arxiv.org/pdf/1910.10683) (2019): ablation methodology.
- **S**: [FineWeb](https://arxiv.org/pdf/2406.17557) (2024): read the ablation methodology for how to evaluate cheaply.
- **K**: The M1 milestone in `docs/project-nanoscope.md`: this lesson is its small version.
