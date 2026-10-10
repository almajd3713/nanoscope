A dense MLP runs every parameter on every token. A mixture of experts keeps several MLPs, the
*experts*, and a small *router* that picks `top_k` of them per token. The model has `experts`
times the MLP parameters, but each token only pays for `top_k` of them: more capacity for the
same work per token.

## Surface

The picture, per token: the router scores all experts, you keep the best `top_k`, rescale their
scores to sum to 1, run only those experts, and add their outputs weighted by the rescaled scores.

Build `MyMoE` in `starter.py` from the `Expert` (a SwiGLU you already wrote) and a router named
`gate`. Only `forward` is left to write, in the four steps the starter lists. The names matter:
the check reads `gate.weight` and `experts.N.w1.weight`, `.w3.weight` and `.proj.weight`.

```bash
nanoscope learn check extend/04-moe
```

It compares your output with a reference that handles one token and one expert at a time, in
plain loops, on random weights.

Passing unlocks the shipped `MoE` block, which is the same thing plus a load-balancing loss.

### Using it

Compare a dense model with one whose MLP is `MoE(experts=4, top_k=2)` on `tinystories-5min`
(`nanoscope describe` shows the parameter count and the FLOPs per token for each):

- total parameters: the MoE model has far more;
- FLOPs per token: `top_k` of 4 experts costs about as much as a dense MLP of `top_k` experts' size.

Then train both and ask `compare()` whether the extra parameters bought anything at this size.
Do not be surprised by "within noise": a small model on a small dataset has little to use the
capacity for.

On a GPU, use 8 experts and `tinystories-30min`.

## Deep

1. **Load.** Count how many tokens each expert gets in a batch. What does an untrained router do?
   Why does the shipped block add a balancing loss?
2. **Rescaling.** Skip step 2 (use the raw probabilities). Which check fails, and by how much?
3. **Dropped tokens.** Real MoE systems give each expert a fixed capacity and drop the overflow.
   What would that change in this code?

## Reading

- **B**: the forward pass you wrote.
- **R**: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021), for the balancing loss.
- **S**: [Mixtral of Experts](https://arxiv.org/abs/2401.04088): top-2 routing with the rescaling used here.
