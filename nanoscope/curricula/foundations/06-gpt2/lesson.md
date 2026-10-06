You have every part. GPT-2 is the parts in a line: look up each token's embedding, add an
embedding for its *position* (attention alone has no idea of order), pass the result through a
stack of identical blocks, normalise once more, and turn each vector into a score per token
with the same table that did the embedding.

## Surface

`starter.py` gives you `MyBlock` (your lesson 5, finished) and `init_weights`, which sets the
weights the way GPT-2 does. Write `MyGPT2`:

1. `tok_emb`: `nn.Embedding(vocab_size, d_model)`; `pos_emb`:
   `nn.Embedding(context_length, d_model)`.
2. `blocks`: an `nn.ModuleList` of `n_layers` separate `MyBlock`s (not one block reused: each
   layer has its own weights).
3. `ln_f`: a final `nn.LayerNorm`.
4. `head`: `nn.Linear(d_model, vocab_size, bias=False)` whose weight *is* `tok_emb.weight`
   (`self.head.weight = self.tok_emb.weight`). Sharing this one table means a token's embedding
   and its "output direction" are the same thing.
5. call `init_weights(self, n_layers)` at the end of `__init__`.

`forward`: `x = tok_emb(idx) + pos_emb(arange(T))`, then each block, then `ln_f`, then `head`.

The check trains your model on your CPU (a few minutes) and compares it with the shipped GPT-2
baseline: your result should land inside the range its three seeds span. Passing **unlocks
`Decoder`**, the library's stack, so models you compose from now on can use it.

## Deep

Run three seeds and look at the spread; that spread is what "reproduces" means. Then change one
thing (no tied weights, no position embedding, 8 layers) and ask the question every later lesson
asks: is the difference bigger than the noise?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **R**: [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019): the decoder-only, zero-shot-as-text framing.
- **B**: The Illustrated / Annotated Transformer, and Karpathy's nanoGPT.
- **R**: [T5](https://arxiv.org/pdf/1910.10683) (2019): read it for the *ablation methodology*, not the model: this is how a systematic architecture study is run.
- **S**: [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361) (Kaplan et al., 2020): why one more layer is not the point.
