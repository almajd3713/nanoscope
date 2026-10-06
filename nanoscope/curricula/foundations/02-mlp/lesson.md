The bigram sees one token. "Once upon a" is far more predictable than "a" alone, so let the
model look at the last few tokens: put their embeddings side by side, and let a small neural
network (a **multi-layer perceptron**, MLP) decide what comes next.

## Surface

`starter.py` gives you `window(emb, k)`, which turns embeddings of shape
`(batch, time, d)` into `(batch, time, k * d)`: for each position, the embeddings of the last
`k` tokens side by side (zeros before the start of the text).

You write `MyMLP`:

1. embed the ids,
2. build windows of `window_size` tokens,
3. pass them through a hidden layer with a non-linearity (`torch.relu`),
4. project to one score per token.

The check trains it and requires **1.6 bits per byte or better**: clearly below the bigram's
1.74, so the extra context is paying for itself.

A model like this has a fixed, short memory: anything older than the window is invisible to it.
That is exactly the limit attention will lift in the next lessons.

## Deep

Change `window_size` (1, 2, 4, 8, 16) and train with three seeds each. With `window_size=1` you
have a bigram with a hidden layer. Where does the curve flatten, and what does that tell you
about how far back the text matters?

## Reading

- Bengio et al., "A Neural Probabilistic Language Model" (2003): this model, twenty years ago.
- Karpathy, "Building makemore Part 2: MLP".
