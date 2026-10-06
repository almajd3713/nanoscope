A language model reads some text and guesses what comes next. The simplest one looks only at
the *current* token: given "the", how likely is each possible next token? That is a bigram
model, and it is the whole machine in miniature: tokens go in, a score for every token in the
vocabulary comes out, and training nudges the scores toward the text it sees.

## Surface

You will write `MyBigram` in `starter.py`, in two layers:

1. An **embedding**: a table with one row of `d_model` numbers for each token id.
2. A **linear layer** (no bias) that turns those `d_model` numbers into one score per
   vocabulary entry.

`forward` takes token ids of shape `(batch, time)` and returns scores of shape
`(batch, time, vocab_size)`. Nothing looks at neighbouring tokens: each position is
predicted on its own.

Then run `nanoscope learn check foundations/01-bigram`. It builds your class, trains it for a
few hundred steps on TinyStories on your CPU, and checks that it learned (bits per byte well
below 2) and that it lands where the shipped bigram does.

**Bits per byte** is how many bits the model needs, on average, to encode one byte of text.
Lower is better, and unlike loss per token it can be compared across tokenizers.

## Deep

Run it with three seeds (`seeds=3`). Each seed starts from different random weights and sees
the data in a different order, so each is a separate measurement. The three results give a
rough idea of how much a number moves by chance, which is what the `reproduces` check uses: a
single run of yours should land inside the range that the shipped bigram's three seeds span.

Ask yourself: why can a bigram never do better than a certain score, however long you train
it? What information does it throw away?

## Reading

Tiers: **B** build it, **R** read closely, **S** skim, **K** know it exists.

- **B**: Karpathy, "The spelled-out intro to language modeling: building makemore" (the bigram part): build it again from a blank file.
- **S**: Eldan and Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (2023), the dataset the lessons train on.
