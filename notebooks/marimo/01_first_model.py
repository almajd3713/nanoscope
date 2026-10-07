import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Your first language model

    You write the model. nanoscope handles the data, the training loop, evaluation and checkpoints.

    The simplest language model there is: a **bigram** model. It predicts the next token by looking only at the current one.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    class Bigram(nn.Module):
        def __init__(self, vocab_size: int, d_model: int = 32):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)  # token id -> vector
            self.head = nn.Linear(d_model, vocab_size, bias=False)    # vector -> score for every next token

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            # idx: (batch, time) token ids -> logits: (batch, time, vocab_size)
            return self.head(self.token_embedding(idx))

    return (Bigram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train it

    `run` does all of this for you:

    - downloads 100,000 short stories from TinyStories
    - trains a 4,096-token tokenizer on them
    - trains your model for 500 steps
    - evaluates it on 200 held-out stories it never trains on

    The first run spends a minute or two preparing data, and that's cached afterwards. Training takes about a minute on a laptop CPU.

    Run the cell again and it loads the finished run instead of retraining. Interrupt it and it picks up where it stopped.
    """)
    return


@app.cell
def _(Bigram):
    from nanoscope import compare, run

    result = run(Bigram, preset="tinystories-5min")
    return compare, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## See what it learned

    The orange **val loss** is measured on text the model has never seen. If it keeps falling, the model is learning something general, not memorizing.

    `compare` then puts your model next to a GPT-2-style transformer that ships with nanoscope, trained on the same data with 3 seeds. Lower bits per byte is better.
    """)
    return


@app.cell
def _(compare, mo, result):
    figure = result.plot()
    result.print_samples(n=2)
    mo.vstack([figure, mo.plain_text(str(compare(result, "gpt2")))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try next

    - `run(Bigram, learning_rate=1e-2)`: bigram models like a high learning rate. Each change of settings gets its own folder under `runs/`, so you can compare them: `compare(new_result, result)`.
    - `run(Bigram, d_model=64)`: a bigger embedding.
    - `result.generate("Once upon a time")`: write your own prompt.
    - `from nanoscope.models import GPT2, Modern`: the transformers behind the baselines. Read their source, then train one: `run(Modern)`.
    - `run(Bigram, seeds=3)`: three seeds. With three or more per side, `compare` gives a 95% confidence interval and says whether a difference is real or within noise.
    - Bits per byte is the loss converted to bits per byte of text. Unlike loss, it stays comparable when you change the tokenizer (`vocab_size=...`, `tokenizer="bytes"`).
    """)
    return


if __name__ == "__main__":
    app.run()
