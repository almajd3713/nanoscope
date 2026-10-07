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
    # A real transformer: GPT-2

    The bigram model only sees the current token. A transformer lets every token look back at all the earlier ones through **attention**.

    nanoscope ships the GPT-2 design in `nanoscope/models/gpt2.py` (about 30 lines, one `Decoder` of library blocks): token + position embeddings, then blocks of LayerNorm → causal self-attention → LayerNorm → MLP, each with a residual connection. Open it and read it alongside this notebook; each part lives in `nanoscope/blocks/` (`structure.py` has the layer loop, `attention.py` the attention).
    """)
    return


@app.cell
def _():
    from nanoscope import compare, run
    from nanoscope.models import GPT2

    gpt2 = run(GPT2, preset="tinystories-5min")
    return compare, gpt2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same data, same tokenizer, same number of training steps as the bigram in notebook 01. Everything the model can do better comes from the architecture.

    Compare the two. If you ran notebook 01 first, its results are already on disk.
    """)
    return


@app.cell
def _(compare, gpt2, mo):
    figure = gpt2.plot()
    gpt2.print_samples(n=2)
    mo.vstack([figure, mo.plain_text(str(compare(gpt2, "bigram")))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try next

    - `run(GPT2, n_layers=2)` or `run(GPT2, d_model=256, n_heads=8)`: how do depth and width change the loss? Each setting gets its own folder under `runs/`.
    - `gpt2.generate("Once upon a time")`: write your own prompt.
    - Next notebook: the modern block, and why it beats GPT-2.
    """)
    return


if __name__ == "__main__":
    app.run()
