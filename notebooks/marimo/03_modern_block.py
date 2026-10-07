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
    # The modern block

    Since GPT-2 (2019) most open models changed the same few things. `nanoscope.models.Modern` has each as a switch:

    | Switch | Replaces | Idea |
    |---|---|---|
    | `rope` | learned position embeddings | rotate queries and keys by position, so attention depends on relative distance |
    | `rmsnorm` | LayerNorm | normalise by scale only, cheaper and as good |
    | `swiglu` | GELU MLP | a gated MLP |
    | `n_kv_heads` | one key/value per head | share keys and values across heads (GQA) |
    | `qk_norm` | nothing | normalise queries and keys, which steadies attention |
    | `z_loss` | nothing | penalise huge logits |
    """)
    return


@app.cell
def _():
    from nanoscope import compare, run
    from nanoscope.models import Modern

    modern = run(Modern, preset="tinystories-5min")
    return compare, modern


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now compare against GPT-2. Both ship with 3-seed baselines trained on the same data, so the table shows the mean difference with a 95% confidence interval. A difference counts only if the interval excludes zero.
    """)
    return


@app.cell
def _(compare, mo, modern):
    figure = modern.plot()
    mo.vstack([figure, mo.plain_text(str(compare(modern, "gpt2")))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Try next

    - `run(Modern, rope=False)`: switch one part off. Whether it hurts is exactly what the next notebook measures properly.
    - Read `nanoscope/models/modern.py` and find which block each switch picks; the blocks themselves are in `nanoscope/blocks/`.
    """)
    return


if __name__ == "__main__":
    app.run()
