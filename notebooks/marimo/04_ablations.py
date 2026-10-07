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
    # Ablations: which part matters?

    One run per setting is not evidence: a different seed can flip the result. A **Study** trains every variant with several seeds, on the same token budget and with matched parameter counts, then reports differences with confidence intervals.

    Below, each variant is the modern block with one component removed. This is a small TinyStories version; the research version of the same file is `studies/m1_ablation.py`.
    """)
    return


@app.cell
def _():
    from nanoscope import Study, Tokens
    from nanoscope.models import Modern

    study = Study("ablation-demo", preset="tinystories-5min", seeds=3, budget=Tokens(2e6),
                  baseline="modern")
    study.add("modern", Modern)
    study.add("no-rope", Modern, rope=False)
    study.add("no-swiglu", Modern, swiglu=False)
    study.add("no-qk-norm", Modern, qk_norm=False)
    study.run()
    return (study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This takes a while on CPU and a few minutes on a GPU. Progress is on screen, and `nanoscope status runs` shows it from another terminal. If it stops, run the cell again: finished runs are kept and the rest resume.
    """)
    return


@app.cell
def _(mo, study):
    mo.plain_text(str(study.report()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the table as: *Δ vs modern* is the change from removing that part, so a positive loss difference means the part helps. A row whose interval includes zero is **within noise**: with this many seeds you can't tell.

    ## Going further

    For preregistered studies, parameter matching, several GPUs or Kaggle, see the research guide in `docs/research.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
