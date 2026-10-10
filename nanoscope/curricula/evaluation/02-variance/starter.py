"""Evaluation 2: two kinds of seed.   Check: nanoscope learn check evaluation/02-variance

`grid[i][j]` is the final val_bpb of a run with initial-weights seed i and data-order seed j.
Split the variation in the grid into the part that comes from i, the part that comes from j,
and what neither explains. The lesson text explains the three numbers.
"""


def variance_split(grid):
    """Return (init, data, residual) as mean squares.

    With I rows, J columns, row means r_i, column means c_j and the grand mean g:

        init     = J * sum((r_i - g)**2) / (I - 1)
        data     = I * sum((c_j - g)**2) / (J - 1)
        residual = sum((grid[i][j] - r_i - c_j + g)**2) / ((I - 1) * (J - 1))
    """
    # TODO
    raise NotImplementedError
