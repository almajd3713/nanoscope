def variance_split(grid):
    rows, cols = len(grid), len(grid[0])
    grand = sum(sum(r) for r in grid) / (rows * cols)
    row_means = [sum(r) / cols for r in grid]
    col_means = [sum(grid[i][j] for i in range(rows)) / rows for j in range(cols)]
    init = cols * sum((m - grand) ** 2 for m in row_means) / (rows - 1)
    data = rows * sum((m - grand) ** 2 for m in col_means) / (cols - 1)
    resid = sum((grid[i][j] - row_means[i] - col_means[j] + grand) ** 2
                for i in range(rows) for j in range(cols)) / ((rows - 1) * (cols - 1))
    return init, data, resid
