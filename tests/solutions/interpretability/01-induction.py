def induction_score(attn, half):
    rows = [attn[i][i - half + 1] for i in range(half, 2 * half)]
    return sum(rows) / len(rows)


def previous_token_score(attn):
    rows = [attn[i][i - 1] for i in range(1, len(attn))]
    return sum(rows) / len(rows)


def first_step_above(steps, scores, threshold):
    for i in range(len(steps)):
        if all(s >= threshold for s in scores[i:]):
            return steps[i]
    return -1
