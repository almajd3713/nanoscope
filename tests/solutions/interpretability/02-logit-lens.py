def top_token(logits):
    return max(range(len(logits)), key=lambda i: logits[i])


def rank_of(logits, target):
    return sum(1 for v in logits if v > logits[target])


def first_settled_layer(layer_logits, target):
    for k in range(len(layer_logits)):
        if all(top_token(row) == target for row in layer_logits[k:]):
            return k
    return -1
