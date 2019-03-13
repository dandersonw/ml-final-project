def top_k_accuracy(gold,
                   pred,
                   k=1):
    pred_strs, pred_weights = pred
    correct = 0
    count = 0
    for g, p in zip(gold, pred_strs):
        count += 1
        if g in p[:k]:
            correct += 1
    return correct / count


def mrr(gold, pred, k=5):
    pred_strs, pred_weights = pred
    rr = 0
    count = 0
    for g, p in zip(gold, pred_strs):
        count += 1
        p = p[:k]
        if g in p:
            rr += 1 / (p.index(g) + 1)
    return rr / count
