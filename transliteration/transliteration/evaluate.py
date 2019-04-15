import numpy as np

import numba

from .script import SCRIPTS


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


def edit_distance(a, b, *, script_name):
    scriptt = SCRIPTS[script_name]
    assert(all(scriptt._char_in_range(c) for c in a))
    assert(all(scriptt._char_in_range(c) for c in b))
    a = np.asarray([scriptt._intern_char(c)
                    for c in scriptt.preprocess_string(a)])
    b = np.asarray([scriptt._intern_char(c)
                    for c in scriptt.preprocess_string(b)])
    dp = np.full([len(a) + 1, len(b) + 1], np.nan, np.float64)
    return _edit_distance(0, 0, a, b, dp, scriptt.ins_cost, scriptt.sub_cost)


@numba.jit(nopython=True)
def _edit_distance(i, j, sa, sb, dp, ins_cost, sub_cost):
    if i >= len(sa) and j >= len(sa):
        return 0
    if np.isnan(dp[i, j]):
        min_cost = np.inf
        if i < len(sa):
            ins_j = (ins_cost[sa[i]]
                     + _edit_distance(i + 1, j, sa, sb, dp, ins_cost, sub_cost))
            min_cost = min(min_cost, ins_j)
        if j < len(sb):
            ins_i = (ins_cost[sb[j]]
                     + _edit_distance(i, j + 1, sa, sb, dp, ins_cost, sub_cost))
            min_cost = min(min_cost, ins_i)
        if i < len(sa) and j < len(sb):
            if sa[i] == sb[j]:
                sub = _edit_distance(i + 1, j + 1, sa, sb, dp, ins_cost, sub_cost)
            else:
                sub = (sub_cost[sb[j], sa[i]]
                       + _edit_distance(i + 1, j + 1, sa, sb, dp, ins_cost, sub_cost))
            min_cost = min(min_cost, sub)
        dp[i, j] = min_cost
    return dp[i, j]
