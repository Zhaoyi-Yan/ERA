import torch

__all__ = ['pdist']


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    # let the diag values to be 0, ie, i not equal j
    res[range(len(e)), range(len(e))] = 0
    return res

