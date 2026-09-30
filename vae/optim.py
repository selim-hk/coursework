import numpy as np


def adam_init(P):
    return {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in P.items()}


def adam_step(P, g, state, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    for k in g:
        m, v = state[k]
        m[:] = b1 * m + (1 - b1) * g[k]
        v[:] = b2 * v + (1 - b2) * g[k] ** 2
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        P[k] -= lr * mhat / (np.sqrt(vhat) + eps)
