"""Data generating model  P_X^a

    X_i = sqrt(a) * V_i + W_i,   V_i ~ N(0, I_d),   W_i ~ Unif(W-set)
    p(x) = (1/n_atom) sum_k N(x | w^(k), a * I_d)

The mixture centres are drawn once and then passed around, never read from
module state. The Generator is passed explicitly too, so nothing depends on
global RNG state.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def make_wset(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-3.0, 3.0, size=(cfg.n_atom, cfg.d))


def rP(n: int, a: float, wset: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_atom, d = wset.shape
    V = rng.standard_normal((n, d))
    W = wset[rng.integers(0, n_atom, size=n)]
    return np.sqrt(a) * V + W
