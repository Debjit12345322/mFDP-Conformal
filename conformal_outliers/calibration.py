"""Calibration-conditional machinery.

Each method builds a simultaneous confidence band `aseq` for the uniform
order statistics; betainv_generic maps a marginal p-value through it. aseq
depends ONLY on (n_cal, delta, k), never on the data, so the bands are built
once by build_bands() and reused for every test set. Per test set the
correction is a lookup.

Index convention: the R original is 1-based. Every place where an R
subscript appears in the maths, the Python code keeps the 1-based *value*
and subtracts 1 only at the point of indexing. Comments mark those spots.
"""

from __future__ import annotations

import numpy as np

from .config import Config

MARGINAL = "Marginal"


def compute_cn(delta: float, n: int) -> float:
    cn = (-np.log(-np.log(1 - delta)) + 2 * np.log(np.log(n))
          + 0.5 * np.log(np.log(np.log(n))) - 0.5 * np.log(np.pi))
    return cn / np.sqrt(2 * np.log(np.log(n)))


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Rolling mean of width k, 'valid' mode -> length len(x) - k + 1."""
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return (cs[k:] - cs[:-k]) / k


def compute_aseq(n: int, k: int, delta: float) -> np.ndarray:
    k = int(k)
    # R: mean(log((n - k + 1):n))
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n - k + 1, n + 1)))
    fac2 = moving_average(np.log(np.arange(1, n + 1)), k)
    return np.concatenate((np.zeros(k - 1), np.exp(fac2 + fac1)))


def betainv_generic(pvals: np.ndarray, aseq: np.ndarray) -> np.ndarray:
    """Map marginal p-values through a band. aseq must have length n_cal."""
    pvals = np.asarray(pvals, dtype=float)
    n = aseq.size
    # R computes a 1-based index, clipped to [1, n]; subtract 1 to index.
    idx = np.floor((n + 1) * (1 - pvals)).astype(np.int64)
    idx = np.clip(idx, 1, n) - 1
    return 1.0 - aseq[idx]


def aseq_simes(n: int, k: int, delta: float) -> np.ndarray:
    return compute_aseq(n, k, delta)


def aseq_asymptotic(n: int, delta: float) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    cn = compute_cn(delta, n)
    b = i / n + cn * np.sqrt(i * (n - i)) / (n * np.sqrt(n))
    return 1.0 - np.minimum(1.0, b[::-1])


def compute_hybrid_bound(delta: float, n: int, gamma: float) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)          # 1-based values
    cna = compute_cn(delta - gamma, n)
    bound = i / n + cna * np.sqrt(i * (n - i)) / (n * np.sqrt(n))

    # linearise past n/2.  R: kl <- as.integer(n/2); slope from bound[kl] -
    # bound[kl-1]; overwrite bound[(kl+1):n].  Those are 1-based subscripts.
    kl = int(n / 2)
    slope = bound[kl - 1] - bound[kl - 2]
    bound[kl:] = bound[kl - 1] + slope * (i[kl:] - kl)

    # and cap by the Simes band
    bound_s = 1.0 - compute_aseq(n, int(n / 2), delta)[::-1]
    return np.minimum(bound_s, bound)


def estimate_fs_correction(delta: float, n: int, n_mc: int = 10000,
                           rng: np.random.Generator | None = None) -> float:
    """Binary search for the finite-sample correction gamma: the value making
    the simulated probability of the empirical process crossing the hybrid
    bound equal to delta. Expensive -- io.cached_fs_correction caches it."""
    if rng is None:
        rng = np.random.default_rng(0)
    Ut = np.sort(rng.random((n, n_mc)), axis=0)

    def prob_crossing(gamma: float) -> float:
        bound = compute_hybrid_bound(delta, n, gamma)
        return float(np.mean(np.any(Ut > bound[:, None], axis=0)))

    g0 = -(1 - 1e-6 - delta)
    g1 = delta - 1e-6
    gamma = g1
    while abs(g1 - g0) > 1e-6:
        gamma = (g0 + g1) / 2
        if prob_crossing(gamma) > delta:
            g0 = gamma
        else:
            g1 = gamma
    return gamma


def build_bands(cfg: Config, fs_correction: float) -> dict[str, np.ndarray]:
    """The three conditional bands, keyed by method name. fs_correction is
    passed in rather than computed here so it can be cached across runs."""
    if not cfg.include_conditional:
        return {}                     # marginal conformal p-values only
    n = cfg.n_cal
    return {
        "Conditional (Simes)": aseq_simes(n, cfg.k_ccv, cfg.delta_ccv),
        "Conditional (Asymptotic)": aseq_asymptotic(n, cfg.delta_ccv),
        "Conditional (Monte Carlo)":
            1.0 - np.minimum(
                1.0,
                compute_hybrid_bound(cfg.delta_ccv, n, fs_correction)[::-1]),
    }


def method_names(bands: dict[str, np.ndarray]) -> list[str]:
    return [MARGINAL] + list(bands.keys())


def calibrate(p: np.ndarray, method: str,
              bands: dict[str, np.ndarray]) -> np.ndarray:
    """Apply the calibration named by `method` to marginal p-values."""
    if method == MARGINAL:
        return np.asarray(p, dtype=float)
    return np.minimum(1.0, betainv_generic(p, bands[method]))
