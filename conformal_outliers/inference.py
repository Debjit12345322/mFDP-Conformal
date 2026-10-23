"""Conformal p-values and the multiple-testing step.

    u_hat(X) = (1 + #{i : s(X_cal_i) <= s(X)}) / (n_cal + 1)
    Granularity floor: min p = 1/(n_cal + 1) ~ 9.99e-4.

Two adjustments:
    bh_adjust    port of R's p.adjust(p, "BH")
    mfdp_adjust  port of mFDP::mFDP.adjust (Jesse Hemerik, CRAN mFDP, GPL-3)

Both are straight translations of the R sources. Stable sorts are used
throughout: conformal p-values are heavily tied (every value is a multiple
of 1/(n_cal+1)), and R's order() is stable for numeric vectors, so an
unstable argsort would permute tied entries differently and change which
hypotheses end up flagged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import Config


def conformal_p(s_test: np.ndarray, s_cal_sorted: np.ndarray) -> np.ndarray:
    """Equivalent of R's (1 + findInterval(s_test, s_cal_sorted)) / (n+1).

    findInterval counts calibration scores <= each test score, i.e. the
    number of breakpoints at or below it, which is np.searchsorted with
    side='right'."""
    n = s_cal_sorted.size
    counts = np.searchsorted(s_cal_sorted, s_test, side="right")
    return (1.0 + counts) / (n + 1.0)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Port of stats::p.adjust(p, method = "BH").

    R:  i <- n:1; o <- order(p, decreasing=TRUE); ro <- order(o)
        pmin(1, cummin(n/i * p[o]))[ro]
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    if n <= 1:
        return p.copy()
    o = np.argsort(-p, kind="stable")
    ro = np.argsort(o, kind="stable")
    i = np.arange(n, 0, -1, dtype=float)      # n, n-1, ..., 1
    adj = np.minimum.accumulate((n / i) * p[o])
    return np.minimum(1.0, adj)[ro]


def mfdp_adjust(p, c="1/(2m)", s1: float = 0.0, s2: float = 0.1) -> np.ndarray:
    """Port of mFDP::mFDP.adjust(P, c, s1, s2).

    Returns mFDP-adjusted p-values. Some can be infinity, which is to be
    interpreted as 1; a `padj <= alpha` test handles that correctly.

    Original R implementation by Jesse Hemerik (CRAN package mFDP, GPL-3).
    This is a translation, not new work.
    """
    p_unsorted = np.asarray(p, dtype=float)
    m = p_unsorted.size

    # --- checks, mirroring the R ones -----------------------------------
    if s1 < 0:
        raise ValueError("s1 should be non-negative")
    if s2 > 1:
        raise ValueError("s2 should be at most 1")
    if s2 <= s1:
        raise ValueError("s2 should be larger than s1")
    if m <= 1:
        raise ValueError("P should be a vector")
    if p_unsorted.max() > 1 or p_unsorted.min() < 0:
        raise ValueError("The p-values in P should be between 0 and 1")

    if isinstance(c, str):
        if c == "1/(2m)":
            c = 1.0 / (2 * m)
        elif c == "1/m":
            c = 1.0 / m
        else:
            raise ValueError('c should be numeric or "1/(2m)" or "1/m"')
    c = float(c)
    if c < 0:
        raise ValueError("c should be non-negative")
    if c > 1:
        raise ValueError("c should be small")

    # --- confidence envelope (i.e. kappa.max) ---------------------------
    order = np.argsort(p_unsorted, kind="stable")
    P = p_unsorted[order]                      # sorted ascending

    mincutoff, maxcutoff = s1, s2
    nr_large = int(np.sum(P >= 1 - maxcutoff))
    nr_toolarge = int(np.sum(P > 1 - mincutoff))

    n_ge_mincut = int(np.sum(P >= 1 - mincutoff))
    kappa0 = np.inf if n_ge_mincut == 0 else (mincutoff + c) / n_ge_mincut

    kappa_max = kappa0
    if nr_large > nr_toolarge:
        # at least one p-value for which 1-p lies in [mincutoff, maxcutoff].
        # R loops i over (m - nr.large + 1):(m - nr.toolarge), 1-based.
        kappas_i = np.full(m, np.inf)
        for i1 in range(m - nr_large + 1, m - nr_toolarge + 1):
            i0 = i1 - 1
            kappas_i[i0] = (1 - P[i0] + c) / np.sum(P >= P[i0])
        # (paradoxically, kappa.max equals this minimum)
        kappa_max = min(kappa0, float(kappas_i.min()))
    if nr_large == 0:
        kappa_max = np.inf

    # --- adjusted p-values ----------------------------------------------
    padj = np.zeros(m)
    r = int(np.sum(P <= s2))                   # 1-based count
    if r < m:
        padj[r:] = np.inf                      # R: Padj[(r+1):m] <- Inf

    if r > 0:
        n_le_s1 = int(np.sum(P <= s1))
        if s1 <= P[r - 1]:
            padj[r - 1] = np.floor((P[r - 1] + c) / kappa_max) / r
        else:
            padj[r - 1] = np.floor((s1 + c) / kappa_max) / n_le_s1

        l = r - 1                              # 1-based
        cont = l > 0 and s1 <= P[l - 1]
        while cont:
            # R: Padj[l] <- min(Padj[l+1], floor((P[l]+c)/kappa.max)/l)
            padj[l - 1] = min(padj[l],
                              np.floor((P[l - 1] + c) / kappa_max) / l)
            l -= 1
            if l == 0:
                cont = False
            else:
                cont = not (P[l - 1] < s1)

        if l > 0:
            padj[:l] = min(padj[l],
                           np.floor((s1 + c) / kappa_max) / n_le_s1)

    out = np.empty(m)
    out[order] = padj
    return out


def make_adjuster(cfg: Config) -> Callable[[np.ndarray], np.ndarray]:
    """THE SWAP POINT, driven by cfg.adjust.

    mfdp_adjust takes no alpha, so adjusted p-values serve the whole alpha
    grid from one call."""
    if cfg.adjust == "BH":
        return bh_adjust
    if cfg.adjust == "mFDP":
        return lambda p: mfdp_adjust(p, c=cfg.mfdp_c,
                                     s1=cfg.mfdp_s1, s2=cfg.mfdp_s2)
    raise ValueError(f"unknown adjust method: {cfg.adjust}")


# ---------------------------------------------------------------------------
# Critical values: the number each sorted p-value must be under, BH-style.
#
#   BH    at rank k:  k * alpha / m
#   mFDP  at rank k:  (floor(k * gamma) + 1) * kappa - c
#
# The rule for both: find the largest rank k whose p-value is under its
# critical value, and reject the k smallest p-values. So a p-value above its
# own critical value can still be rejected if a larger rank passes.
# For mFDP a p-value must also be <= s2 to be rejectable.
# ---------------------------------------------------------------------------

def _resolve_c(c, m: int) -> float:
    if isinstance(c, str):
        return {"1/(2m)": 1.0 / (2 * m), "1/m": 1.0 / m}[c]
    return float(c)


def mfdp_kappa(p, c="1/(2m)", s1: float = 0.0, s2: float = 0.1) -> float:
    """The envelope constant kappa from mFDP.adjust (1/kappa is the slope)."""
    P = np.sort(np.asarray(p, dtype=float))
    m = P.size
    c = _resolve_c(c, m)
    n_ge = int(np.sum(P >= 1 - s1))
    kappa0 = np.inf if n_ge == 0 else (s1 + c) / n_ge
    nr_large = int(np.sum(P >= 1 - s2))
    nr_toolarge = int(np.sum(P > 1 - s1))
    kappa = kappa0
    if nr_large > nr_toolarge:
        idx = np.arange(m - nr_large, m - nr_toolarge)
        counts = m - np.searchsorted(P, P[idx], side="left")   # #{P >= P[i]}
        kappa = min(kappa0, float(np.min((1 - P[idx] + c) / counts)))
    if nr_large == 0:
        kappa = np.inf
    return kappa


def critical_values(p, cfg: Config, alpha: float) -> np.ndarray:
    """Critical value for each p-value, at that p-value's rank."""
    p = np.asarray(p, dtype=float)
    m = p.size
    k = np.empty(m)
    k[np.argsort(p, kind="stable")] = np.arange(1, m + 1)
    if cfg.adjust == "BH":
        return k * alpha / m
    c = _resolve_c(cfg.mfdp_c, m)
    kappa = mfdp_kappa(p, c, cfg.mfdp_s1, cfg.mfdp_s2)
    if not np.isfinite(kappa):
        return np.full(m, np.inf)
    return (np.floor(k * alpha + 1e-12) + 1) * kappa - c
