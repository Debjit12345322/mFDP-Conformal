"""Summaries, computed from the per-test-set records.

Notation: FDP_jl is the FDP of practitioner j on test set l.

    conditional (per practitioner, over its L test sets)
        cFDR_j   = mean_l FDP_jl        (Bates et al. 2023)
        cmFDP_j  = median_l FDP_jl      (conditional median FDP)
        cPower_j = mean_l power_jl      (Bates et al. 2023)

    marginal (over everything)
        mFDR     = mean_j cFDR_j  = mean of all J x L FDPs   (Bates et al.)
        MFDP     = median of all J x L FDPs  (pooled; NOT a mean or median
                   of the cmFDP_j)
        share_FDP_le_alpha = fraction of all J x L test sets with FDP <= alpha;
                   mFDP's guarantee (Theorem 1) says this is >= 1/2
        mPower   = mean_j cPower_j                            (Bates et al.)

Naming: in Bates et al. the "m" of mFDR means marginal; in Hemerik et al. the
"m" of mFDP means median. MFDP here is the marginal median FDP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def practitioner_summary(ts: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, a, alpha, practitioner)."""
    keys = [k for k in ("method", "procedure", "a", "alpha", "practitioner")
            if k in ts.columns]
    return (ts.groupby(keys, observed=True, sort=False)
              .agg(cFDR=("FDP", "mean"),
                   cmFDP=("FDP", "median"),
                   cPower=("power", "mean"),
                   cmPower=("power", "median"))
              .reset_index())


def summarise_results(ts: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, [procedure,] a, alpha) with every summary."""
    prac = practitioner_summary(ts)
    keys = [k for k in ("method", "procedure", "a", "alpha") if k in ts.columns]

    pooled = (ts.assign(_le=ts["FDP"] <= ts["alpha"])
                .groupby(keys, observed=True, sort=False)
                .agg(MFDP=("FDP", "median"),
                     share_FDP_le_alpha=("_le", "mean"),
                     n_test_sets=("FDP", "size"))
                .reset_index())

    cond = (prac.assign(_cfdr_above=prac["cFDR"] > prac["alpha"],
                        _cmfdp_above=prac["cmFDP"] > prac["alpha"])
                .groupby(keys, observed=True, sort=False)
                .agg(mFDR=("cFDR", "mean"),
                     cFDR_q90=("cFDR", lambda x: x.quantile(0.9)),
                     frac_cFDR_above=("_cfdr_above", "mean"),
                     cmFDP_median=("cmFDP", "median"),
                     cmFDP_q90=("cmFDP", lambda x: x.quantile(0.9)),
                     frac_cmFDP_above=("_cmfdp_above", "mean"),
                     mPower=("cPower", "mean"),
                     cPower_median=("cPower", "median"))
                .reset_index())

    out = cond.merge(pooled, on=keys)
    return out.sort_values(keys, kind="stable").reset_index(drop=True)
