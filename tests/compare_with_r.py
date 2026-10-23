"""Cross-validate the ports against R, to machine precision.

Step 1 -- in R, from the R project root:

    P <- runif(500); P2 <- sort(runif(300)); P3 <- rep(c(.02,.5,.9), 100)
    write.csv(data.frame(P, bh = p.adjust(P, "BH"),
                         mfdp = mFDP::mFDP.adjust(P)),
              "r_check_1.csv", row.names = FALSE)
    write.csv(data.frame(P = P2, bh = p.adjust(P2, "BH"),
                         mfdp = mFDP::mFDP.adjust(P2)),
              "r_check_2.csv", row.names = FALSE)
    write.csv(data.frame(P = P3, bh = p.adjust(P3, "BH"),
                         mfdp = mFDP::mFDP.adjust(P3)),
              "r_check_3.csv", row.names = FALSE)

    ## and the bands, which depend on no data at all
    source("R/calibration.R"); source("R/config.R")
    cfg <- default_config()
    write.csv(data.frame(
      simes = compute_aseq(cfg$n_cal, cfg$k_ccv, cfg$delta_ccv),
      asymp = aseq_asymptotic(cfg$n_cal, cfg$delta_ccv)),
      "r_check_bands.csv", row.names = FALSE)

Step 2 -- copy those CSVs next to this script and run:

    python tests/compare_with_r.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conformal_outliers.calibration import (aseq_asymptotic,  # noqa: E402
                                            compute_aseq)
from conformal_outliers.config import default_config  # noqa: E402
from conformal_outliers.inference import bh_adjust, mfdp_adjust  # noqa: E402

TOL = 1e-12
here = os.path.dirname(os.path.abspath(__file__))
fails = 0


def check(label, ours, theirs, tol=TOL):
    global fails
    ours, theirs = np.asarray(ours, float), np.asarray(theirs, float)
    both_inf = np.isinf(ours) & np.isinf(theirs)
    finite = ~(np.isinf(ours) | np.isinf(theirs))
    okay = (np.array_equal(np.isinf(ours), np.isinf(theirs))
            and np.allclose(ours[finite], theirs[finite], atol=tol, rtol=0))
    worst = (np.max(np.abs(ours[finite] - theirs[finite]))
             if finite.any() else 0.0)
    print(f"{'ok  ' if okay else 'FAIL'} {label:32s} "
          f"max|diff| = {worst:.3e}  ({both_inf.sum()} shared Inf)")
    if not okay:
        fails += 1
        bad = np.argsort(-np.abs(np.where(finite, ours - theirs, 0)))[:5]
        for i in bad:
            print(f"       i={i}  python={ours[i]!r}  R={theirs[i]!r}")


found = False
for n in (1, 2, 3):
    path = os.path.join(here, f"r_check_{n}.csv")
    if not os.path.exists(path):
        continue
    found = True
    df = pd.read_csv(path)
    p = df["P"].to_numpy()
    check(f"BH   vs R (r_check_{n})", bh_adjust(p), df["bh"].to_numpy())
    check(f"mFDP vs R (r_check_{n})", mfdp_adjust(p), df["mfdp"].to_numpy())

bpath = os.path.join(here, "r_check_bands.csv")
if os.path.exists(bpath):
    found = True
    b = pd.read_csv(bpath)
    cfg = default_config()
    check("Simes band vs R",
          compute_aseq(cfg.n_cal, cfg.k_ccv, cfg.delta_ccv),
          b["simes"].to_numpy())
    check("Asymptotic band vs R",
          aseq_asymptotic(cfg.n_cal, cfg.delta_ccv),
          b["asymp"].to_numpy())

if not found:
    print(__doc__)
    sys.exit("no r_check_*.csv files found -- see the instructions above")

print()
sys.exit(f"{fails} comparison(s) failed" if fails else "all comparisons match R")
