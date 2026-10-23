#!/usr/bin/env python3
"""Individual outlier detection (Bates et al. 2023, Section 5.2.2).

    python run.py --adjust BH   --outdir results/bh
    python run.py --adjust mFDP --outdir results/mfdp
    python run.py --J 5 --L 5 --outdir results/test        # quick check

Every test set's FDP and power is saved, so all summaries (cFDR, cmFDP, mFDR,
the pooled MFDP, power) come from the same run. The combined figures and the
table for the paper are made afterwards by analyse.py.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from conformal_outliers.calibration import build_bands, method_names
from conformal_outliers.config import default_config
from conformal_outliers.dgp import make_wset
from conformal_outliers.experiment import run_experiment
from conformal_outliers.io import cached_fs_correction, save_results
from conformal_outliers.plots import make_plots
from conformal_outliers.scoring import check_orientation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adjust", choices=["BH", "mFDP"])
    p.add_argument("--J", type=int)
    p.add_argument("--L", type=int)
    p.add_argument("--n-jobs", type=int, dest="n_jobs")
    p.add_argument("--seed", type=int)
    p.add_argument("--outdir")
    p.add_argument("--conditional", action="store_true",
                   help="also run the calibration-conditional p-values")
    p.add_argument("--no-pvalues", action="store_true")
    a = p.parse_args()
    opts = {k: v for k, v in vars(a).items()
            if v is not None and k not in ("no_pvalues", "conditional")}
    if a.no_pvalues:
        opts["save_pvalues"] = False
    if a.conditional:
        opts["include_conditional"] = True
    return opts


def main():
    cfg = default_config(**parse_args())
    os.makedirs(cfg.outdir, exist_ok=True)
    print(f"J = {cfg.J}, L = {cfg.L}, adjust = {cfg.adjust}, "
          f"levels = {list(cfg.alpha_grid)}, outdir = {cfg.outdir}")

    rng = np.random.default_rng(cfg.seed)
    wset = make_wset(cfg, rng)
    fs = cached_fs_correction(cfg) if cfg.include_conditional else None
    bands = build_bands(cfg, fs)
    sgn = check_orientation(cfg, wset, rng)

    ts = run_experiment(cfg, wset, bands, sgn)
    summ = save_results(ts, cfg, wset, bands, sgn, fs)
    make_plots(ts, cfg, bands)

    main = summ[np.isclose(summ["alpha"], cfg.main_alpha)]
    cols = (["method", "a", "mFDR", "frac_cFDR_above", "mPower"] if cfg.adjust == "BH"
            else ["method", "a", "MFDP", "share_FDP_le_alpha", "cmFDP_median",
                  "frac_cmFDP_above", "mPower"])
    print(f"\nelapsed: {ts.attrs['elapsed_min']:.1f} min; "
          f"{len(ts)} test-set records for {len(method_names(bands))} method(s)")
    print(f"level = {cfg.main_alpha:g}:")
    print(main[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwritten to {os.path.abspath(cfg.outdir)}")


if __name__ == "__main__":
    main()
