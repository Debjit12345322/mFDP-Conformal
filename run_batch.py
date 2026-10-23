#!/usr/bin/env python3
"""Batch outlier detection (Bates et al. 2023, Section 5.2.3).

    python run_batch.py --outdir results/batch
    python run_batch.py --J 5 --L 5 --outdir results/batch_test   # quick check
    python run_batch.py --no-fwer          # skip the global-null experiment

BH, Storey-BH and mFDP are all run on the same batch p-values. Every test
set's FDP and power is saved, so cFDR/mFDR (for BH and Storey-BH) and
cmFDP/MFDP (for mFDP) all come from this one run.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from conformal_outliers.batch import (make_batch_plots, run_batch_experiment,
                                      run_global_null)
from conformal_outliers.calibration import build_bands
from conformal_outliers.config import default_config
from conformal_outliers.dgp import make_wset
from conformal_outliers.io import cached_fs_correction
from conformal_outliers.scoring import check_orientation
from conformal_outliers.summarise import practitioner_summary, summarise_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int)
    ap.add_argument("--L", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--batch-combine", choices=["fisher", "fisher_dep"],
                    dest="batch_combine")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--outdir", default="results/batch")
    ap.add_argument("--conditional", action="store_true",
                    help="also run the calibration-conditional p-values")
    ap.add_argument("--no-fwer", action="store_true")
    a = ap.parse_args()
    opts = {k: v for k, v in vars(a).items()
            if v is not None and k not in ("no_fwer", "conditional")}
    if a.conditional:
        opts["include_conditional"] = True
    cfg = default_config(**opts)
    os.makedirs(cfg.outdir, exist_ok=True)

    print(f"J = {cfg.J}, L = {cfg.L}, batches of {cfg.batch_size}, "
          f"a = {cfg.batch_a}, combine = {cfg.batch_combine}, outdir = {cfg.outdir}")
    rng = np.random.default_rng(cfg.seed)
    wset = make_wset(cfg, rng)
    fs = cached_fs_correction(cfg) if cfg.include_conditional else None
    bands = build_bands(cfg, fs)
    sgn = check_orientation(cfg, wset, rng)

    ts = run_batch_experiment(cfg, wset, bands, sgn)
    summ = summarise_results(ts)
    ts.to_csv(os.path.join(cfg.outdir, "batch_per_testset.csv"), index=False)
    practitioner_summary(ts).to_csv(
        os.path.join(cfg.outdir, "batch_per_practitioner.csv"), index=False)
    summ.to_csv(os.path.join(cfg.outdir, "batch_summary.csv"), index=False)
    cols = ["method", "procedure", "alpha", "mFDR", "MFDP",
            "share_FDP_le_alpha", "mPower"]
    print(summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fw = None
    if not a.no_fwer:
        fw = run_global_null(cfg, wset, bands, sgn)
        fw.to_csv(os.path.join(cfg.outdir, "batch_fwer_global_null.csv"), index=False)
        print(fw.groupby(["method", "batch_size"], sort=False)["FWER"]
                .median().unstack().round(3).to_string())

    make_batch_plots(ts, fw, cfg, bands, cfg.outdir)
    print(f"\nwritten to {os.path.abspath(cfg.outdir)}")


if __name__ == "__main__":
    main()
