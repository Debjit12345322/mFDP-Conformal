"""Output tree and the finite-sample-correction cache."""

from __future__ import annotations

import json
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd

from .calibration import estimate_fs_correction, method_names
from .config import Config
from .summarise import practitioner_summary, summarise_results


def cached_fs_correction(cfg: Config, verbose: bool = True) -> float:
    """gamma is deterministic given (delta, n_cal, n_mc) up to MC noise, and
    the binary search is the slowest setup step, so cache it. Delete the
    cache directory to force a recompute."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = os.path.join(
        cfg.cache_dir,
        f"fs_correction_delta{cfg.delta_ccv:g}_n{cfg.n_cal}_mc{cfg.n_mc}_seed{cfg.seed}.json")

    if os.path.exists(path):
        with open(path) as f:
            v = json.load(f)["fs_correction"]
        if verbose:
            print(f"  fs_correction = {v:.6f}   (cached)")
        return v

    t0 = time.time()
    rng = np.random.default_rng([cfg.seed, 99])
    v = estimate_fs_correction(cfg.delta_ccv, cfg.n_cal, cfg.n_mc, rng)
    with open(path, "w") as f:
        json.dump({"fs_correction": v, "delta": cfg.delta_ccv,
                   "n_cal": cfg.n_cal, "n_mc": cfg.n_mc}, f)
    if verbose:
        print(f"  fs_correction = {v:.6f}   ({time.time() - t0:.1f} s)")
    return v


def save_results(ts: pd.DataFrame, cfg: Config, wset: np.ndarray, bands: dict,
                 sgn: int, fs_correction) -> pd.DataFrame:
    """Write the three result tables and a pickle with everything needed to
    reproduce the run. Returns the summary table.

        per_testset.csv      one row per (method, a, alpha, practitioner, test set)
        per_practitioner.csv cFDR, cmFDP, cPower, cmPower for each practitioner
        summary.csv          mFDR, MFDP, share of test sets with FDP <= alpha,
                             mPower, ... for each (method, a, alpha)
    """
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)
    prac = practitioner_summary(ts)
    summ = summarise_results(ts)

    ts.to_csv(os.path.join(outdir, "per_testset.csv"), index=False)
    prac.to_csv(os.path.join(outdir, "per_practitioner.csv"), index=False)
    summ.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    with open(os.path.join(outdir, "results.pkl"), "wb") as f:
        pickle.dump({
            "per_testset": ts,
            "summary": summ,
            "bands": bands,
            "params": {
                **{k: v for k, v in vars(cfg).items()},
                "methods": method_names(bands),
                "fs_correction": fs_correction,
                "wset": wset,
                "sign": sgn,
                "elapsed_min": ts.attrs.get("elapsed_min"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }, f)
    return summ
