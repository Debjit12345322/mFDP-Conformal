"""One practitioner, and the loop over J of them.

D_j is drawn with a = 1 regardless of test-set signal strength, so the SVM
and calibration scores are fitted once and reused across the whole a_grid.
Only test sets are regenerated.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .calibration import calibrate, method_names
from .config import Config
from .dgp import rP
from .inference import conformal_p, critical_values, make_adjuster
from .scoring import fit_scorer, score_points


def _slug(m: str) -> str:
    out = "".join(ch if ch.isalpha() else "_" for ch in m)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def run_practitioner(j: int, cfg: Config, wset: np.ndarray,
                     bands: dict, sgn: int, adjust) -> pd.DataFrame:
    methods = method_names(bands)
    n_meth = len(methods)

    # Each practitioner gets its own independent stream, so results do not
    # depend on how many practitioners ran before it (and parallelising is
    # safe). R used L'Ecuyer-CMRG for the same reason.
    rng = np.random.default_rng([cfg.seed, j])

    D = rP(cfg.n_train + cfg.n_cal, 1.0, wset, rng)
    X_train = D[:cfg.n_train]
    X_cal = D[cfg.n_train:]

    fit = fit_scorer(X_train, cfg)
    s_cal_sorted = np.sort(sgn * score_points(fit, X_cal))

    n_out, n_in = cfg.n_out, cfg.n_in
    out_idx = np.arange(n_in, cfg.n_test)          # 0-based positions
    alphas = np.asarray(cfg.alpha_grid, dtype=float)
    n_al = alphas.size
    L = cfg.L

    save_p = cfg.save_pvalues and j in cfg.pval_practitioners
    out, p_store = [], []

    for a in cfg.a_grid:
        # all L test sets for this signal strength, scored in one call
        Xt = np.vstack((rP(n_in * L, 1.0, wset, rng),
                        rP(n_out * L, float(a), wset, rng)))
        st = sgn * score_points(fit, Xt)
        S_in = st[:n_in * L].reshape(n_in, L, order="F")
        S_out = st[n_in * L:].reshape(n_out, L, order="F")

        # at a = 1 the "outlier" block is drawn from a = 1 too: no signal,
        # every rejection is false, power undefined
        is_out = np.zeros(cfg.n_test, dtype=bool)
        if a > 1:
            is_out[out_idx] = True
        n_truth = int(is_out.sum())

        if save_p:
            P_raw = np.full((n_meth, cfg.n_test, L), np.nan)
            R_raw = np.zeros((n_meth, cfg.n_test, L), dtype=bool)
            C_raw = np.full((n_meth, cfg.n_test, L), np.nan)   # critical values
            A_raw = np.full((n_meth, cfg.n_test, L), np.nan)   # adjusted p-values
            T_raw = np.full((n_meth, L), np.nan)               # final cutoff

        fdp = np.full((L, n_al, n_meth), np.nan)
        pwr = np.full((L, n_al, n_meth), np.nan)
        nrj = np.zeros((L, n_al, n_meth), dtype=int)

        for l in range(L):
            p_marg = conformal_p(np.concatenate((S_in[:, l], S_out[:, l])),
                                 s_cal_sorted)

            for mi, m in enumerate(methods):
                p_cal = calibrate(p_marg, m, bands)
                padj = adjust(p_cal)

                for k in range(n_al):
                    rej = padj <= alphas[k]
                    n_rej = int(rej.sum())
                    nrj[l, k, mi] = n_rej
                    fdp[l, k, mi] = (int((rej & ~is_out).sum()) / n_rej
                                     if n_rej else 0.0)
                    pwr[l, k, mi] = (int((rej & is_out).sum()) / n_truth
                                     if n_truth else np.nan)

                if save_p:
                    rej0 = padj <= cfg.main_alpha
                    P_raw[mi, :, l] = p_cal
                    A_raw[mi, :, l] = padj
                    R_raw[mi, :, l] = rej0
                    C_raw[mi, :, l] = critical_values(p_cal, cfg, cfg.main_alpha)
                    if rej0.any():
                        T_raw[mi, l] = p_cal[rej0].max()

        # one row per (test set, level, method): nothing is collapsed here,
        # so any summary (mean or median, over L or over J, or pooled) can be
        # computed afterwards from the same run
        ll, kk, mm = np.meshgrid(np.arange(L), np.arange(n_al), np.arange(n_meth),
                                 indexing="ij")
        out.append(pd.DataFrame({
            "practitioner": j,
            "a": a,
            "alpha": alphas[kk.ravel()],
            "method": np.asarray(methods, dtype=object)[mm.ravel()],
            "test_set": ll.ravel() + 1,
            "n_rej": nrj.ravel(),
            "FDP": fdp.ravel(),
            "power": pwr.ravel(),
        }))

        if save_p:
            block = {
                "practitioner": j,
                "a": a,
                "test_set": np.repeat(np.arange(1, L + 1), cfg.n_test),
                "index": np.tile(np.arange(1, cfg.n_test + 1), L),
                "is_outlier": np.tile(is_out, L),
            }
            for mi, m in enumerate(methods):
                s = _slug(m).lower().replace("conditional_", "")
                block[f"p_{s}"] = P_raw[mi].reshape(-1, order="F")
                block[f"adj_{s}"] = A_raw[mi].reshape(-1, order="F")
                block[f"rej_{s}"] = R_raw[mi].reshape(-1, order="F")
                block[f"crit_{s}"] = C_raw[mi].reshape(-1, order="F")
                block[f"cutoff_{s}"] = np.repeat(T_raw[mi], cfg.n_test)
            p_store.append(pd.DataFrame(block))

    if save_p and p_store:
        d = os.path.join(cfg.outdir, "pvalues")
        os.makedirs(d, exist_ok=True)
        pd.concat(p_store, ignore_index=True).to_csv(
            os.path.join(d, f"practitioner_{j:02d}.csv"), index=False)

    return pd.concat(out, ignore_index=True)


def _worker(j: int, cfg: Config, wset: np.ndarray, bands: dict,
            sgn: int) -> pd.DataFrame:
    # Module-level so a process pool can pickle it. The adjuster is built
    # here rather than passed in, because the mFDP one is a lambda and
    # lambdas cannot be pickled.
    return run_practitioner(j, cfg, wset, bands, sgn, make_adjuster(cfg))


def run_experiment(cfg: Config, wset: np.ndarray, bands: dict, sgn: int,
                   verbose: bool = True) -> pd.DataFrame:
    adjust = make_adjuster(cfg)
    methods = method_names(bands)

    def one(j: int) -> pd.DataFrame:
        return run_practitioner(j, cfg, wset, bands, sgn, adjust)

    t0 = time.time()
    if cfg.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        work = partial(_worker, cfg=cfg, wset=wset, bands=bands, sgn=sgn)
        with ProcessPoolExecutor(max_workers=cfg.n_jobs) as ex:
            res = list(ex.map(work, range(1, cfg.J + 1)))
    else:
        res = []
        for j in range(1, cfg.J + 1):
            if verbose:
                print(f"practitioner {j:3d} / {cfg.J}", flush=True)
            res.append(one(j))

    results = pd.concat(res, ignore_index=True)
    results["method"] = pd.Categorical(results["method"], categories=methods,
                                       ordered=True)
    results = results.sort_values(["method", "a", "alpha", "practitioner", "test_set"],
                                  kind="stable").reset_index(drop=True)
    results.attrs["elapsed_min"] = (time.time() - t0) / 60.0
    return results
