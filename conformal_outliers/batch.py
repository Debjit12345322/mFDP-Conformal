"""Batch outlier detection -- Bates et al. (2023), Section 5.2.3.

Each test set of n_test points is split into batches of size b. For each
batch the b (calibrated) conformal p-values are combined into one batch
p-value for the batch-level global null "no outliers in this batch", and a
multiple-testing procedure is then applied across the batches.

Paper setup (the defaults in Config):
    n_test = 1000 -> 100 batches of 10
    90% of batches are pure inliers; in the other 10%, half the points are
    outliers drawn at a = 1.75
    Fisher's combination per batch, then BH with Storey's correction
    (lambda = 0.5, as in the authors' evaluate_all_methods), evaluated over a
    grid of nominal FDR levels.

Procedures compared across batches: plain BH, Storey-BH (the paper's choice),
and mFDP.

Second experiment (paper Figure 9): only inliers, one batch per test set,
reject if the combined p-value is < fwer_level; the FWER is the rejection
rate. Marginal conformal p-values are positively dependent (Lemma 1), which
breaks Fisher's test (Theorem 1); calibration-conditional p-values are
conditionally independent, which keeps it valid.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .calibration import calibrate, method_names
from .config import Config
from .dgp import rP
from .inference import bh_adjust, conformal_p, mfdp_adjust
from .scoring import fit_scorer, score_points
from .summarise import practitioner_summary, summarise_results

PROCEDURES = ("BH", "Storey-BH", "mFDP")


# ---------------------------------------------------------------------------
# Combining p-values within a batch
# ---------------------------------------------------------------------------

def combine_batches(p: np.ndarray, b: int, how: str = "fisher",
                    n_cal: int | None = None) -> np.ndarray:
    """Combine consecutive groups of b p-values into one p-value per group.

    fisher      T = -2 sum log p  ~  chi2(2b) under independence and uniformity
    fisher_dep  paper eq. (6): the dependence correction for marginal
                conformal p-values, with gamma = b / n_cal,
                T' = (T + 2(sqrt(1+gamma) - 1) b) / sqrt(1+gamma)
    """
    P = np.asarray(p, dtype=float).reshape(-1, b)
    T = -2.0 * np.log(P).sum(axis=1)
    if how == "fisher_dep":
        g = b / n_cal
        T = (T + 2.0 * (np.sqrt(1 + g) - 1.0) * b) / np.sqrt(1 + g)
    elif how != "fisher":
        raise ValueError(how)
    return chi2.sf(T, 2 * b)


def _combine_for(method: str, cfg: Config) -> str:
    """The eq. (6) correction targets the dependence among *marginal*
    p-values. Calibration-conditional p-values are independent given the
    calibration data, so plain Fisher is already valid for them."""
    return cfg.batch_combine if method == "Marginal" else "fisher"


# ---------------------------------------------------------------------------
# Procedures across batches
# ---------------------------------------------------------------------------

def storey_pi0(p: np.ndarray, lam: float) -> float:
    """pi0-hat = (1 + #{p > lam}) / (m (1 - lam)); not capped at 1, exactly
    as in the authors' evaluate_all_methods."""
    return (1.0 + np.sum(p > lam)) / (p.size * (1.0 - lam))


def reject_batches(p: np.ndarray, alpha: float, procedure: str,
                   cfg: Config) -> np.ndarray:
    """Boolean rejection vector over batches."""
    if procedure == "BH":
        return bh_adjust(p) <= alpha
    if procedure == "Storey-BH":
        return bh_adjust(p) <= alpha / storey_pi0(p, cfg.storey_lambda)
    if procedure == "mFDP":
        return mfdp_adjust(p, c=1.0 / (2 * p.size), s1=cfg.mfdp_s1,
                           s2=cfg.mfdp_s2) <= alpha
    raise ValueError(procedure)


# ---------------------------------------------------------------------------
# Test-set layout
# ---------------------------------------------------------------------------

def batch_layout(cfg: Config):
    """Which batches contain outliers, and which points are outliers.

    Null batches come first, then the non-null ones; inside a non-null batch
    the outliers are the last positions. Order does not matter for any
    procedure used here, it only keeps the bookkeeping simple.
    """
    b = cfg.batch_size
    n_b = cfg.n_test // b
    n_nonnull = int(round(cfg.batch_prop_nonnull * n_b))
    n_out_per = int(round(cfg.batch_prop_out * b))
    batch_nonnull = np.zeros(n_b, dtype=bool)
    batch_nonnull[n_b - n_nonnull:] = True
    point_out = np.zeros((n_b, b), dtype=bool)
    point_out[batch_nonnull, b - n_out_per:] = True
    return batch_nonnull, point_out.reshape(-1)


# ---------------------------------------------------------------------------
# One practitioner
# ---------------------------------------------------------------------------

def run_batch_practitioner(j: int, cfg: Config, wset: np.ndarray,
                           bands: dict, sgn: int) -> pd.DataFrame:
    methods = method_names(bands)
    rng = np.random.default_rng([cfg.seed, 7, j])

    D = rP(cfg.n_train + cfg.n_cal, 1.0, wset, rng)
    fit = fit_scorer(D[:cfg.n_train], cfg)
    s_cal = np.sort(sgn * score_points(fit, D[cfg.n_train:]))

    batch_nonnull, point_out = batch_layout(cfg)
    n_out_total = int(point_out.sum())
    L = cfg.L
    alphas = np.asarray(cfg.batch_alpha_grid, dtype=float)

    # all L test sets scored in one call
    n_in_total = cfg.n_test - n_out_total
    X_in = rP(n_in_total * L, 1.0, wset, rng)
    X_out = rP(n_out_total * L, cfg.batch_a, wset, rng)
    s_in = (sgn * score_points(fit, X_in)).reshape(L, n_in_total)
    s_out = (sgn * score_points(fit, X_out)).reshape(L, n_out_total)

    shape = (len(methods), len(PROCEDURES), alphas.size, L)
    fdp = np.zeros(shape)
    pwr = np.zeros(shape)
    nrj = np.zeros(shape, dtype=int)
    n_nonnull = int(batch_nonnull.sum())

    for l in range(L):
        s = np.empty(cfg.n_test)
        s[~point_out] = s_in[l]
        s[point_out] = s_out[l]
        p_marg = conformal_p(s, s_cal)

        for mi, m in enumerate(methods):
            pb = combine_batches(calibrate(p_marg, m, bands), cfg.batch_size,
                                 _combine_for(m, cfg), cfg.n_cal)
            for pi, proc in enumerate(PROCEDURES):
                for ai, a in enumerate(alphas):
                    rej = reject_batches(pb, a, proc, cfg)
                    nr = int(rej.sum())
                    nrj[mi, pi, ai, l] = nr
                    fdp[mi, pi, ai, l] = (int((rej & ~batch_nonnull).sum()) / nr
                                          if nr else 0.0)
                    pwr[mi, pi, ai, l] = (int((rej & batch_nonnull).sum()) / n_nonnull
                                          if n_nonnull else np.nan)

    # one row per (method, procedure, level, test set); summaries come later
    mm, pp, aa, ll = np.meshgrid(np.arange(len(methods)), np.arange(len(PROCEDURES)),
                                 np.arange(alphas.size), np.arange(L), indexing="ij")
    return pd.DataFrame({
        "practitioner": j,
        "method": np.asarray(methods, dtype=object)[mm.ravel()],
        "procedure": np.asarray(PROCEDURES, dtype=object)[pp.ravel()],
        "alpha": alphas[aa.ravel()],
        "test_set": ll.ravel() + 1,
        "n_rej": nrj.ravel(),
        "FDP": fdp.ravel(),
        "power": pwr.ravel(),
    })


def run_batch_experiment(cfg: Config, wset, bands, sgn,
                         verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    res = []
    for j in range(1, cfg.J + 1):
        if verbose:
            print(f"batch practitioner {j:3d} / {cfg.J}", flush=True)
        res.append(run_batch_practitioner(j, cfg, wset, bands, sgn))
    out = pd.concat(res, ignore_index=True)
    out.attrs["elapsed_min"] = (time.time() - t0) / 60.0
    return out


# ---------------------------------------------------------------------------
# Global-null FWER as a function of batch size (paper Figure 9)
# ---------------------------------------------------------------------------

def run_global_null(cfg: Config, wset, bands, sgn,
                    verbose: bool = True) -> pd.DataFrame:
    """Only inliers. For each batch size b, draw fwer_n_tests batches, combine
    each with Fisher, reject if < fwer_level. The FWER per practitioner is
    the rejection rate; it should be <= fwer_level."""
    methods = method_names(bands)
    rows = []
    for j in range(1, cfg.J + 1):
        if verbose:
            print(f"global-null practitioner {j:3d} / {cfg.J}", flush=True)
        rng = np.random.default_rng([cfg.seed, 11, j])
        D = rP(cfg.n_train + cfg.n_cal, 1.0, wset, rng)
        fit = fit_scorer(D[:cfg.n_train], cfg)
        s_cal = np.sort(sgn * score_points(fit, D[cfg.n_train:]))
        for b in cfg.fwer_batch_sizes:
            X = rP(b * cfg.fwer_n_tests, 1.0, wset, rng)
            p_marg = conformal_p(sgn * score_points(fit, X), s_cal)
            for m in methods:
                pb = combine_batches(calibrate(p_marg, m, bands), b,
                                     _combine_for(m, cfg), cfg.n_cal)
                rows.append({"practitioner": j, "batch_size": b, "method": m,
                             "FWER": float(np.mean(pb < cfg.fwer_level))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries and plots
# ---------------------------------------------------------------------------

def summarise_batch(ts: pd.DataFrame) -> pd.DataFrame:
    """Same summaries as for individual detection (see summarise.py), one row
    per (method, procedure, alpha)."""
    return summarise_results(ts)


def make_batch_plots(ts: pd.DataFrame, fw: pd.DataFrame | None,
                     cfg: Config, bands: dict, outdir: str) -> None:
    """FDP and power against the nominal level, procedures side by side.
    BH and Storey-BH are shown on their own guarantee (cFDR, mean over the L
    test sets), mFDP on its own (cmFDP, median over L); power is cPower for
    all three."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plots import dodged_boxes, fdr_column

    methods = method_names(bands)
    prac = practitioner_summary(ts)
    alphas = list(cfg.batch_alpha_grid)
    procs = list(PROCEDURES)

    for kind, fname in (("fdp", "batch_fdp.png"), ("power", "batch_power.png")):
        fig, axes = plt.subplots(len(methods), 1, figsize=(9, 3.8 * len(methods)),
                                 squeeze=False)
        for mi, m in enumerate(methods):
            ax = axes[mi, 0]

            def data_of(si, a, m=m):
                proc = procs[si]
                col = fdr_column(proc)[0] if kind == "fdp" else "cPower"
                sel = ((prac["method"] == m) & (prac["procedure"] == proc) &
                       np.isclose(prac["alpha"], a))
                return prac.loc[sel, col].dropna().values

            labels = ([f"{p} ({fdr_column(p)[1]})" for p in procs] if kind == "fdp"
                      else procs)
            dodged_boxes(ax, alphas, None, data_of, labels)
            if kind == "fdp":
                ax.plot(range(len(alphas)), alphas, "--", color="grey")
            ax.set_ylabel("FDP summary" if kind == "fdp" else "cPower")
            ax.set_xlabel("nominal level")
            ax.set_title(f"{m}: batches of {cfg.batch_size}, a = {cfg.batch_a}, "
                         f"{cfg.J} practitioners", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)

    if fw is not None and len(fw):
        sizes = list(cfg.fwer_batch_sizes)
        fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 3.6),
                                 sharey=True, squeeze=False)
        for mi, m in enumerate(methods):
            ax = axes[0, mi]
            g = fw[fw.method == m]
            ax.boxplot([g.loc[g.batch_size == b, "FWER"].values for b in sizes],
                       tick_labels=[str(b) for b in sizes], widths=0.6,
                       flierprops={"markersize": 2})
            ax.axhline(cfg.fwer_level, ls="--", color="grey")
            ax.set_title(m, fontsize=9)
            ax.set_xlabel("batch size")
        axes[0, 0].set_ylabel("FWER under the global null")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "batch_fwer_global_null.png"), dpi=150)
        plt.close(fig)
