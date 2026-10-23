"""Every tunable parameter lives here. Nothing else in the package reads a
global. Override at the call site:  default_config(J=100, L=100)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass
class Config:
    # --- data generating model -------------------------------------------
    d: int = 50           # ambient dimension
    n_atom: int = 50      # |W| : number of mixture components
    n_train: int = 1000   # fit the one-class SVM
    n_cal: int = 1000     # calibrate conformal p-values
    n_test: int = 1000    # size of each test set
    prop_out: float = 0.10  # fraction of outliers in each test set

    # --- experiment size --------------------------------------------------
    J: int = 100         # practitioners        (paper: 100)
    L: int = 100        # test sets each       (paper: 100)

    a_grid: Sequence[float] = (1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3)
    # nominal levels: alpha for BH, gamma for mFDP. All are evaluated in the
    # same run; main_alpha is the level used in the main figures.
    alpha_grid: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    main_alpha: float = 0.1

    # --- which p-values ---------------------------------------------------
    # False: marginal conformal p-values only (the paper). True: also the
    # three calibration-conditional corrections of Bates et al. (2023).
    include_conditional: bool = False

    # --- calibration-conditional correction (paper 5.2.2) -----------------
    delta_ccv: float = 0.1   # confidence for the CCV correction
    simes_kden: int = 2      # k = n_cal / simes_kden
    n_mc: int = 10000        # MC replicates for the finite-sample correction

    # --- multiple testing -------------------------------------------------
    # "BH"   guarantees  E[FDP] <= alpha         (mean)
    # "mFDP" guarantees  P(FDP <= alpha) >= 0.5  (median)
    adjust: str = "mFDP"
    mfdp_c: object = "1/(2m)"   # "1/(2m)", "1/m", or a float
    mfdp_s1: float = 0.0
    mfdp_s2: float = 0.1     # window [s1, s2] of thresholds, Hemerik et al.'s
                             # default; fixed, whatever levels are evaluated

    # --- machinery --------------------------------------------------------
    # One-class SVM, matching the authors' code (msesia/conditional-conformal-
    # pvalues): OneClassSVM(nu=contamination, kernel="rbf", gamma=0.1) with
    # contamination clamped to 0.004.
    svm_nu: float = 0.004
    svm_gamma: float = 0.1
    n_jobs: int = 1        # >1 parallelises over practitioners
    seed: int = 2024

    # --- raw + calibrated p-values and rejection flags ---------------------
    save_pvalues: bool = True
    pval_practitioners: Sequence[int] = (1,)   # range(1, J+1) saves all

    # --- batch outlier detection (Bates et al. 2023, Section 5.2.3) ---------
    batch_size: int = 10              # points per batch; n_test must divide
    batch_prop_nonnull: float = 0.10  # fraction of batches containing outliers
    batch_prop_out: float = 0.50      # fraction of outliers inside such a batch
    batch_a: float = 1.75             # outlier signal strength in batches
    batch_alpha_grid: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    batch_combine: str = "fisher"     # "fisher", or "fisher_dep" = paper eq. (6)
    storey_lambda: float = 0.5        # as in the authors' code
    # global-null experiment (paper Figure 9)
    fwer_batch_sizes: Sequence[int] = (1, 2, 5, 10, 20, 50, 100)
    fwer_level: float = 0.1
    fwer_n_tests: int = 200           # test batches per practitioner and size

    outdir: str = "results"
    cache_dir: str = "results/cache"

    # derived
    k_ccv: int = field(init=False, default=0)

    def __post_init__(self):
        self.k_ccv = int(self.n_cal / self.simes_kden)
        self.validate()

    def validate(self):
        if self.adjust not in ("BH", "mFDP"):
            raise ValueError(f"adjust must be 'BH' or 'mFDP', got {self.adjust!r}")
        if not 0 < self.prop_out < 1:
            raise ValueError("prop_out must be in (0, 1)")
        if any(a < 1 for a in self.a_grid):
            raise ValueError("a_grid values must be >= 1")
        if any(not 0 < al < 1 for al in self.alpha_grid):
            raise ValueError("alpha_grid values must be in (0, 1)")
        if not 2 <= self.k_ccv <= self.n_cal:
            raise ValueError("k_ccv out of range")
        if self.n_test % self.batch_size != 0:
            raise ValueError("n_test must be divisible by batch_size")
        if self.batch_combine not in ("fisher", "fisher_dep"):
            raise ValueError("batch_combine must be 'fisher' or 'fisher_dep'")
        if not 0 <= self.mfdp_s1 < self.mfdp_s2 <= 1:
            raise ValueError("need 0 <= mfdp_s1 < mfdp_s2 <= 1")
        if self.main_alpha not in tuple(self.alpha_grid):
            raise ValueError("main_alpha must be one of alpha_grid")

    @property
    def n_out(self) -> int:
        return int(round(self.prop_out * self.n_test))

    @property
    def n_in(self) -> int:
        return self.n_test - self.n_out

    def replace(self, **kwargs) -> "Config":
        return replace(self, **kwargs)


def default_config(**kwargs) -> Config:
    return Config(**kwargs)


def adjust_label(cfg: Config) -> str:
    """Label used in plot titles. Reads the config rather than being
    hard-coded, so a BH run is never labelled 'mFDP'."""
    return cfg.adjust
