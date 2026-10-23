"""Scoring function and orientation.

sklearn's OneClassSVM.decision_function is positive for inliers, same
convention as e1071, so LOWER = more outlying. Checked rather than assumed;
the flip fixes the library's sign convention, not the method.

nu and gamma come from the config and default to the authors' values
(nu = 0.004, gamma = 0.1). An earlier version used gamma = 1/(d * var(X)),
about 20x smaller, which made the detector far weaker than the paper's and
shifted every power curve to the right.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM

from .config import Config
from .dgp import rP


def fit_scorer(X: np.ndarray, cfg: Config) -> OneClassSVM:
    fit = OneClassSVM(kernel="rbf", nu=cfg.svm_nu, gamma=cfg.svm_gamma)
    fit.fit(X)
    return fit


def score_points(fit: OneClassSVM, X: np.ndarray) -> np.ndarray:
    return fit.decision_function(X)


def check_orientation(cfg: Config, wset: np.ndarray,
                      rng: np.random.Generator, n_probe: int = 2000) -> int:
    """+1 if a higher decision value means more inlying (the expected
    convention), -1 if this build has it inverted."""
    Xtr = rP(cfg.n_train, 1.0, wset, rng)
    f = fit_scorer(Xtr, cfg)
    s_out = score_points(f, rP(n_probe, 3.0, wset, rng)).mean()
    s_in = score_points(f, rP(n_probe, 1.0, wset, rng)).mean()
    if s_out > s_in:
        print("Decision values inverted on this build; flipping sign.")
        return -1
    return 1
