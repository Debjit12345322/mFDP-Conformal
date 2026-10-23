"""Per-run plots: the distribution across the J practitioners at each signal
strength, at the main level. Each procedure is shown on its own guarantee:
cFDR (mean over the L test sets) for BH, cmFDP (median over L) for mFDP.
Power is cPower (mean over L) for both, so runs can be compared.

The combined BH-versus-mFDP figures for the paper are made by analyse.py.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .calibration import method_names  # noqa: E402
from .config import Config  # noqa: E402
from .summarise import practitioner_summary  # noqa: E402


def fdr_column(procedure: str) -> tuple[str, str]:
    """(column, label) of the FDP summary matching the procedure's guarantee."""
    return ("cmFDP", "cmFDP") if procedure == "mFDP" else ("cFDR", "cFDR")


def dodged_boxes(ax, groups, positions_of, data_of, labels, width=0.8,
                 colours=None):
    """Draw one box per (x position, series), side by side."""
    k = len(labels)
    w = width / k
    handles = []
    for si, lab in enumerate(labels):
        data = [data_of(si, g) for g in groups]
        data = [d if d.size else np.array([np.nan]) for d in data]
        pos = np.arange(len(groups)) + (si - (k - 1) / 2) * w
        bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True,
                        flierprops={"markersize": 2}, manage_ticks=False)
        col = (colours or plt.cm.tab10.colors)[si % 10]
        for b in bp["boxes"]:
            b.set_facecolor(col)
            b.set_alpha(0.6)
        handles.append(bp["boxes"][0])
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([f"{g:g}" for g in groups])
    if k > 1:
        ax.legend(handles, labels, fontsize=8)


def make_plots(ts: pd.DataFrame, cfg: Config, bands: dict) -> list[str]:
    methods = method_names(bands)
    prac = practitioner_summary(ts)
    prac = prac[np.isclose(prac["alpha"], cfg.main_alpha)]
    a_grid = list(cfg.a_grid)
    col, lab = fdr_column(cfg.adjust)

    def data_of(column):
        return lambda si, a: prac.loc[(prac["method"] == methods[si]) &
                                      (prac["a"] == a), column].dropna().values

    files = []
    for column, ylabel, fname, hline in ((col, lab, "fdp.png", True),
                                         ("cPower", "cPower", "power.png", False)):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        dodged_boxes(ax, a_grid, None, data_of(column), methods)
        if hline:
            ax.axhline(cfg.main_alpha, ls="--", color="grey")
        ax.set_xlabel("Signal strength")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} across {cfg.J} practitioners "
                     f"({cfg.adjust}, level = {cfg.main_alpha:g})")
        fig.tight_layout()
        path = os.path.join(cfg.outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(path)
    return files
