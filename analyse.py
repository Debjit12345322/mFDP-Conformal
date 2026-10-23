#!/usr/bin/env python3
"""Figures and tables for the paper, from finished runs. Nothing is re-simulated.

    python analyse.py --bh results/bh --mfdp results/mfdp --batch results/batch

Section 3 (individual detection), written to --out (default results/paper):
    fig_fdp.png         BH cFDR and mFDP cmFDP across practitioners, per signal
                        strength, at the main level
    fig_power.png       cPower for both, per signal strength
    fig_levels.png      mFDR (BH) and MFDP (mFDP), and mPower, against the
                        nominal level, for a few signal strengths
    table_section3.csv  one row per signal strength at the main level

Section 4 (batch detection), if --batch is given:
    table_section4.csv  one row per (procedure, level)

All quantities use the definitions in conformal_outliers/summarise.py:
cFDR / mFDR (mean over the L test sets / over everything) for BH and
Storey-BH, cmFDP (median over L) and the pooled MFDP for mFDP, and cPower /
mPower (means) for every procedure.
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from conformal_outliers.plots import dodged_boxes  # noqa: E402

METHOD = "Marginal"


def load(run_dir: str):
    prac = pd.read_csv(os.path.join(run_dir, "per_practitioner.csv"))
    summ = pd.read_csv(os.path.join(run_dir, "summary.csv"))
    with open(os.path.join(run_dir, "results.pkl"), "rb") as f:
        params = pickle.load(f)["params"]
    keep = prac["method"] == METHOD
    return prac[keep], summ[summ["method"] == METHOD], params


def section3(bh_dir: str, mfdp_dir: str, out: str, level: float | None,
             a_lines: list[float]) -> None:
    pb, sb, par_b = load(bh_dir)
    pm, sm, par_m = load(mfdp_dir)
    if par_b["adjust"] != "BH" or par_m["adjust"] != "mFDP":
        raise SystemExit("--bh must be a BH run and --mfdp an mFDP run")
    for k in ("J", "L", "seed", "a_grid", "n_cal", "n_test", "prop_out"):
        if par_b[k] != par_m[k]:
            print(f"warning: the two runs differ in {k}: {par_b[k]} vs {par_m[k]}")
    level = level if level is not None else par_b["main_alpha"]
    J = par_b["J"]
    a_grid = list(par_b["a_grid"])

    at = lambda d: d[np.isclose(d["alpha"], level)]
    pb_l, pm_l = at(pb), at(pm)

    # --- FDP and power figures -------------------------------------------
    for fname, cols, ylabel, hline in (
        ("fig_fdp.png", ("cFDR", "cmFDP"), "FDP summary per practitioner", True),
        ("fig_power.png", ("cPower", "cPower"), "cPower", False),
    ):
        series = [(pb_l, cols[0]), (pm_l, cols[1])]
        labels = ([f"BH ({cols[0]})", f"mFDP ({cols[1]})"] if hline
                  else ["BH", "mFDP"])

        def data_of(si, a):
            d, c = series[si]
            return d.loc[d["a"] == a, c].dropna().values

        fig, ax = plt.subplots(figsize=(9, 4.2))
        dodged_boxes(ax, a_grid, None, data_of, labels)
        if hline:
            ax.axhline(level, ls="--", color="grey")
        ax.set_xlabel("Signal strength $a$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Marginal conformal p-values, {J} practitioners, "
                     f"level {level:g}")
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=200)
        plt.close(fig)

    # --- levels figure -----------------------------------------------------
    levels = sorted(set(sb["alpha"]).intersection(set(sm["alpha"])))
    a_lines = [a for a in a_lines if a in a_grid]
    if len(levels) > 1 and a_lines:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for i, a in enumerate(a_lines):
            col = plt.cm.tab10.colors[i]
            b = sb[sb["a"] == a].sort_values("alpha")
            m = sm[sm["a"] == a].sort_values("alpha")
            axes[0].plot(b["alpha"], b["mFDR"], "o-", color=col, label=f"BH mFDR, a={a:g}")
            axes[0].plot(m["alpha"], m["MFDP"], "s--", color=col, label=f"mFDP MFDP, a={a:g}")
            axes[1].plot(b["alpha"], b["mPower"], "o-", color=col, label=f"BH, a={a:g}")
            axes[1].plot(m["alpha"], m["mPower"], "s--", color=col, label=f"mFDP, a={a:g}")
        axes[0].plot(levels, levels, ":", color="grey")
        axes[0].set_xlabel("nominal level")
        axes[0].set_ylabel("mFDR (BH) / MFDP (mFDP)")
        axes[1].set_xlabel("nominal level")
        axes[1].set_ylabel("mPower")
        for ax in axes:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_levels.png"), dpi=200)
        plt.close(fig)

    # --- table -------------------------------------------------------------
    b = at(sb).set_index("a")
    m = at(sm).set_index("a")
    table = pd.DataFrame({
        "BH_mFDR": b["mFDR"],
        "BH_cFDR_q90": b["cFDR_q90"],
        "BH_frac_cFDR_above": b["frac_cFDR_above"],
        "BH_mPower": b["mPower"],
        "mFDP_MFDP": m["MFDP"],
        "mFDP_share_FDP_le_level": m["share_FDP_le_alpha"],
        "mFDP_cmFDP_median": m["cmFDP_median"],
        "mFDP_cmFDP_q90": m["cmFDP_q90"],
        "mFDP_frac_cmFDP_above": m["frac_cmFDP_above"],
        "mFDP_mPower": m["mPower"],
    }).reset_index()
    table.to_csv(os.path.join(out, "table_section3.csv"), index=False)
    print(f"Section 3, level {level:g}:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def section4(batch_dir: str, out: str) -> None:
    s = pd.read_csv(os.path.join(batch_dir, "batch_summary.csv"))
    s = s[s["method"] == METHOD]
    rows = []
    for r in s.itertuples():
        mfdp = r.procedure == "mFDP"
        rows.append({
            "procedure": r.procedure,
            "level": r.alpha,
            "FDP_summary": "MFDP" if mfdp else "mFDR",
            "FDP_value": r.MFDP if mfdp else r.mFDR,
            "share_FDP_le_level": r.share_FDP_le_alpha,
            "frac_practitioners_above": (r.frac_cmFDP_above if mfdp
                                         else r.frac_cFDR_above),
            "mPower": r.mPower,
        })
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_section4.csv"), index=False)
    print("\nSection 4 (batch detection):")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fw = os.path.join(batch_dir, "batch_fwer_global_null.csv")
    if os.path.exists(fw):
        f = pd.read_csv(fw)
        f = f[f["method"] == METHOD]
        g = (f.groupby("batch_size")["FWER"]
               .agg(median="median", q90=lambda x: x.quantile(0.9), max="max")
               .reset_index())
        g.to_csv(os.path.join(out, "table_section4_fwer.csv"), index=False)
        print("\nFisher's test under the global null (FWER across practitioners):")
        print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bh", default="results/bh")
    ap.add_argument("--mfdp", default="results/mfdp")
    ap.add_argument("--batch", default=None)
    ap.add_argument("--out", default="results/paper")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--a-lines", type=float, nargs="+", default=[2.0, 2.5, 3.0],
                    dest="a_lines", help="signal strengths drawn in fig_levels")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    section3(a.bh, a.mfdp, a.out, a.level, a.a_lines)
    if a.batch:
        section4(a.batch, a.out)
    print(f"\nwritten to {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
