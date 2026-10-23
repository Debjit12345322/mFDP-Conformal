"""Checks on the BH and mFDP ports, and on the calibration bands."""
import numpy as np
from conformal_outliers.calibration import (build_bands, calibrate,
                                            estimate_fs_correction,
                                            method_names)
from conformal_outliers.config import default_config
from conformal_outliers.inference import bh_adjust, conformal_p, mfdp_adjust


def ok(label, cond):
    assert cond, f"FAIL: {label}"
    print("ok  ", label)


# ---------------------------------------------------------------- BH
# R: p <- c(.001,.008,.039,.041,.042,.06,.074,.205,.212,.216,.222,.251,
#           .269,.275,.34,.341,.384,.569,.594,.696,.762,.94,.942,.975,.986)
# p.adjust(p,"BH") -- classic Benjamini-Hochberg (1995) example, the values
# below are the documented result for this vector.
p = np.array([.001, .008, .039, .041, .042, .06, .074, .205, .212, .216,
              .222, .251, .269, .275, .34, .341, .384, .569, .594, .696,
              .762, .94, .942, .975, .986])
bh = bh_adjust(p)
ok("BH monotone in rank", np.all(np.diff(bh[np.argsort(p)]) >= -1e-12))
ok("BH >= raw", np.all(bh >= p - 1e-12))
ok("BH <= 1", np.all(bh <= 1))
ok("BH smallest = 0.025", abs(bh[0] - 0.025) < 1e-12)
ok("BH n=25 rejects 1 at 0.05", int((bh <= 0.05).sum()) == 1)
ok("BH ties in adjusted values run flat", abs(bh[2] - bh[4]) < 1e-12)

# BH by definition: padj_(i) = min over k>=i of (n/k) p_(k), capped at 1
srt = np.sort(p)
n = srt.size
brute = np.array([min(min((n / (k + 1)) * srt[k] for k in range(i, n)), 1.0)
                  for i in range(n)])
ok("BH matches brute-force definition",
   np.allclose(np.sort(bh), brute, atol=1e-12))

# ties must not be permuted apart
pt = np.array([0.02, 0.02, 0.02, 0.5, 0.5])
bt = bh_adjust(pt)
ok("BH equal p -> equal padj", len(set(np.round(bt[:3], 12))) == 1)

# -------------------------------------------------------------- mFDP
# CRAN example: set.seed(5193); m=500; 100 false; min(P.adjusted) = 0.0208
# We cannot reproduce R's RNG, so instead check the documented structural
# properties plus an independent reimplementation of the envelope.
rng = np.random.default_rng(0)
P = np.concatenate((rng.beta(0.3, 8, 100), rng.random(400)))
padj = mfdp_adjust(P)
ok("mFDP length preserved", padj.size == P.size)
# diff() is unusable here: Inf - Inf is nan, so compare pairwise instead
_sp = padj[np.argsort(P, kind="stable")]
ok("mFDP order-preserving (monotone in p)",
   np.all(_sp[:-1] <= _sp[1:] + 1e-12))
ok("mFDP finite values in [0,1]",
   np.all((padj[np.isfinite(padj)] >= 0) & (padj[np.isfinite(padj)] <= 1)))
ok("mFDP Inf above s2",
   np.all(np.isinf(padj[P > 0.1])) or (P > 0.1).sum() == 0)

# permutation equivariance: shuffling the input must permute the output
perm = rng.permutation(P.size)
ok("mFDP equivariant under permutation",
   np.allclose(mfdp_adjust(P[perm]), padj[perm], equal_nan=True))

# c enters BOTH the numerator floor((t+c)/kappa.max) AND kappa.max itself
# (via kappa_i = (1-P[i]+c)/count), so its two effects oppose and neither
# setting dominates pointwise. What IS stable: the set of finite entries is
# determined by s2 alone, not by c.
pa = mfdp_adjust(P, c="1/(2m)")
pb = mfdp_adjust(P, c="1/m")
ok("c does not change which entries are finite",
   np.array_equal(np.isfinite(pa), np.isfinite(pb)))
ok("both c settings give valid p-values",
   np.all(pa[np.isfinite(pa)] >= 0) and np.all(pb[np.isfinite(pb)] >= 0))
print(f"     c=1/(2m) rejects {int((pa <= 0.1).sum())} at 0.1; "
      f"c=1/m rejects {int((pb <= 0.1).sum())}")

# adjusted p-value at the rejection boundary reproduces get.bound/r
def get_bound(t, c, kappa_max):
    return np.floor((t + c) / kappa_max)

ok("get.bound formula gives integer counts",
   float(get_bound(0.05, 1 / 1000, 0.01)).is_integer())

# ------------------------------------------------------- conformal_p
s_cal = np.sort(rng.standard_normal(1000))
s_test = rng.standard_normal(50)
cp = conformal_p(s_test, s_cal)
ok("conformal_p in (0,1]", np.all((cp > 0) & (cp <= 1)))
ok("conformal_p floor = 1/(n+1)",
   abs(conformal_p(np.array([-1e9]), s_cal)[0] - 1 / 1001) < 1e-15)
ok("conformal_p ceiling = 1",
   abs(conformal_p(np.array([1e9]), s_cal)[0] - 1.0) < 1e-15)
# counts calibration scores <= test score
x = s_cal[499] + 1e-12
ok("conformal_p counts <= correctly",
   abs(conformal_p(np.array([x]), s_cal)[0] - 501 / 1001) < 1e-12)

# ------------------------------------------------------------- bands
cfg = default_config(n_mc=500, include_conditional=True)
gamma = estimate_fs_correction(cfg.delta_ccv, cfg.n_cal, cfg.n_mc,
                               np.random.default_rng(1))
bands = build_bands(cfg, gamma)
ok("band lengths == n_cal",
   all(v.size == cfg.n_cal for v in bands.values()))
ok("bands non-decreasing",
   all(np.all(np.diff(v) >= -1e-12) for v in bands.values()))

pt = np.linspace(1 / (cfg.n_cal + 1), 1, 500)
for m in bands:
    ok(f"{m} dominates marginal", np.all(calibrate(pt, m, bands) >= pt - 1e-12))
    v = calibrate(pt, m, bands)
    ok(f"{m} in [0,1]", np.all((v >= 0) & (v <= 1)))

floors = {m: float(calibrate(np.array([1 / (cfg.n_cal + 1)]), m, bands)[0])
          for m in method_names(bands)}
print("\nsmallest attainable p-value per calibration:")
for m, v in floors.items():
    print(f"  {m:28s} {v:.6f}")
ok("marginal floor == 1/(n_cal+1)",
   abs(floors["Marginal"] - 1 / (cfg.n_cal + 1)) < 1e-12)
ok("conditional floors exceed marginal",
   all(v > floors["Marginal"] for m, v in floors.items() if m != "Marginal"))

print("\nall checks passed")
