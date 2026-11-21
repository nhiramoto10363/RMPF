"""
Effect of outlying particles on one merging step and on the next weighting (reviewer 1, Sect. 6.2.1).

A synthetic weighted ensemble -- a Gaussian bulk N(0, I_d) plus m outliers around D e_1 -- is passed
through one resampling + merging / jittering step of each method (the forecast is omitted, so that
only the merging is examined), and the result is then weighted with one Gaussian observation of a
single coordinate. The merging kernels are those of rmpf.filters, so the step is the one used in
the filters.

Methods (as in run_experiments.py): PF (resampling only), MPF:k (fixed coefficient set k, n = 3),
RMPF:n, LW:h, RPF:c. The ensemble X and its weights are shared by all methods within a repetition.

Merging step (before the second weighting); "child" = particle after merging / jittering
  d_mean, d_cov    error of the mean / relative Frobenius error of the covariance of the children with
                   respect to the weighted mean / covariance of X (RMPF preserves both exactly when n
                   divides N; otherwise the last, padded group breaks this slightly)
  n_touched        children with at least one outlier among their parents
  n_far, n_mirror  children with x_1 > tau (outlier side) and x_1 < -tau (opposite side, where there
                   was no particle); tau = 4 bulk standard deviations
  x1_max, x1_min   extreme values of x_1 among the children
  kurt1            excess kurtosis of x_1 of the children (weighted kurtosis of X: kurt1_ref)
  bulk_var1/2      variance of x_1 / x_2 of the children whose parents are all bulk particles
                   (1 if the bulk is not affected; LW and RPF use the global covariance, which the
                   outliers inflate in the direction e_1)

Next weighting (scenarios; likelihood N(y; x_j, r^2), y noise-free)
  bad_obs     outliers are wrong, their direction is observed:    j = 1, y = 0
  bad_unobs   outliers are wrong, their direction is unobserved:  j = 2, y = 0
  good_obs    outliers are right, their direction is observed:    j = 1, y = D
  ess          ESS / N after the weighting
  mass_far, mass_mirror   posterior weight of the children with x_1 > tau and x_1 < -tau
  mean1, sd1   posterior mean / sd of x_1;  dmean1 = mean1 - ref_mean1, where the reference (ref_*) is
               the exact posterior of the empirical prior (X with weights w * likelihood, i.e. without
               resampling or merging noise). ref_sd1 can be ~0 (good_obs with a single outlier), so sd1
               and ref_sd1 are reported separately rather than as a ratio
  ref_mass_far posterior weight of the reference on x_1 > tau

Usage
  python theory/outlier_merging.py                 # default grid (a few minutes)
  python theory/outlier_merging.py --quick         # small grid for a test
Output: results/outlier/outlier_<timestamp>.csv (one row per repetition, method and scenario) and a
summary (means over repetitions) printed and written to results/outlier/summary_<timestamp>.csv.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpf.benchmarks import MPF_COEFFICIENTS  # noqa: E402
from rmpf.filters import (batch_resampling, grouped_resampling, merge_fixed,  # noqa: E402
                          random_merging_matrices, rpf_optimal_bandwidth, safe_cholesky,
                          systematic_resampling)
from rmpf.rng import set_numba_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 4.0
SCENARIOS = {"bad_obs": (0, "zero"), "bad_unobs": (1, "zero"), "good_obs": (0, "D")}
DEFAULT_METHODS = (["PF"] + ["MPF:%d" % k for k in (1, 2, 3)] + ["RMPF:%d" % n for n in (3, 5, 10, 40, 100)]
                   + ["LW:0.5", "LW:0.9", "RPF:0.5", "RPF:1"])


def make_ensemble(rng, N, d, m, D, outlier_sd, outlier_weight):
    """Bulk N(0, I_d) of N - m particles and m outliers N(D e_1, outlier_sd^2 I); weights (1, outlier_weight)."""
    X = rng.standard_normal((N, d))
    is_out = np.zeros(N, dtype=bool)
    if m > 0:
        is_out[rng.choice(N, m, replace=False)] = True
        X[is_out] *= outlier_sd
        X[is_out, 0] += D
    w = np.where(is_out, outlier_weight, 1.0)
    return X, w / w.sum(), is_out


def merge_step(method, X, w, rng):
    """Children (N, d) and their parents (N, n_parents) after one resampling + merging / jittering step."""
    N, d = X.shape
    name, _, p = method.partition(":")
    if name == "PF":
        idx = systematic_resampling(w, N)
        return X[idx], idx[:, None]
    if name == "MPF":
        alpha = np.asarray(MPF_COEFFICIENTS[int(p)], dtype=np.float64)
        idx = batch_resampling(w, N, len(alpha))
        return merge_fixed(X[idx], N, len(alpha), alpha), idx.reshape(N, len(alpha))
    if name == "RMPF":
        n = int(p)
        G = (N + n - 1) // n
        idx = grouped_resampling(w, N, n)
        children = np.matmul(random_merging_matrices(G, n), X[idx].reshape(G, n, d)).reshape(-1, d)[:N]
        return children, np.repeat(idx.reshape(G, n), n, axis=0)[:N]
    if name in ("LW", "RPF"):
        mean = w @ X
        A = X - mean
        chol = safe_cholesky((A * w[:, None]).T @ A)
        idx = systematic_resampling(w, N)
        eps = rng.standard_normal((N, d))
        if name == "LW":
            h = float(p)
            a = np.sqrt(1.0 - h**2)
            centers = a * X[idx] + (1.0 - a) * mean
        else:
            h = float(p) * rpf_optimal_bandwidth(N, d)
            centers = X[idx]
        return centers + h * (eps @ chol.T), idx[:, None]
    raise ValueError(method)


def weighted_moments(x, w):
    m = w @ x
    v = w @ (x - m) ** 2
    return m, np.sqrt(v), (w @ (x - m) ** 4) / v**2 - 3.0


def merge_metrics(children, parents, X, w, is_out):
    mean_w = w @ X
    A = X - mean_w
    cov_w = (A * w[:, None]).T @ A
    cov_c = np.cov(children.T, bias=True)
    touched = is_out[parents].any(axis=1)
    bulk = children[~touched]
    x1 = children[:, 0]
    return dict(
        d_mean=float(np.linalg.norm(children.mean(axis=0) - mean_w)),
        d_cov=float(np.linalg.norm(cov_c - cov_w) / np.linalg.norm(cov_w)),
        n_touched=int(touched.sum()), n_far=int((x1 > TAU).sum()), n_mirror=int((x1 < -TAU).sum()),
        x1_max=float(x1.max()), x1_min=float(x1.min()),
        kurt1=float(weighted_moments(x1, np.full(len(x1), 1.0 / len(x1)))[2]),
        kurt1_ref=float(weighted_moments(X[:, 0], w)[2]),
        bulk_var1=float(bulk[:, 0].var()) if len(bulk) > 1 else np.nan,
        bulk_var2=float(bulk[:, 1].var()) if len(bulk) > 1 else np.nan,
    )


def posterior_weights(x, prior_w, j, y, r):
    ll = -0.5 * ((x[:, j] - y) / r) ** 2
    u = prior_w * np.exp(ll - ll.max())
    return u / u.sum()


def weight_metrics(children, X, w, j, y, r):
    N = len(children)
    w2 = posterior_weights(children, np.full(N, 1.0 / N), j, y, r)
    wr = posterior_weights(X, w, j, y, r)
    x1 = children[:, 0]
    m1, s1, _ = weighted_moments(x1, w2)
    rm1, rs1, _ = weighted_moments(X[:, 0], wr)
    return dict(ess=float(1.0 / np.sum(w2**2) / N),
                mass_far=float(w2[x1 > TAU].sum()), mass_mirror=float(w2[x1 < -TAU].sum()),
                mean1=float(m1), sd1=float(s1), dmean1=float(m1 - rm1), ref_sd1=float(rs1),
                ref_mass_far=float(wr[X[:, 0] > TAU].sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", type=int, nargs="+", default=[3, 40])
    p.add_argument("--counts", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--distances", type=float, nargs="+", default=[10.0])
    p.add_argument("--outliers", nargs="+", default=["0", "1", "1%"],
                   help="number of outliers m; 'p%%' means p percent of N")
    p.add_argument("--outlier-weights", type=float, nargs="+", default=[1.0, 10.0],
                   help="weight of an outlier relative to a bulk particle before resampling")
    p.add_argument("--outlier-sd", type=float, default=1.0)
    p.add_argument("--obs-std", type=float, default=1.0)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="d = 3, N = 256, 10 repetitions")
    args = p.parse_args()
    if args.quick:
        args.dims, args.counts, args.reps = [3], [256], 10

    rows = []
    t0 = time.time()
    for d in args.dims:
        for N in args.counts:
            for D in args.distances:
                for m_spec in args.outliers:
                    m = int(round(N * float(m_spec[:-1]) / 100)) if m_spec.endswith("%") else int(m_spec)
                    for ow in (args.outlier_weights if m > 0 else [1.0]):
                        config = dict(d=d, N=N, D=D, m=m, m_spec=m_spec, outlier_weight=ow)
                        for rep in range(args.reps):
                            rng = np.random.default_rng([args.seed, d, N, int(D * 100), m, int(ow * 100), rep])
                            X, w, is_out = make_ensemble(rng, N, d, m, D, args.outlier_sd, ow)
                            for k, method in enumerate(args.methods):
                                set_numba_seed(int(rng.integers(2**31 - 1)))
                                children, parents = merge_step(method, X, w, rng)
                                mm = merge_metrics(children, parents, X, w, is_out)
                                for sc, (j, ytype) in SCENARIOS.items():
                                    y = 0.0 if ytype == "zero" else D
                                    rows.append(dict(config, rep=rep, method=method, scenario=sc, **mm,
                                                     **weight_metrics(children, X, w, j, y, args.obs_std)))
                        print(f"d={d} N={N} D={D:g} m={m} w_out={ow:g}: done ({time.time() - t0:.0f} s)", flush=True)

    out_dir = os.path.join(ROOT, "results", "outlier")
    os.makedirs(out_dir, exist_ok=True)
    stamp = "%d_%d" % (time.time(), os.getpid())
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "outlier_%s.csv" % stamp), index=False)

    keys = ["d", "N", "D", "m", "outlier_weight", "scenario", "method"]
    summary = df.groupby(keys, sort=False).mean(numeric_only=True).drop(columns="rep").reset_index()
    summary.to_csv(os.path.join(out_dir, "summary_%s.csv" % stamp), index=False)
    cols = ["n_touched", "n_far", "n_mirror", "x1_max", "x1_min", "kurt1", "bulk_var1", "d_cov",
            "ess", "mass_far", "mass_mirror", "dmean1", "sd1", "ref_sd1"]
    with pd.option_context("display.width", 250, "display.max_rows", None, "display.float_format", "{:.3g}".format):
        for (d, N, D, m, ow, sc), s in summary.groupby(keys[:-1], sort=False):
            print(f"\n== d={d} N={N} D={D:g} m={m} w_out={ow:g} scenario={sc} "
                  f"(kurt1_ref {s.kurt1_ref.iloc[0]:.3g}, ref_mass_far {s.ref_mass_far.iloc[0]:.3g})")
            print(s.set_index("method")[cols].to_string())
    print("\nwritten:", out_dir, stamp)


if __name__ == "__main__":
    main()
