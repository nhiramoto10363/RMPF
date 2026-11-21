"""
Duplicated parents in the merging groups during filtering (appendix table of the paper).

A merging group whose parents are not all distinct spans an affine hull of dimension
k = rank(X P) <= min(n_distinct - 1, d) < n - 1, so the law of a child is the kernel of dimension k
(theory/check_merging_theory.py, check C4), and a group with a single distinct parent (k = 0) only
produces copies of it. This script runs the filters exactly as run_experiments.py does (same seeds
and random streams; the mean RMSE is printed for comparison with results/) and records, at every
assimilation step after the warm-up, the parent indices of every group:

  RMPF:n   groups of n parents of one systematic resampling, randomly permuted
  MPF:k    n = 3 parents from 3 independent (permuted) systematic resamplings, fixed coefficients

Reported per (method, N), averaged over steps and seeds: median ESS, mean number of distinct parents
per group, fraction of groups without duplicates, mean k (rank of the centred group, d-capped),
fraction of children that are exact copies of a parent (groups with one distinct parent).

Usage:  python theory/duplication_stats.py            (about 10 min with 24 threads)
        python theory/duplication_stats.py --quick
Output: results/theory/duplication_<ts>.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpf.benchmarks import BENCHMARKS, ROOT, SEEDS, make_filter  # noqa: E402
from rmpf.filters import batch_resampling, grouped_resampling, merge_fixed  # noqa: E402
from rmpf.rng import set_numba_seed  # noqa: E402

SETTINGS = [  # (benchmark, method, N); n of RMPF: the best n at N in results/ plus n = 3 / 10
    ("l63", "MPF:2", 64), ("l63", "RMPF:3", 64), ("l63", "RMPF:20", 64),
    ("l63", "MPF:2", 1024), ("l63", "RMPF:3", 1024), ("l63", "RMPF:20", 1024),
    ("l63", "MPF:2", 4096), ("l63", "RMPF:3", 4096), ("l63", "RMPF:20", 4096),
    ("l96", "MPF:2", 1024), ("l96", "RMPF:10", 1024), ("l96", "RMPF:80", 1024),
    ("l96", "MPF:2", 4096), ("l96", "RMPF:10", 4096), ("l96", "RMPF:44", 4096),
]


def run(bench, method, N, seed, max_steps=None):
    obs, truth = bench.load_data(max_steps)
    np.random.seed(seed)
    set_numba_seed(seed)
    tr, ob, init = bench.build_models(obs, N)
    f = make_filter(method, N, bench.dim, tr, ob, init.copy())
    rows, rmse = [], []
    for t in range(len(obs)):      # same calls in the same order as rmpf.experiment.run_filter
        if t > 0:
            for _ in range(bench.obs_interval):
                f.forecast()
        f._weight(obs[t])          # as in the filters' filtering()
        if method.startswith("RMPF"):
            idx = grouped_resampling(f.weights, N, f.n)
            parents = f.particles[idx]
            groups = idx[: (N // f.n) * f.n].reshape(-1, f.n)     # complete groups only
            f.particles = np.matmul(f.omegas_t, parents.reshape(f.num_groups, f.n, bench.dim)
                                    ).reshape(-1, bench.dim)[:N]
        else:
            idx = batch_resampling(f.weights, N, f.n)
            groups = idx.reshape(N, f.n)
            f.particles = merge_fixed(f.particles[idx], N, f.n, f.alpha)
        if t < bench.warmup_steps:
            continue
        rmse.append(np.sqrt(np.mean((f.estimate - truth[t]) ** 2)))
        srt = np.sort(groups, axis=1)
        distinct = 1 + np.count_nonzero(np.diff(srt, axis=1), axis=1)
        rows.append((1.0 / np.sum(f.weights**2), distinct.mean(), np.mean(distinct == groups.shape[1]),
                     np.minimum(distinct - 1, bench.dim).mean(), np.mean(distinct == 1)))
    a = np.array(rows)
    return dict(benchmark=bench.name, method=method, n=f.n, num_particles=N, seed=seed,
                rmse=float(np.mean(rmse)), ess_median=float(np.median(a[:, 0])),
                distinct_mean=float(a[:, 1].mean()), no_duplicate=float(a[:, 2].mean()),
                k_upper_mean=float(a[:, 3].mean()), copy_fraction=float(a[:, 4].mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    settings, seeds, max_steps = SETTINGS, SEEDS[:3], None  # Table B1: seeds 42-44
    if args.quick:
        settings, seeds, max_steps = [("l63", "RMPF:3", 1024), ("l96", "RMPF:44", 1024)], [42], 400
    out = []
    for key, method, N in settings:
        for seed in seeds:
            r = run(BENCHMARKS[key], method, N, seed, max_steps)
            out.append(r)
            print(f"{r['benchmark']:9s} {method:8s} N={N:5d} seed={seed}: RMSE {r['rmse']:.3f}  ESS {r['ess_median']:7.1f}"
                  f"  distinct {r['distinct_mean']:5.2f}/{r['n']}  no-dup {r['no_duplicate']:.3f}"
                  f"  k {r['k_upper_mean']:5.2f}  copies {r['copy_fraction']:.4f}", flush=True)
    df = pd.DataFrame(out)
    out_dir = os.path.join(ROOT, "results", "theory")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"duplication_{'quick_' if args.quick else ''}{int(time.time())}.csv")
    df.to_csv(path, index=False)
    summary = df.groupby(["benchmark", "method", "n", "num_particles"], sort=False).mean(numeric_only=True)
    print(summary.drop(columns="seed").round(3).to_string())
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
