"""
Main comparison: PF / MPF / RMPF / Liu-West / RPF / ETKF / EnKF on one benchmark.

Every (method, N, seed) is one run of run_filter over the whole observation series; one CSV row per
run. All runs of a benchmark should be executed in one job on one node so that the computation
times are comparable (see job.sh). Finished (method, N, seed) are skipped, so re-submitting the same
command resumes an interrupted job.

Usage:
    python run_experiments.py l63                       # development grid N = 64, 1024, 4096
    python run_experiments.py l96 --full                # full grid of particle numbers
    python run_experiments.py l96 --counts 1024 --methods RMPF:10 RPF:1
    python run_experiments.py l63 --quick               # smoke test (short series, few methods)
    python run_experiments.py l96d80 --full --seeds 42 --max-steps 1000 --methods RMPF:100   # d = 80, one method
    python run_experiments.py l63 --counts 3 5 10 20 --subdir experiments_smalln   # small ensembles (Appendix C)
Output: results/<benchmark>/<subdir>/runs_<timestamp>_<pid>.csv (subdir: experiments)
"""
import argparse
import glob
import os
import time

import numba
import pandas as pd

from rmpf.benchmarks import BENCHMARKS, DEV_COUNTS, ROOT, SEEDS, method_grid, method_label, parse_method
from rmpf.experiment import run_filter, summarize

QUICK = dict(methods=["PF", "MPF:2", "RMPF:10", "LW:0.9", "RPF:1", "ETKF:1.02", "EnKF:1.02"],
             counts=[1024], seeds=[42], max_steps=400)


def results_dir(bench, quick=False, subdir="experiments"):
    return os.path.join(ROOT, "results", bench.key, "experiments_quick" if quick else subdir)


def finished_runs(out_dir):
    done = set()
    for path in glob.glob(os.path.join(out_dir, "runs_*.csv")):
        for r in pd.read_csv(path).itertuples():
            done.add((method_label(r.method, r.param), int(r.num_particles), int(r.seed)))
    return done


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=sorted(BENCHMARKS))
    p.add_argument("--full", action="store_true", help="full grid of particle numbers (default: N = 64, 1024, 4096)")
    p.add_argument("--counts", type=int, nargs="+", help="particle numbers (overrides --full)")
    p.add_argument("--methods", nargs="+", help="restrict to these methods, e.g. RMPF:10 RPF:1 PF")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--subdir", default="experiments",
                   help="output directory under results/<benchmark>/ (kept apart from the main grid)")
    p.add_argument("--quick", action="store_true", help="smoke test; output goes to experiments_quick/")
    args = p.parse_args()

    bench = BENCHMARKS[args.bench]
    if args.quick:
        args.methods, args.counts, args.seeds, args.max_steps = (
            QUICK["methods"], QUICK["counts"], QUICK["seeds"], QUICK["max_steps"])
    counts = args.counts or (bench.full_counts if args.full else DEV_COUNTS)
    grid = method_grid(bench, counts)
    if args.methods:
        want = {method_label(*parse_method(m)) for m in args.methods}
        grid = [(m, N) for m, N in grid if method_label(*parse_method(m)) in want]

    out_dir = results_dir(bench, args.quick, args.subdir)
    os.makedirs(out_dir, exist_ok=True)
    done = set() if args.quick else finished_runs(out_dir)
    todo = [(m, N, s) for m, N in grid for s in args.seeds if (method_label(*parse_method(m)), N, s) not in done]
    path = os.path.join(out_dir, f"runs_{int(time.time())}_{os.getpid()}.csv")   # several jobs may write to out_dir
    print(f"[{bench.name}] runs to do: {len(todo)} (already done: {len(done)}; "
          f"numba threads {numba.get_num_threads()}) -> {path}", flush=True)

    obs, truth = bench.load_data(args.max_steps)
    for i, (m, N, s) in enumerate(todo, 1):
        row = summarize(bench, m, N, s, run_filter(bench, m, N, s, obs, truth), truth)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
        print(f"[{i}/{len(todo)}] {m:10s} N={N:<6d} seed={s}: RMSE {row['rmse']:.3f} spread {row['spread']:.3f} "
              f"ESS {row['ess']:.1f} filter {1e3 * row['median_filter_time']:.2f} ms "
              f"step {1e3 * row['median_step_time']:.2f} ms{' DIVERGED' if row['diverged'] else ''}", flush=True)


if __name__ == "__main__":
    main()
