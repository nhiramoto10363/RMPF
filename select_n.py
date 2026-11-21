"""
Selection of the merge size n by Particle Reallocation Search (PRS).

For a fixed particle budget N, n is chosen by maximizing the log marginal likelihood logZ of the
RMPF. The process and observation noise are known; the truth is never used for the selection.

  inner (screening): M candidate values of n run simultaneously with N/M particles each, under
      common random numbers (same initial ensemble, process noise and resampling uniforms). The
      series is split into segments; at each boundary the top half by logZ survives and the freed
      particles are reallocated to the survivors by deterministic replication (no restart).
  outer (zoom): one-dimensional search on log2(n); the interval is re-centered on the incumbent,
      shrunk by gamma after an improvement and widened by psi otherwise. Candidates are rounded to
      integers; if the interval holds at most M integers, all of them are used.
PRS hyperparameters are those of the PRS paper (eta = 0.5, gamma = 0.5, psi = 1.2, 10 rounds;
M = 16 for Lorenz-63 and 9 for Lorenz-96).

Validation: the search uses only the first half of the series. Every candidate of the grid
(--grid, default: the merge sizes of the main comparison) and the selected n are then run with N
particles and independent seeds (default 42-51) over the whole series; the RMSE over the second half (not
used by the search) is reported. These evaluation runs are identical to the runs of
run_experiments.py with the same (n, N, seed).

Usage:
    python select_n.py l96                          # N = 1024, 4096: grid + PRS + validation
    python select_n.py l63 --budgets 4096 --no-grid
    python select_n.py l63 --quick                  # smoke test
    python select_n.py l96 --dim 20 --budgets 32768 --grid 2 3 5 10 20 30 40 50 60 70 80 90 100 110 120 \
        --eval-seeds 42 43 44 45 46 47 48 49 50 51 --seed 1 2 3 4 5 6 7 8 9 10
                                                    # Lorenz-96 with d = 20 (selected n against d), 10 searches
Several search seeds (--seed 1 2 ...) give independent searches (different common random numbers); each
is saved under its own tag. Rerunning the same command resumes: evaluations (n, N, seed) already in an
eval_*.csv of the output folder and searches (N, seed) already in a prs_rounds_*.csv are skipped. The
evaluation of a selected n that is also on the grid is therefore not repeated.
Output: results/<benchmark>/select_n/
    prs_rounds_N<N>_s<seed>_<ts>.csv       search interval, winner and incumbent of every round, cost
    prs_candidates_N<N>_s<seed>_<ts>.csv   screening logZ of every candidate
    eval_<ts>.csv                          evaluation runs (role = grid | selected), one row per (n, N, seed)
"""
import argparse
import glob
import os
import time

import numba
import numpy as np
import pandas as pd

from rmpf.benchmarks import BENCHMARKS, ROOT, SEEDS, lorenz96_with_dim
from rmpf.experiment import climatological_std, run_filter
from rmpf.filters import RMPF, random_merging_matrices
from rmpf.rng import set_numba_seed

ETA_KEEP = 0.5     # fraction kept at each segment boundary
GAMMA = 0.5        # width factor after an improvement
EXPAND = 1.2       # width factor psi otherwise
ROUNDS = 10
WIDTH_FLOOR = 1e-4
SEARCH_SETTINGS = {"l63": dict(M=16, n_bounds=(2, 128)), "l96": dict(M=9, n_bounds=(2, 128))}
SEARCH_SEED = 1    # seed of the common random numbers of the search (disjoint from the evaluation seeds)
DEFAULT_BUDGETS = [1024, 4096]   # N = 64 leaves only 4-7 particles per candidate


def _mix(*xs):
    """Deterministic seed from a tuple of integers."""
    return int(np.random.SeedSequence([int(x) for x in xs]).generate_state(1)[0] & 0x7FFFFFFF)


def reallocate(pf, n_new, seed):
    """Grow the RMPF to n_new particles by deterministic replication (tile = arange(n_new) % N)."""
    n_old = pf.num_particles
    if n_new == n_old:
        return
    pf.particles = pf.particles[np.arange(n_new) % n_old].copy()
    pf.num_particles = n_new
    set_numba_seed(seed)
    g_new = (n_new + pf.n - 1) // pf.n
    if g_new > pf.num_groups:   # existing merging matrices are kept; only the new groups get fresh ones
        pf.omegas_t = np.ascontiguousarray(
            np.concatenate([pf.omegas_t, random_merging_matrices(g_new - pf.num_groups, pf.n)]))
    pf.num_groups = g_new


# =====================================================================
# inner: screening
# =====================================================================
def segment_bounds(T, n_ck):
    ck = sorted(set(int(round(T * k / (n_ck + 1))) for k in range(1, n_ck + 1)))
    b = [0] + [c for c in ck if 0 < c < T] + [T]
    return list(zip(b[:-1], b[1:]))


def default_checkpoints(M, eta=ETA_KEEP):
    return int(np.ceil(np.log(M) / np.log(1 / eta))) if M > 1 else 0


def inner_screen(bench, n_list, N, obs, round_seed, eta=ETA_KEEP):
    """Run the candidates n_list on budget N simultaneously; return the one with the largest logZ."""
    T, M = len(obs), len(n_list)
    n0 = max(1, N // M)
    pfs = []
    for n in n_list:
        np.random.seed(round_seed)
        set_numba_seed(round_seed)
        transition, observation, initial = bench.build_models(obs, n0)
        set_numba_seed(_mix(round_seed, n, 0))
        pfs.append(RMPF(n0, bench.dim, transition, observation, initial.copy(), n=int(n)))

    logZ = np.zeros(M)
    active = list(range(M))
    segs = segment_bounds(T, default_checkpoints(M, eta))
    cost = 0
    for si, (t0, t1) in enumerate(segs):
        n_part = pfs[active[0]].num_particles
        for t in range(t0, t1):
            for m in active:
                pf = pfs[m]
                if t > 0:
                    for _ in range(bench.obs_interval):
                        pf.forecast()
                else:
                    set_numba_seed(round_seed)   # same resampling uniforms for every candidate
                pf.filtering(obs[t])
                logZ[m] += pf.log_lik_increment
        cost += len(active) * n_part * (t1 - t0)

        if si < len(segs) - 1 and len(active) > 1:
            keep = max(1, int(np.ceil(eta * len(active))))
            order = sorted(active, key=lambda m: logZ[m], reverse=True)
            active, dropped = order[:keep], order[keep:]
            n_new = max(n_part, N // keep)
            for m in active:
                reallocate(pfs[m], n_new, _mix(round_seed, n_list[m], si + 1))
            for m in dropped:
                pfs[m] = None

    win = max(active, key=lambda m: logZ[m])
    return dict(winner=int(n_list[win]), winner_logZ=float(logZ[win]), logZ_all=logZ, cost=int(cost))


# =====================================================================
# outer: zoom on log2(n)
# =====================================================================
def _box(center, width, bounds):
    return max(bounds[0], center - width / 2), min(bounds[1], center + width / 2)


def sample_candidates(center, width, M, bounds, rng):
    """At most M integer candidates in [2^lo, 2^hi]: all of them if few, else rounded LHS draws on log2(n)."""
    lo, hi = _box(center, width, bounds)
    ints = np.arange(int(np.ceil(2**lo - 1e-9)), int(np.floor(2**hi + 1e-9)) + 1)
    if len(ints) == 0:
        return [int(round(2**center))]
    if len(ints) <= M:
        return [int(x) for x in ints]
    u = (np.arange(M) + rng.random(M)) / M
    cand = set(int(np.clip(round(2 ** (lo + ui * (hi - lo))), ints[0], ints[-1])) for ui in u)
    rest = [x for x in ints if x not in cand]
    if len(cand) < M:
        cand.update(int(x) for x in rng.choice(rest, M - len(cand), replace=False))
    return sorted(cand)


def zoom_search(bench, obs, N, M, n_bounds, rounds=ROUNDS, seed=SEARCH_SEED):
    bounds = (np.log2(n_bounds[0]), np.log2(n_bounds[1]))
    center, width = (bounds[0] + bounds[1]) / 2, bounds[1] - bounds[0]
    inc_n, inc_logZ = None, -np.inf
    history = []
    for r in range(rounds):
        rng = np.random.default_rng(seed + 100 * (r + 1))
        n_list = sample_candidates(center, width, M, bounds, rng)
        t0 = time.time()
        res = inner_screen(bench, n_list, N, obs, _mix(seed, r))
        improved = inc_n is None or res["winner_logZ"] > inc_logZ
        if improved:
            inc_n, inc_logZ = res["winner"], res["winner_logZ"]
        lo, hi = _box(center, width, bounds)
        history.append(dict(round=r, n_lo=2**lo, n_hi=2**hi, candidates=n_list, logZ_all=res["logZ_all"],
                            winner=res["winner"], winner_logZ=res["winner_logZ"], improved=improved,
                            inc_n=inc_n, inc_logZ=inc_logZ, cost=res["cost"], wall=time.time() - t0))
        print(f"[round {r + 1}/{rounds}] n in [{2**lo:.1f}, {2**hi:.1f}] cand={n_list}  winner n={res['winner']} "
              f"logZ={res['winner_logZ']:.2f} [{'+' if improved else '.'}] | incumbent n={inc_n} "
              f"({time.time() - t0:.0f}s)", flush=True)
        center = float(np.clip(np.log2(inc_n), bounds[0], bounds[1]))
        width = max(WIDTH_FLOOR, width * (GAMMA if improved else EXPAND))
    return dict(inc_n=inc_n, inc_logZ=inc_logZ, history=history)


# =====================================================================
# evaluation
# =====================================================================
def evaluate(bench, n, N, seed, obs, truth, t_split):
    t0 = time.time()
    out = run_filter(bench, f"RMPF:{n}", N, seed, obs, truth)
    T, w = len(obs), min(bench.warmup_steps, len(obs))
    clim = climatological_std(truth)
    rmse = lambda a, b: float(np.mean(out["rmse"][a:b])) if b > a else np.nan
    row = dict(benchmark=bench.key, dim=bench.dim, n=int(n), num_particles=int(N), seed=int(seed), num_steps=T, t_split=int(t_split),
               logZ_search=float(out["log_lik"][:t_split].sum()), logZ_holdout=float(out["log_lik"][t_split:].sum()),
               rmse_search=rmse(w, t_split), rmse_holdout=rmse(t_split, T), rmse=rmse(w, T),
               climatological_std=clim, total_time=time.time() - t0, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    # rmse_search is NaN when the search period lies inside the warm-up (only with --quick)
    row["diverged"] = bool(not np.isfinite(row["rmse_holdout"]) or row["rmse_holdout"] > clim
                           or row["rmse_search"] > clim)
    return row


def finished_evaluations(out_dir):
    """(n, N, seed) of the evaluation runs already saved in out_dir."""
    done = set()
    for f in glob.glob(os.path.join(out_dir, "eval_*.csv")):
        df = pd.read_csv(f)
        done.update(zip(df.n.astype(int), df.num_particles.astype(int), df.seed.astype(int)))
    return done


def finished_search(out_dir, N, seed):
    """Selected n of a search (N, seed) already saved in out_dir, or None."""
    for f in sorted(glob.glob(os.path.join(out_dir, f"prs_rounds_N{N}_*.csv"))):
        df = pd.read_csv(f)
        if len(df) and int(df.seed.iloc[0]) == seed:
            return int(df.inc_n.iloc[-1])
    return None


def run_evaluations(bench, n_list, N, seeds, obs, truth, t_split, path, role, done=frozenset()):
    rows = []
    for n in n_list:
        for s in seeds:
            if (int(n), int(N), int(s)) in done:
                print(f"  [{role}] n={n:3d} N={N} seed={s}: already done", flush=True)
                continue
            row = dict(evaluate(bench, n, N, s, obs, truth, t_split), role=role)
            rows.append(row)
            pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            print(f"  [{role}] n={n:3d} N={N} seed={s}: logZ_search={row['logZ_search']:.1f} "
                  f"RMSE search={row['rmse_search']:.3f} holdout={row['rmse_holdout']:.3f}", flush=True)
    return pd.DataFrame(rows)


def search(bench, obs, truth, N, M, n_bounds, t_split, seed, args, out_dir, eval_path, ts):
    """One PRS search with seed `seed` on budget N, followed by the evaluation of the selected n."""
    n_sel = finished_search(out_dir, N, seed)
    if n_sel is not None:
        print(f"\n=== PRS: N={N}, seed={seed}: already done (selected n={n_sel}) ===", flush=True)
    else:
        lo, hi = n_bounds[0], min(n_bounds[1], max(2, N // M))
        print(f"\n=== PRS: N={N}, seed={seed}, M={M}, rounds={args.rounds}, n in [{lo}, {hi}] ===", flush=True)
        t0 = time.time()
        res = zoom_search(bench, obs[:t_split], N, M, (lo, hi), rounds=args.rounds, seed=seed)
        cost = sum(h["cost"] for h in res["history"])
        print(f"selected n={res['inc_n']}  wall={time.time() - t0:.0f}s  cost={cost:,} particle-steps "
              f"= {cost / (N * t_split):.2f} x one run of N particles over the search period", flush=True)
        tag = f"N{N}_s{seed}_{ts}"
        pd.DataFrame([dict(round=h["round"], n_lo=h["n_lo"], n_hi=h["n_hi"],
                           candidates=" ".join(map(str, h["candidates"])), winner=h["winner"],
                           winner_logZ=h["winner_logZ"], improved=h["improved"], inc_n=h["inc_n"],
                           inc_logZ=h["inc_logZ"], cost=h["cost"], wall=h["wall"], budget=N, M=M,
                           t_split=t_split, seed=seed) for h in res["history"]]
                     ).to_csv(os.path.join(out_dir, f"prs_rounds_{tag}.csv"), index=False)
        pd.DataFrame([dict(round=h["round"], n=n, logZ_screen=float(h["logZ_all"][i]), is_winner=n == h["winner"])
                      for h in res["history"] for i, n in enumerate(h["candidates"])]
                     ).to_csv(os.path.join(out_dir, f"prs_candidates_{tag}.csv"), index=False)
        n_sel = res["inc_n"]
    run_evaluations(bench, [n_sel], N, args.eval_seeds, obs, truth, t_split, eval_path, "selected",
                    finished_evaluations(out_dir))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=sorted(BENCHMARKS))
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS, help="particle budgets N")
    p.add_argument("--search-steps", type=int, default=None, help="length of the search period (default: half)")
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--seed", type=int, nargs="+", default=[SEARCH_SEED],
                   help="seeds of the searches (one independent search per seed)")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--grid", type=int, nargs="+", default=None,
                   help="values of n evaluated exhaustively (default: merge sizes of the main comparison)")
    p.add_argument("--no-grid", action="store_true", help="skip the exhaustive evaluation")
    p.add_argument("--dim", type=int, default=None,
                   help="l96 only: state dimension d (data/lorenz96_d<d>, see generate_data.py --l96-dims); "
                        "output to results/<benchmark>/select_n_dims/")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="smoke test (short series, N = 512, 2 rounds)")
    args = p.parse_args()

    bench = BENCHMARKS[args.bench]
    if args.dim is not None:
        if args.bench != "l96":
            raise SystemExit("--dim is for l96 only")
        bench = lorenz96_with_dim(args.dim)
    if args.quick:
        args.max_steps, args.budgets, args.rounds, args.eval_seeds, args.grid = 200, [512], 2, [42], [3, 10]
    obs, truth = bench.load_data(args.max_steps)
    T = len(obs)
    t_split = args.search_steps or T // 2
    M, n_bounds = SEARCH_SETTINGS[args.bench]["M"], SEARCH_SETTINGS[args.bench]["n_bounds"]
    sub = "select_n_dims" if args.dim is not None else "select_n"
    out_dir = os.path.join(ROOT, "results", bench.key, sub + ("_quick" if args.quick else ""))
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    eval_path = os.path.join(out_dir, f"eval_{ts}.csv")
    print(f"[{bench.name}, d={bench.dim}] T={T} (search [0,{t_split}), holdout [{t_split},{T}))  "
          f"numba threads={numba.get_num_threads()}", flush=True)

    for N in args.budgets:
        if not args.no_grid:
            grid = [n for n in (args.grid or bench.merge_sizes) if n <= N]
            print(f"\n=== grid: N={N}, n={grid} ===", flush=True)
            run_evaluations(bench, grid, N, args.eval_seeds, obs, truth, t_split, eval_path, "grid",
                            finished_evaluations(out_dir))
        for seed in args.seed:
            search(bench, obs, truth, N, M, n_bounds, t_split, seed, args, out_dir, eval_path, ts)
    print(f"[+] saved {out_dir}")


if __name__ == "__main__":
    main()
