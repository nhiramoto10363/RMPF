"""
Truth and observations of both benchmarks (data/<benchmark>/).

The truth is a noise-free forward-Euler trajectory sampled every obs_interval model steps; the
observations add N(0, obs_noise_std^2) noise to every component (seed 42). Lorenz-96 filters use
only the even components. Lorenz-96 with |x| observed (data/lorenz96_abs) takes the absolute value of
the Lorenz-96 observations: y = |x + e| with the same truth and noise. The CSV files shipped in data/ were produced by this script.

Usage:
    python generate_data.py                 # write data/lorenz63 and data/lorenz96
    python generate_data.py --check         # regenerate in memory and compare with data/ (bitwise)
    python generate_data.py --l96-dims 10 20 30   # also Lorenz-96 with d = 10, 20, 30 (for select_n.py --dim);
                                                   # data/lorenz96_d80 (benchmark l96d80) is always written
"""
import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from rmpf.benchmarks import BENCHMARKS, lorenz96_with_dim
from rmpf.models import lorenz63_trajectory, lorenz96_trajectory

SETTINGS = {
    # model steps, warm-up model steps before the first saved state
    "l63": dict(steps=50000, warmup_steps=1000, sigma=10.0, rho=28.0, beta=8 / 3),
    "l96": dict(steps=20000, warmup_steps=2000, F=8.0),
}


def generate(key, bench=None):
    b = bench or BENCHMARKS[key]
    if b.obs_operator == "abs":
        truth, obs, meta = generate("l96", replace(b, obs_operator="identity"))
        meta = {k: v for k, v in meta.items() if k != "num_observations"}
        return truth, obs.abs(), dict(meta, obs_operator="abs", num_observations=len(truth))
    s = SETTINGS[key]
    if key == "l63":
        X = lorenz63_trajectory(np.array([1.0, 1.0, 1.0]), s["steps"], b.dt, s["sigma"], s["rho"], s["beta"],
                                s["warmup_steps"])
        columns = ["x", "y", "z"]
    else:
        x0 = np.full(b.dim, 8.0)
        x0[min(19, b.dim - 1)] = 8.008     # x_20 is perturbed; the last variable if d < 20
        X = lorenz96_trajectory(x0, b.dim, s["F"], b.dt, s["steps"], s["warmup_steps"])
        columns = [f"x{i}" for i in range(b.dim)]
    truth = X[::b.obs_interval]
    np.random.seed(42)
    obs = truth + np.random.normal(0, b.obs_noise_std, truth.shape)
    meta = dict(s, dim=b.dim, dt=b.dt, obs_interval=b.obs_interval, obs_noise_std=b.obs_noise_std,
                num_observations=len(truth))
    return pd.DataFrame(truth, columns=columns), pd.DataFrame(obs, columns=columns), meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--l96-dims", type=int, nargs="+", default=[],
                   help="also Lorenz-96 with these state dimensions (data/lorenz96_d<d>; for select_n.py --dim)")
    args = p.parse_args()
    # every Lorenz-96 variant (l96abs, l96d80, --l96-dims) uses the Lorenz-96 settings of SETTINGS
    benches = [("l63" if key == "l63" else "l96", b) for key, b in BENCHMARKS.items()]
    benches += [("l96", lorenz96_with_dim(d)) for d in args.l96_dims if d != BENCHMARKS["l96"].dim]
    seen = set()
    benches = [(k, b) for k, b in benches if not (b.data_dir in seen or seen.add(b.data_dir))]
    for key, b in benches:
        truth, obs, meta = generate(key, b)
        if args.check:
            for name, df in (("true_states", truth), ("observations", obs)):
                ref = pd.read_csv(os.path.join(b.data_dir, f"{name}.csv"), float_precision="round_trip")
                same = ref.shape == df.shape and np.array_equal(ref.to_numpy(), df.to_numpy())
                print(f"{b.name} {name}: {'identical' if same else 'DIFFERENT'} {df.shape}")
            continue
        os.makedirs(b.data_dir, exist_ok=True)
        truth.to_csv(os.path.join(b.data_dir, "true_states.csv"), index=False)
        obs.to_csv(os.path.join(b.data_dir, "observations.csv"), index=False)
        pd.Series(meta).to_csv(os.path.join(b.data_dir, "metadata.csv"))
        print(f"wrote {b.data_dir} ({len(truth)} observation times)")


if __name__ == "__main__":
    main()
