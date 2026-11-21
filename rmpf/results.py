"""
Loading and aggregating the run CSVs written by run_experiments.py (shared by the figure scripts).

Aggregation over seeds for every (method, param, N):
  rmse         mean over seeds of the per-run time-mean RMSE; +inf if any seed became non-finite
  rmse_std     standard deviation over seeds
  rmse_median  median over seeds
  diverged_rate  fraction of seeds with rmse > climatological std (or non-finite); diagnostic only, not
                 used by the tables and figures of the paper
The "best" configuration of a method at a given N is the parameter (n, h, c, lambda, coefficient
set) with the smallest finite mean RMSE over the seeds (a configuration in which any seed became
non-finite has mean +inf and is never selected).
"""
import glob
import os

import numpy as np
import pandas as pd

from .benchmarks import ROOT

# drawing order (later on top), label, color, marker
METHOD_STYLES = [
    ("PF", "Bootstrap PF", "#444444", "o"),
    ("MPF", "MPF  (best fixed α)", "#F58518", "v"),
    ("RPF", "RPF  (best over c)", "#B279A2", "s"),
    ("LW", "Liu-West  (best over h)", "#4C78A8", "^"),
    ("ETKF", "ETKF  (best over λ)", "#54A24B", "D"),
    ("EnKF", "EnKF, perturbed obs.  (best over λ)", "#EECA3B", "P"),
    ("RMPF", "RMPF  (best over n)", "#E45756", "o"),
]


def read_runs(bench_key, subdir="experiments"):
    files = sorted(glob.glob(os.path.join(ROOT, "results", bench_key, subdir, "runs_*.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["diverged"] = df["diverged"].astype(str).str.strip().eq("True")
    return df.drop_duplicates(["method", "param", "num_particles", "seed"], keep="last")


def aggregate(df, time_col="median_step_time"):
    df = df.assign(rmse=df["rmse"].where(np.isfinite(df["rmse"]), np.inf))
    return df.groupby(["method", "param", "num_particles"], as_index=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        rmse_median=("rmse", "median"),
        time=(time_col, "mean"),
        n_seeds=("rmse", "size"),
        diverged_rate=("diverged", "mean"),
        clim=("climatological_std", "first"),
    )


def usable(agg):
    return agg[np.isfinite(agg["rmse_mean"])]


def best_per_count(agg):
    """Best parameter of one method at every N (rows sorted by N)."""
    ok = usable(agg)
    if ok.empty:
        return ok
    return ok.loc[ok.groupby("num_particles")["rmse_mean"].idxmin()].sort_values("num_particles")


def collect_legend(fig, axes, extra=()):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    for h, l in extra:
        handles.append(h)
        labels.append(l)
    return handles, labels
