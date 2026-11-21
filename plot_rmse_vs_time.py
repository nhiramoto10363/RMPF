"""
Main figure: RMSE vs computation time per assimilation step (Reviewer 1, major comment 5).

For every method and N the best parameter (seed-mean RMSE) is selected and the points are joined in
order of N; all other (parameter, N) configurations are shown as faint dots. Error bars are the
standard deviation over seeds. Configurations in which a seed became non-finite are not shown. All methods of a benchmark must come from one job on one node (job.sh),
otherwise the time axis is not comparable.

Usage:
    python plot_rmse_vs_time.py                               # figures/rmse_vs_time.{png,pdf}
    python plot_rmse_vs_time.py --time-metric median_filter_time
    python plot_rmse_vs_time.py --paper --exclude ETKF EnKF --output paper_rmse_vs_time   # figure of the paper
    python plot_rmse_vs_time.py --benchmarks l96abs --output l96abs_rmse_vs_time          # |x| observed
Output: figures/<output>.png/.pdf and figures/<output>_best.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from rmpf.benchmarks import BENCHMARKS, ROOT
from rmpf.results import METHOD_STYLES, aggregate, best_per_count, collect_legend, read_runs


def plot_panel(ax, agg, title):
    fronts = []
    for z, (name, label, color, marker) in enumerate(METHOD_STYLES):
        sub = agg[agg["method"] == name]
        if sub.empty:
            continue
        ok = sub[np.isfinite(sub["rmse_mean"])]
        if name != "PF":
            ax.scatter(ok["time"], ok["rmse_mean"], s=12, color=color, alpha=0.22, linewidths=0, zorder=2)
        line = best_per_count(sub)
        if line.empty:
            continue
        ax.errorbar(line["time"], line["rmse_mean"], yerr=line["rmse_std"].fillna(0), marker=marker, ms=4.5,
                    lw=2.0 if name == "RMPF" else 1.5, capsize=2, color=color, label=label, zorder=3 + z)
        fronts.append(line.assign(method=name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Computation time per assimilation step [s]")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.30, lw=0.6)
    ax.grid(True, which="minor", alpha=0.12, lw=0.4)
    return pd.concat(fronts, ignore_index=True) if fronts else pd.DataFrame()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--time-metric", default="median_step_time",
                   choices=["median_step_time", "median_filter_time", "total_time"])
    p.add_argument("--output", default="rmse_vs_time")
    p.add_argument("--exclude", nargs="+", default=[], help="methods not drawn, e.g. ETKF EnKF")
    p.add_argument("--paper", action="store_true", help="no title (run information goes to the caption)")
    p.add_argument("--benchmarks", nargs="+", default=["l63", "l96"], choices=sorted(BENCHMARKS),
                   help="panels (default: the two benchmarks of the main comparison; l96abs: |x| observed)")
    args = p.parse_args()
    if args.paper:   # figure is printed at text width: larger fonts
        plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 14,
                             "xtick.labelsize": 13, "ytick.labelsize": 13})

    panels = [(BENCHMARKS[k], read_runs(BENCHMARKS[k].key)) for k in args.benchmarks]
    panels = [(b, df) for b, df in panels if df is not None]
    if not panels:
        raise SystemExit("no results/<benchmark>/experiments/runs_*.csv; run run_experiments.py first")

    fig, axes = plt.subplots(1, len(panels), figsize=(6.75 * len(panels), 5.6), squeeze=False)
    fronts, notes = [], []
    for ax, (b, df) in zip(axes[0], panels):
        agg = aggregate(df, args.time_metric)
        front = plot_panel(ax, agg[~agg["method"].isin(args.exclude)], f"{b.name} (d={b.dim})")
        ax.axhline(float(agg["clim"].iloc[0]), color="#999999", lw=0.8, ls=":")
        if not front.empty:
            fronts.append(front.assign(benchmark=b.name))
        hosts = ",".join(sorted(df["host"].astype(str).unique()))
        threads = ",".join(str(t) for t in sorted(df["numba_threads"].unique()))
        notes.append(f"{b.name}: {len(df)} runs, seeds {','.join(map(str, sorted(df['seed'].unique())))}, "
                     f"{hosts}, {threads} threads")

    extra = [(Line2D([], [], marker="o", ls="none", ms=5, alpha=0.3, color="#888888"), "other parameters (mean over seeds)")]
    extra.append((Line2D([], [], color="#999999", lw=0.8, ls=":"), "climatological standard deviation"))
    handles, labels = collect_legend(fig, axes, extra)
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if not args.paper:
        fig.suptitle("RMSE vs computation time (mean ± s.d. over seeds)\n" + ";   ".join(notes), y=0.995, fontsize=10)
    bottom = (0.2 if len(panels) > 1 else 0.13) if args.paper else 0.1   # room for the legend below
    fig.tight_layout(rect=(0, bottom, 1, 0.95))

    out = os.path.join(ROOT, "figures", args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    if fronts:
        cols = ["benchmark", "method", "num_particles", "param", "time", "rmse_mean", "rmse_std", "rmse_median",
                "n_seeds", "diverged_rate"]
        pd.concat(fronts, ignore_index=True)[cols].to_csv(f"{out}_best.csv", index=False)
    print(f"Saved {out}.png / .pdf / _best.csv")


if __name__ == "__main__":
    main()
