"""
RMSE of every run against the tuning parameter, for small N (Reviewer 1, minor comment 16).

One panel per particle filter (bootstrap PF against N; MPF against the coefficient set; RMPF against n;
RPF against c; LW against h). Every point is one run (one seed); the colour gives N. Runs whose RMSE is
not finite are drawn as crosses at the top of the panel. The dotted line is the climatological standard
deviation. No threshold for "failure" is used: the figure shows how the
runs split into those that track the truth and those that lose it, and how this depends on the parameter.

The table figures/<output>_median.csv (and the LaTeX body tables/tableC2_median_body.tex, Table C2) lists, for every method and N, the best parameter (smallest mean
RMSE over seeds, as in Table 3) with the mean and
and the median RMSE over seeds.

Usage:
    python plot_rmse_vs_param.py                       # l63, N = 32 ... 1024
    python plot_rmse_vs_param.py --paper --output l63_rmse_vs_param
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rmpf.benchmarks import BENCHMARKS, ROOT
from rmpf.results import aggregate, best_per_count, read_runs

PANELS = [("PF", "Bootstrap PF", "N", True),
          ("MPF", r"MPF (fixed $\alpha$)", "coefficient set", False),
          ("RMPF", "RMPF", "merging size n", True),
          ("RPF", "RPF", "bandwidth scale c", True),
          ("LW", "Liu–West", "h", False)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="l63", choices=sorted(BENCHMARKS))
    p.add_argument("--counts", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--output", default="l63_rmse_vs_param")
    p.add_argument("--paper", action="store_true", help="no title (run information goes to the caption)")
    args = p.parse_args()
    if args.paper:
        plt.rcParams.update({"font.size": 17, "axes.titlesize": 18, "axes.labelsize": 17, "legend.fontsize": 17,
                             "xtick.labelsize": 14, "ytick.labelsize": 14})

    b = BENCHMARKS[args.benchmark]
    df = read_runs(b.key)
    if df is None:
        raise SystemExit("no results/%s/experiments/runs_*.csv" % b.key)
    df = df[df.num_particles.isin(args.counts)]
    clim = float(df.climatological_std.iloc[0])
    finite = np.isfinite(df.rmse)
    top = 10 ** (np.log10(max(df.rmse[finite].max(), clim)) + 0.12)
    bottom = df.rmse[finite].min() / 1.25
    colors = dict(zip(args.counts, plt.cm.viridis(np.linspace(0.0, 0.9, len(args.counts)))))
    offsets = dict(zip(args.counts, np.linspace(-1, 1, len(args.counts))))

    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.4 * len(PANELS), 4.2 if args.paper else 3.9), sharey=True)
    for ax, (name, title, xlabel, logx) in zip(axes, PANELS):
        sub = df[df.method == name]
        if name == "PF":
            x0 = sub.num_particles.astype(float)
            spread = 0.0
        else:
            x0 = sub.param.astype(float)
            values = np.sort(sub.param.unique())
            gaps = np.diff(np.log(values)) if logx else np.diff(values)
            spread = 0.18 * (gaps.min() if len(gaps) else 1.0)
        shift = sub.num_particles.map(offsets) * spread
        x = x0 * np.exp(shift) if logx else x0 + shift
        ok = np.isfinite(sub.rmse)
        ax.scatter(x[ok], sub.rmse[ok], s=22 if args.paper else 14, c=[colors[N] for N in sub.num_particles[ok]], alpha=0.85,
                   edgecolors="none")
        if (~ok).any():
            ax.scatter(x[~ok], [top] * int((~ok).sum()), marker="x", s=18,
                       c=[colors[N] for N in sub.num_particles[~ok]])
        ax.axhline(clim, color="#777777", lw=0.8, ls=":")
        if logx:
            ax.set_xscale("log", base=2 if name == "PF" else 10)
        if name == "MPF":
            ax.set_xticks([1, 2, 3])
            ax.set_xlim(0.5, 3.5)
        if name == "RMPF":
            ax.set_xticks([3, 5, 10, 20, 40, 100])
            ax.set_xticklabels(["3", "5", "10", "20", "40", "100"])
        ax.set_yscale("log")
        ax.set_ylim(bottom, top * 1.25)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", alpha=0.30, lw=0.6)
    axes[0].set_ylabel("RMSE")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=colors[N], label=f"N={N}") for N in args.counts]
    fig.legend(handles=handles, loc="lower center", ncol=len(args.counts), frameon=False, bbox_to_anchor=(0.5, -0.01))
    if not args.paper:
        fig.suptitle(f"{b.name}: RMSE of every run (seed) against the parameter; dotted: climatology; "
                     "x: non-finite", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96 if not args.paper else 1))
    out = os.path.join(ROOT, "figures", args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")

    rows = []
    agg = aggregate(df)
    for name, _, _, _ in PANELS:
        best = best_per_count(agg[agg.method == name])
        for _, r in best.iterrows():
            runs = df[(df.method == name) & (df.num_particles == r.num_particles) & (df.param == r.param)]
            rows.append(dict(method=name, num_particles=int(r.num_particles), param=r.param,
                             rmse_mean=r.rmse_mean, rmse_median=float(np.median(runs.rmse)),
                             n_seeds=len(runs)))
    med = pd.DataFrame(rows)
    med.to_csv(out + "_median.csv", index=False)
    # LaTeX body: median over seeds at the best parameter; "n.f." if every parameter became non-finite in a seed
    lines = []
    for N in args.counts:
        cells = []
        for name, _, _, _ in PANELS:
            r = med[(med.method == name) & (med.num_particles == N)]
            cells.append("n.f." if r.empty else "%.3f" % r.rmse_median.iloc[0])
        lines.append("%d & %s \\\\" % (N, " & ".join(cells)))
    os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
    open(os.path.join(ROOT, "tables", "tableC2_median_body.tex"), "w").write("\n".join(lines) + "\n")
    print("written", out + ".{png,pdf}", out + "_median.csv", "tables/tableC2_median_body.tex")


if __name__ == "__main__":
    main()
