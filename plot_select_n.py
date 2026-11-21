"""
Figure of the merge-size selection (output of select_n.py).

Columns = (benchmark, N). Top: logZ over the search period as a function of n (grid evaluation;
median over seeds with the min-max band), the grid-best n (largest logZ) and the n selected by PRS
(dashed line and star; the star is the selected n evaluated with the same seeds). Bottom: RMSE over
the held-out period, which the search never used. n at which some seed diverged are drawn open.
Medians are used because a single diverged seed dominates a mean over three seeds. The vertical
range follows the non-diverged n; points outside it are drawn as triangles at the edge.

With --l96-dims (columns = state dimension d of Lorenz-96; several PRS searches per d, select_n.py
--seed 1 ... 10) the figure shows means over the evaluation seeds instead:
  top:    logZ over the search period relative to the largest grid mean (mean +- standard error);
  bottom: RMSE over the held-out period (mean +- standard error). The grey band marks the near-optimal
          range, mean RMSE within one standard error of the grid minimum. Each PRS search is a red
          marker at its selected n (mean RMSE of the same seeds); repeated selections are stacked.
The selections are read from prs_rounds_*.csv (incumbent of the last round), their RMSE from the
evaluation runs (grid runs are reused when the selected n is on the grid). The table <output>.csv lists,
per d, the grid-best n, the RMSE without merging (n = 2), the selections and their regret, i.e. the
increase of the mean RMSE over the grid minimum.

Usage:
    python plot_select_n.py                         # figures/select_n.{png,pdf,csv}
    python plot_select_n.py --budgets l63:4096 l96:4096
    python plot_select_n.py --l96-dims 10 20 30 40 --output select_n_dims   # columns = state dimension of Lorenz-96
    python plot_select_n.py --l96-dims 10 20 30 40 --paper --output paper_select_n_dims
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from rmpf.benchmarks import BENCHMARKS, ROOT, lorenz96_with_dim

TITLE_FS, PANEL_FS, LABEL_FS, TICK_FS, LEG_FS = 20, 16, 15, 12, 14


def load(bench, sub="select_n"):
    files = glob.glob(os.path.join(ROOT, "results", bench.key, sub, "eval_*.csv"))
    if not files:
        return None
    d = pd.concat(pd.read_csv(f) for f in files)
    d["diverged"] = d["diverged"].astype(str).str.strip().eq("True")
    return d.drop_duplicates(["role", "n", "num_particles", "seed"], keep="last")


def summarize(d, col):
    g = d.groupby("n")
    return pd.DataFrame({"med": g[col].median(), "lo": g[col].min(), "hi": g[col].max(),
                         "div": g["diverged"].any(), "seeds": g["seed"].nunique()}).sort_index()


def _limits(s, sel_med):
    """Vertical range from the bulk of the median curve (ignoring far-off diverged n) and the selected value."""
    m = s.loc[~s["div"], "med"]
    center, spread = m.median(), (m - m.median()).abs().median()
    keep = m[(m - center).abs() <= 20 * spread + 1e-12 * abs(center)] if spread > 0 else m
    vals = np.append(keep.to_numpy(), sel_med)
    lo, hi = vals.min(), vals.max()
    pad = 0.25 * (hi - lo if hi > lo else abs(hi) * 0.01 + 1e-9)
    return lo - pad, hi + pad


def panel(ax, s, sel, best_n, ylabel, maximize, xlim=None, xscale="log2"):
    lo, hi = _limits(s, sel["med"])
    x = s.index.to_numpy()
    ax.fill_between(x, np.clip(s["lo"], lo, hi), np.clip(s["hi"], lo, hi), color="C0", alpha=0.15, zorder=1)
    ax.plot(x, np.clip(s["med"], lo, hi), "-", color="C0", lw=1.8, zorder=2)
    for n, r in s.iterrows():
        y, out = r["med"], not (lo <= r["med"] <= hi)
        if out:   # off-scale: triangle at the edge
            below = y < lo
            ax.scatter([n], [lo if below else hi], marker="v" if below else "^", s=70,
                       facecolor="white", edgecolor="C0", zorder=4, clip_on=False)
        else:
            ax.scatter([n], [y], marker="o", s=40, facecolor="white" if r["div"] else "C0",
                       edgecolor="C0", lw=1.5, zorder=3)
    b = s.loc[best_n]
    if lo <= b["med"] <= hi:
        ax.scatter([best_n], [b["med"]], marker="P", s=220, c="cyan", edgecolor="k", zorder=5)
    ax.axvline(sel["n"], color="red", ls="--", lw=2, zorder=4)
    if lo <= sel["med"] <= hi:
        ax.errorbar([sel["n"]], [sel["med"]], yerr=[[sel["med"] - max(sel["lo"], lo)], [min(sel["hi"], hi) - sel["med"]]],
                    fmt="none", ecolor="red", elinewidth=1.2, capsize=3, zorder=5)
        ax.scatter([sel["n"]], [sel["med"]], marker="*", s=320, c="red", edgecolor="k", zorder=6)
    if xscale == "log2":
        ax.set_xscale("log", base=2)
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
        ax.set_xlim(*(xlim or (x.min() / 1.25, x.max() * 1.25)))
    else:
        ax.set_xlim(*(xlim or (0, x.max() * 1.05)))
    ax.set_ylim(lo, hi)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def load_dims(bench, N):
    """Evaluation runs (one row per (n, seed)) and the n selected by every PRS search (seed -> n)."""
    folder = os.path.join(ROOT, "results", bench.key, "select_n_dims")
    files = glob.glob(os.path.join(folder, "eval_*.csv"))
    if not files:
        return None, None
    d = pd.concat(pd.read_csv(f, float_precision="round_trip") for f in files)
    d = d[d.num_particles == N].drop_duplicates(["n", "seed"], keep="last")
    d["diverged"] = d["diverged"].astype(str).str.strip().eq("True")
    selected = {}
    for f in sorted(glob.glob(os.path.join(folder, f"prs_rounds_N{N}_*.csv"))):
        r = pd.read_csv(f)
        selected[int(r.seed.iloc[0])] = int(r.inc_n.iloc[-1])
    return d, selected


def mean_se(d, col):
    g = d.groupby("n")[col]
    return pd.DataFrame({"mean": g.mean(), "se": g.std(ddof=1) / np.sqrt(g.count()), "seeds": g.count()}).sort_index()


def dims_panel(ax, s, grid, y_sel, ylabel, band=None, robust=15.0, headroom=0.2):
    """Grid mean +- s.e. (line, band), optional near-optimal band, stacked markers of the PRS selections."""
    g = s.loc[grid]
    lo_data = g["mean"] - g["se"]
    # grid means far from the rest (n = 2, 3 for large d) are drawn as triangles at the edge
    mad = (g["mean"] - g["mean"].median()).abs().median()
    vis = g[(g["mean"] - g["mean"].median()).abs() <= robust * mad] if mad > 0 else g
    if band is not None:   # the minimum and the near-optimal band are always shown
        vis = pd.concat([vis, g.loc[[g["mean"].idxmin()]]])
    lo = min((vis["mean"] - vis["se"]).min(), min(y for _, y in y_sel))
    hi = max((vis["mean"] + vis["se"]).max(), max(y for _, y in y_sel))
    lo, hi = lo - 0.1 * (hi - lo), hi + headroom * (hi - lo)
    x = g.index.to_numpy()
    ax.fill_between(x, np.clip(lo_data, lo, hi), np.clip(g["mean"] + g["se"], lo, hi), color="C0", alpha=0.25, lw=0, zorder=2)
    ax.plot(x, np.clip(g["mean"], lo, hi), "-o", color="C0", lw=1.6, ms=4.5, zorder=3)
    labelled = False
    for n, m in g["mean"].items():   # off-scale grid points: triangle at the edge, value of the first one
        if m < lo or m > hi:
            ax.scatter([n], [lo if m < lo else hi], marker="v" if m < lo else "^", s=80, facecolor="white",
                       edgecolor="C0", lw=1.5, zorder=4, clip_on=False)
            if not labelled:
                ax.annotate(f"$n$={n}: {m:.4g}", (n, lo if m < lo else hi), xytext=(8, 5 if m < lo else -14),
                            textcoords="offset points", fontsize=TICK_FS - 1, color="C0")
                labelled = True
    if band is not None:
        ax.axhspan(lo, band, color="0.6", alpha=0.18, lw=0, zorder=1)
    count = {}
    for n, y in sorted(y_sel):   # repeated selections of the same n are stacked
        k = count.get(n, 0)
        count[n] = k + 1
        ax.scatter([n], [y], marker="*", s=230, c="red", edgecolor="k", lw=0.8, zorder=6)
        if k:
            ax.annotate(f"\u00d7{k + 1}", (n, y), xytext=(7, 4), textcoords="offset points", fontsize=TICK_FS - 1, color="red")
    ax.set_xlim(0, 130)
    ax.set_xticks([2, 20, 40, 60, 80, 100, 120])
    ax.set_ylim(lo, hi)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def plot_dims(args):
    cols = []
    for dim in args.l96_dims:
        bench = lorenz96_with_dim(dim)
        d, selected = load_dims(bench, args.budget)
        if d is None or not selected:
            continue
        cols.append((bench, d, selected))
    if not cols:
        raise SystemExit("no select_n_dims results with PRS searches; run select_n.py --dim first")

    fig, axes = plt.subplots(2, len(cols), figsize=(6.0 * len(cols), 8.8), sharex="col", squeeze=False)
    rows = []
    for j, (bench, d, selected) in enumerate(cols):
        grid = sorted(d.loc[d.role == "grid", "n"].unique())
        z, r = mean_se(d, "logZ_search"), mean_se(d, "rmse_holdout")
        missing = sorted({n for n in selected.values() if n not in r.index or r.loc[n, "seeds"] < r["seeds"].max()})
        if missing:
            print(f"warning: d={bench.dim}: selected n without all evaluation seeds: {missing}")
        z_ref = z.loc[grid, "mean"].max()
        z = z.assign(mean=z["mean"] - z_ref)
        best = int(r.loc[grid, "mean"].idxmin())
        r_best, se_best = r.loc[best, "mean"], r.loc[best, "se"]
        band = r_best + se_best
        sel_n = [selected[s] for s in sorted(selected)]
        regret = [100 * (r.loc[n, "mean"] / r_best - 1) for n in sel_n]
        no_merge = 100 * (r.loc[2, "mean"] / r_best - 1) if 2 in r.index else np.nan
        # logZ panel: grid points far below the rest (n = 2, 3 for large d) are drawn as triangles
        zg = z.loc[grid, "mean"]
        dims_panel(axes[0, j], z, grid, [(n, z.loc[n, "mean"]) for n in sel_n],
                   (r"$\Delta\log Z$ (search)" if args.paper else "logZ - max grid mean, search period") if j == 0 else "",
                   )
        dims_panel(axes[1, j], r, grid, [(n, r.loc[n, "mean"]) for n in sel_n],
                   ("RMSE (held-out)" if args.paper else "RMSE, held-out period") if j == 0 else "", band=band,
                   headroom=0.2 if args.paper else 0.55)
        axes[0, j].set_title(f"{bench.name}, d={bench.dim}", fontsize=PANEL_FS)
        axes[1, j].set_xlabel("merge size $n$", fontsize=LABEL_FS)
        if not args.paper:   # the numbers are given in the text of the paper
            axes[1, j].text(0.97, 0.97, f"best $n$={best}: {r_best:.3f}\nno merging: +{no_merge:.1f}%\n"
                            f"PRS: +{np.median(regret):.1f}% (max +{max(regret):.1f}%)", transform=axes[1, j].transAxes,
                            fontsize=TICK_FS, va="top", ha="right",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9), zorder=10)
        rows.append(dict(dim=bench.dim, N=args.budget, seeds=int(r["seeds"].max()), searches=len(sel_n),
                         best_n=best, best_rmse=r_best, best_rmse_se=se_best, rmse_n2=r.loc[2, "mean"],
                         regret_n2_pct=no_merge,
                         near_optimal_n=" ".join(str(n) for n in grid if r.loc[n, "mean"] <= band),
                         selected_n=" ".join(map(str, sel_n)),
                         regret_median_pct=float(np.median(regret)), regret_max_pct=float(max(regret)),
                         selected_in_band=int(sum(r.loc[n, "mean"] <= band for n in sel_n)),
                         best_logZ_n=int(zg.idxmax())))

    n_seeds, n_search = rows[0]["seeds"], rows[0]["searches"]
    if not args.paper:
        fig.suptitle(f"Merge-size selection by PRS, Lorenz-96, N={args.budget}", y=0.995, fontsize=TITLE_FS)
    fig.tight_layout(rect=(0, 0.1 if args.paper else 0.075, 1, 0.97))
    fig.align_ylabels(axes[:, 0])
    handles = [
        Line2D([0], [0], marker="o", color="C0", lw=1.6, ms=5, label=f"grid (mean of {n_seeds} seeds)"),
        Patch(facecolor="C0", alpha=0.25, label="\u00b11 s.e."),
        Patch(facecolor="0.6", alpha=0.3, label="within 1 s.e. of the minimum"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markeredgecolor="k", markersize=16,
               label=f"PRS selections ({n_search} searches)"),
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), fontsize=LEG_FS, framealpha=0.9, handletextpad=0.5, columnspacing=1.6)
    out = os.path.join(ROOT, "figures", args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}.csv", index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}.png / .pdf / .csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="select_n")
    p.add_argument("--paper", action="store_true", help="no title (run information goes to the caption)")
    p.add_argument("--l96-dims", type=int, nargs="+", default=None,
                   help="columns = Lorenz-96 with these state dimensions (results of select_n.py --dim)")
    p.add_argument("--budget", type=int, default=32768, help="N of the --l96-dims runs")
    p.add_argument("--xscale", choices=["log2", "linear"], default=None,
                   help="axis of n (default: linear with --l96-dims, else log2)")
    p.add_argument("--budgets", nargs="+", default=None,
                   help="(benchmark:N) pairs to draw, e.g. l63:1024 l96:4096 (default: all)")
    args = p.parse_args()
    if args.paper:   # figure is printed at text width: larger fonts
        globals().update(TITLE_FS=26, PANEL_FS=24, LABEL_FS=23, TICK_FS=19, LEG_FS=21)
        plt.rcParams.update({"xtick.labelsize": 19, "ytick.labelsize": 19})

    if args.l96_dims:
        return plot_dims(args)

    want = None if args.budgets is None else {(b.split(":")[0], int(b.split(":")[1])) for b in args.budgets}
    xscale = args.xscale or ("linear" if args.l96_dims else "log2")
    cols = []
    for d_ in args.l96_dims or []:
        bench = lorenz96_with_dim(d_)
        d = load(bench, "select_n_dims")
        if d is None:
            continue
        d = d[d.num_particles == args.budget]
        sel = d[d.role == "selected"]
        if len(sel):
            cols.append((f"{bench.name}, d={bench.dim}", args.budget, d[d.role == "grid"], sel))
    for key, bench in ({} if args.l96_dims else BENCHMARKS).items():
        d = load(bench)
        if d is None:
            continue
        for N in sorted(d.loc[d.role == "grid", "num_particles"].unique()):
            if want is not None and (key, int(N)) not in want:
                continue
            sel = d[(d.role == "selected") & (d.num_particles == N)]
            if len(sel):
                cols.append((f"{bench.name}, N={int(N)}", int(N), d[(d.role == "grid") & (d.num_particles == N)], sel))
    if not cols:
        raise SystemExit("no eval_*.csv with both grid and selected rows; run select_n.py first")

    fig, axes = plt.subplots(2, len(cols), figsize=(6.0 * len(cols), 8.6), sharex="col", squeeze=False)
    rows, any_div = [], False
    for j, (name, N, grid, sel) in enumerate(cols):
        n_sel = int(sel["n"].iloc[0])
        sz, sr = summarize(grid, "logZ_search"), summarize(grid, "rmse_holdout")
        any_div |= bool(sz["div"].any())
        selz, selr = summarize(sel, "logZ_search").iloc[0], summarize(sel, "rmse_holdout").iloc[0]
        best_z = int(sz.loc[~sz["div"], "med"].idxmax()) if (~sz["div"]).any() else int(sz["med"].idxmax())
        best_r = int(sr["med"].idxmin())
        panel(axes[0, j], sz, dict(n=n_sel, med=selz["med"], lo=selz["lo"], hi=selz["hi"]), best_z,
              ("log Z (search)" if args.paper else "logZ, search period") if j == 0 else "", True, xscale=xscale)
        panel(axes[1, j], sr, dict(n=n_sel, med=selr["med"], lo=selr["lo"], hi=selr["hi"]), best_r,
              ("RMSE (held-out)" if args.paper else "RMSE, held-out period") if j == 0 else "", False, xscale=xscale)
        axes[0, j].set_title(name, fontsize=PANEL_FS)
        axes[1, j].set_xlabel("merge size $n$" + (" (log scale)" if xscale == "log2" else ""), fontsize=LABEL_FS)
        axes[0, j].text(0.03, 0.03, f"grid-best $n$={best_z}\nPRS $n$={n_sel}", transform=axes[0, j].transAxes,
                        fontsize=TICK_FS, va="bottom", ha="left",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))
        axes[1, j].text(0.97, 0.97, f"min-RMSE $n$={best_r}: {sr.loc[best_r, 'med']:.3f}\n"
                        f"PRS $n$={n_sel}: {selr['med']:.3f}", transform=axes[1, j].transAxes,
                        fontsize=TICK_FS, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85))
        rows.append(dict(benchmark=name, N=N, selected_n=n_sel, grid_best_logZ_n=best_z, min_rmse_n=best_r,
                         selected_logZ=selz["med"], selected_rmse_holdout=selr["med"],
                         min_rmse_holdout=sr.loc[best_r, "med"], seeds=int(sr["seeds"].max())))

    n_seeds = int(max(r["seeds"] for r in rows))
    if not args.paper:
        fig.suptitle("Merge-size selection by PRS: grid evaluation vs selected n", y=0.995, fontsize=TITLE_FS)
    fig.tight_layout(rect=(0, 0.1 if args.paper else 0.075, 1, 0.97))
    fig.align_ylabels(axes[:, 0])
    handles = [
        Line2D([0], [0], marker="o", color="C0", lw=1.8, ms=6, label=f"grid (median of {n_seeds} seeds)"),
        Patch(facecolor="C0", alpha=0.15, label="min–max over seeds"),
        Line2D([0], [0], marker="o", color="C0", mfc="white", lw=0, ms=7, label="some seed diverged"),
        Line2D([0], [0], marker="v", color="C0", mfc="white", lw=0, ms=8, label="off-scale"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="cyan", markeredgecolor="k",
               markersize=14, linestyle="none", label="grid-best"),
        Line2D([0], [0], marker="*", color="red", markerfacecolor="red", markeredgecolor="k",
               markersize=16, ls="--", lw=2, label="PRS selection"),
    ]
    if not any_div:
        handles = [h for h in handles if h.get_label() != "some seed diverged"]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(handles), fontsize=LEG_FS, framealpha=0.9, handletextpad=0.5, columnspacing=1.6)
    out = os.path.join(ROOT, "figures", args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=150)
    pd.DataFrame(rows).to_csv(f"{out}.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved {out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
