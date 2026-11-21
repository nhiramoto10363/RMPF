"""
LaTeX bodies of the tables of the paper, from the CSV files in results/ (no computation).

  tables/table3_body.tex            Table 3   best parameter per method and N, Lorenz-63 and Lorenz-96
  tables/table_l96_time_body.tex    Table 4   Lorenz-96 (d = 40): time per step and of the analysis step of the PFs
  tables/table_d80_body.tex         Table 5   Lorenz-96 with d = 80: RMSE, time per step and of the analysis step
  tables/table_duplicates_body.tex  Table B1  duplicated particles in the merging groups
  tables/tableC1_abs_body.tex       Table C1  as Table 3, Lorenz-96 with |x| observed
  tables/tableC3_smalln_body.tex    Table C3  as Table 3, small ensembles

Table C2 (medians) is written by plot_rmse_vs_param.py and Table C4 (outlying particles) by
theory/outlier_table.py. Table numbers are those of the revised manuscript.

Cell of Tables 3, C1, C3: mean RMSE over the seeds of the parameter with the smallest mean; below it the
standard deviation over the seeds and the parameter. n.f.: for every parameter, at least one seed became
non-finite; --: not run.

Usage:  python make_tables.py
"""
import glob
import os

import numpy as np
import pandas as pd

from rmpf.benchmarks import ROOT
from rmpf.results import aggregate, best_per_count, read_runs

OUT = os.path.join(ROOT, "tables")
METHODS = [("PF", "Bootstrap PF", None), ("MPF", r"MPF (fixed $\boldsymbol{\alpha}$)", "set {:g}"),
           ("RPF", "RPF", "$c$={:g}"), ("LW", "Liu--West", "$h$={:g}"), ("ETKF", "ETKF", r"$\lambda$={:g}"),
           ("EnKF", r"EnKF (pert.\ obs.)", r"$\lambda$={:g}"), ("RMPF", "RMPF", "$n$={:g}")]
HEADER = r"$N$ & " + " & ".join(label for _, label, _ in METHODS) + r" \\"


def cell(agg, method, N):
    s = agg[(agg.method == method) & (agg.num_particles == N)]
    if s.empty:
        return r"\rc{--}{}"
    ok = s[np.isfinite(s.rmse_mean)]
    if ok.empty:
        return r"\rc{n.f.}{}"
    b = ok.loc[ok.rmse_mean.idxmin()]
    fmt = dict((k, f) for k, _, f in METHODS)[method]
    sub = r"$\pm$" + f"{b.rmse_std:.3f}" + (", " + fmt.format(b.param) if fmt else "")
    return r"\rc{" + f"{b.rmse_mean:.3f}" + "}{" + sub + "}"


def best_table(name, blocks, subdir="experiments"):
    """blocks: [(benchmark key, title or None)]; consecutive blocks are separated by a rule."""
    out = []
    for i, (key, title) in enumerate(blocks):
        agg = aggregate(read_runs(key, subdir))
        if i:
            out.append(r"\middlehline")
        if title:
            out.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
        for N in sorted(agg.num_particles.unique()):
            out.append(f"{N} & " + " & ".join(cell(agg, m, N) for m, _, _ in METHODS) + r" \\")
    write(name, HEADER + "\n\\middlehline\n" + "\n".join(out) + "\n")


def d80_table():
    """Three blocks (RMSE, time per step, time of the analysis step) with the methods as columns, to fit one column of the page."""
    df = read_runs("lorenz96_d80")
    methods = ("PF", "RPF", "LW", "RMPF")
    rmse, time, analysis = [], [], []
    for N in sorted(df.num_particles.unique()):
        s = df[df.num_particles == N].set_index("method")
        lab = f"$2^{{{int(np.log2(N))}}}$ & "
        rmse.append(lab + " & ".join(f"{s.rmse[m]:.2f}" for m in methods) + r" \\")
        time.append(lab + " & ".join(f"{s.median_step_time[m]:.2f}" for m in methods) + r" \\")
        analysis.append(lab + " & ".join(f"{s.median_filter_time[m]:.2f}" for m in methods) + r" \\")
    write("table_d80_body.tex", "\n".join([r"\multicolumn{5}{l}{\textit{(a) RMSE}} \\"] + rmse + [r"\midrule",
          r"\multicolumn{5}{l}{\textit{(b) Time per assimilation step (s)}} \\"] + time + [r"\midrule",
          r"\multicolumn{5}{l}{\textit{(c) Time of the analysis step (s)}} \\"] + analysis) + "\n")


def l96_time_table(counts=(1024, 4096, 16384, 65536)):
    """Lorenz-96 (d = 40): time per assimilation step and of the analysis step (ms) of the PFs at the best
    parameter of Table 3, mean over the seeds of the per-run medians over the steps after the warm-up."""
    df = read_runs("lorenz96")
    methods = ("PF", "MPF", "RPF", "LW", "RMPF")
    best = {m: best_per_count(aggregate(df[df.method == m])).set_index("num_particles") for m in methods}
    analysis_time = {m: aggregate(df[df.method == m], time_col="median_filter_time")
                     .set_index(["param", "num_particles"]).time for m in methods}
    step, analysis = [], []
    for N in counts:
        lab = f"$2^{{{int(np.log2(N))}}}$ & "
        step.append(lab + " & ".join(f"{1e3 * best[m].time[N]:.1f}" for m in methods) + r" \\")
        analysis.append(lab + " & ".join(f"{1e3 * analysis_time[m][(best[m].param[N], N)]:.1f}" for m in methods)
                        + r" \\")
    write("table_l96_time_body.tex", "\n".join([r"\multicolumn{6}{l}{\textit{(a) Time per assimilation step (ms)}} \\"]
          + step + [r"\midrule", r"\multicolumn{6}{l}{\textit{(b) Time of the analysis step (ms)}} \\"] + analysis) + "\n")


def duplicates_table():
    f = sorted(glob.glob(os.path.join(ROOT, "results", "theory", "duplication_*.csv")))[-1]
    df = pd.read_csv(f)
    g = df.groupby(["benchmark", "num_particles", "method", "n"], sort=False).mean(numeric_only=True).reset_index()
    rows = []
    for _, r in g.iterrows():
        label = r"MPF (fixed $\boldsymbol{\alpha}$)" if r.method.startswith("MPF") else "RMPF"
        rows.append(f"{r.benchmark} & {r.num_particles} & {label} & {r.n} & {r.ess_median:.1f} & "
                    f"{r.distinct_mean:.2f} & {r.no_duplicate:.3f} & {r.k_upper_mean:.2f} & {r.copy_fraction:.3f} \\\\")
    write("table_duplicates_body.tex", "\n".join(rows) + "\n")


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(text)
    print("written", os.path.join("tables", name))


def main():
    best_table("table3_body.tex", [("lorenz63", "(a) Lorenz-63"), ("lorenz96", "(b) Lorenz-96")])
    l96_time_table()
    d80_table()
    duplicates_table()
    best_table("tableC1_abs_body.tex", [("lorenz96_abs", None)])
    best_table("tableC3_smalln_body.tex", [("lorenz63", r"(a) Lorenz-63"), ("lorenz96", r"(b) Lorenz-96")],
               subdir="experiments_smalln")


if __name__ == "__main__":
    main()
