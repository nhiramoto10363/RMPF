"""
Body of Table C4 (Appendix C4) from the newest results/outlier/summary_*.csv of outlier_merging.py:
d = 40, N = 4096, m = 1% of N, outlier weight 1. Means over the repetitions.
Usage:  python theory/outlier_table.py   -> tables/tableC4_outlier_body.tex
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = [("PF", "Bootstrap PF"), ("MPF:2", "MPF, set 2"), ("MPF:3", "MPF, set 3"), ("RMPF:3", "RMPF, $n=3$"),
           ("RMPF:10", "RMPF, $n=10$"), ("RMPF:40", "RMPF, $n=40$"), ("LW:0.5", "LW, $h=0.5$"),
           ("LW:0.9", "LW, $h=0.9$"), ("RPF:0.5", "RPF, $c=0.5$"), ("RPF:1", "RPF, $c=1$")]

s = pd.read_csv(sorted(glob.glob(os.path.join(ROOT, "results", "outlier", "summary_*.csv")))[-1])
s = s[(s.d == 40) & (s.N == 4096) & (s.m == 41) & (s.outlier_weight == 1)].set_index(["scenario", "method"])
lines = []
for m, label in METHODS:
    b, u, g = s.loc[("bad_obs", m)], s.loc[("bad_unobs", m)], s.loc[("good_obs", m)]
    cells = [f"{b.n_mirror:.1f}", f"{b.x1_max:.1f}", f"{b.bulk_var1:.2f}", f"{b.ess:.2f}", f"{b.dmean1:+.3f}",
             f"{u.mass_mirror:.4f}", f"{u.sd1:.2f}", f"{g.dmean1:+.2f}"]
    lines.append(label + " & " + " & ".join(cells) + r" \\")
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
out = os.path.join(ROOT, "tables", "tableC4_outlier_body.tex")
open(out, "w").write("\n".join(lines) + "\n")
ref = s.loc[("bad_unobs", "PF")]
print(open(out).read(), "reference sd of x1 (bad_unobs):", round(ref.ref_sd1, 3))
