"""
Figure 1 of the paper: law of a merged particle for d = 2.

Left: MPF with the fixed coefficients of Nakano et al. (2007) (set 2), n = 3: the merged particles of three
parents are the 3! = 6 linear combinations with the permuted coefficients (discrete atoms).
Right: RMPF with n = 3, 4, 5, 6 parents equally spaced on the unit circle (k = 2): density of
Proposition 3, proportional to (1 - q)_+^((n - k - 3)/2) with q = (x - xbar)^T S^+ (x - xbar) / a,
a = (n - 1)/n; for n = k + 1 = 3 the child lies on the ellipse q = 1.

The script also checks the drawn densities against children sampled with the random orthogonal
matrices of RMPF (rmpf.filters.mean_preserving_orthogonal_batch): the distribution of q of the
sampled merged particles is compared with Beta(k/2, (n-1-k)/2) (n > 3) or q = 1 (n = 3).

Usage:  python theory/plot_merging_density.py   ->  figures/merging_density.{pdf,png}
"""
import os
import sys
from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rmpf.filters import mean_preserving_orthogonal_batch  # noqa: E402
from rmpf.rng import set_numba_seed  # noqa: E402

FIXED_ALPHA = np.array([3 / 4, (np.sqrt(13) + 1) / 8, -(np.sqrt(13) - 1) / 8])  # set 2
N_LIST = (3, 4, 5, 6)
LABELS = {3: "ellipse", 4: "boundary-concentrated", 5: "uniform", 6: "centre-concentrated"}
LIM = 1.7
RES = 600


def parents_on_circle(n):
    th = 2 * np.pi * np.arange(n) / n + np.pi / 2
    return np.vstack([np.cos(th), np.sin(th)])  # d x n


def q_of(points, X):
    """q(x) = (x - xbar)^T S^+ (x - xbar) / a for points of shape (m, d)."""
    n = X.shape[1]
    P = np.eye(n) - np.ones((n, n)) / n
    S = X @ P @ X.T
    diff = points - X.mean(axis=1)
    return np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(S), diff) / ((n - 1) / n)


def density_grid(X):
    """Unnormalized density of Proposition 3 on a grid (k = 2 < n - 1, i.e. n >= 4)."""
    n, k = X.shape[1], 2
    xs = np.linspace(-LIM, LIM, RES)
    gx, gy = np.meshgrid(xs, xs)
    q = q_of(np.column_stack([gx.ravel(), gy.ravel()]), X).reshape(RES, RES)
    dens = np.zeros_like(q)
    inside = q < 1
    dens[inside] = (1 - q[inside]) ** ((n - k - 3) / 2)
    return xs, dens, inside


def check_against_sampling(n_children=200_000):
    """Compare q of children drawn with RMPF's orthogonal matrices with Proposition 3."""
    set_numba_seed(1)
    k = 2
    for n in N_LIST:
        X = parents_on_circle(n)
        omegas = mean_preserving_orthogonal_batch(n_children // n + 1, n)  # (G, n, n)
        children = np.einsum("dn,gnm->gmd", X, omegas).reshape(-1, 2)
        q = q_of(children, X)
        if n == k + 1:
            print(f"n={n}: max |q - 1| = {np.max(np.abs(q - 1)):.1e}")
        else:
            ks = stats.kstest(q, stats.beta(k / 2, (n - 1 - k) / 2).cdf)
            print(f"n={n}: KS distance to Beta({k / 2}, {(n - 1 - k) / 2}) = {ks.statistic:.4f} "
                  f"(p = {ks.pvalue:.2f}, {len(q)} children)")


def main():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.15), sharey=True)
    parent_style = dict(marker="s", s=22, facecolor="white", edgecolor="black", linewidth=0.9, zorder=4)

    # MPF with fixed coefficients: the six atoms X alpha for the permutations of alpha
    ax = axes[0]
    X = parents_on_circle(3)
    atoms = np.array([X @ np.array(p) for p in set(permutations(FIXED_ALPHA))])
    ax.scatter(atoms[:, 0], atoms[:, 1], marker="^", s=30, facecolor="0.45", edgecolor="black",
               linewidth=0.6, zorder=3)
    ax.scatter(X[0], X[1], **parent_style)
    ax.set_title("MPF, fixed coefficients\n($n=3$)")

    for ax, n in zip(axes[1:], N_LIST):
        X = parents_on_circle(n)
        if n == 3:
            th = np.linspace(0, 2 * np.pi, 400)
            ax.plot(np.cos(th), np.sin(th), color="black", linewidth=1.4)  # q = 1, radius 1 here
        else:
            xs, dens, inside = density_grid(X)
            vals = dens[inside]
            if n == 4:      # diverges at the boundary: saturate the colour scale
                vmax = np.percentile(vals, 96)
            elif n == 5:    # constant: draw it in mid grey
                vmax = 2 * vals.max()
            else:
                vmax = vals.max()
            shown = np.where(inside, dens, np.nan)
            if n == 5:      # constant inside: a filled region (matplotlib 3.5 writes a two-colour image to
                            # the PDF as a 1-bit palette image, which PDF viewers draw with stripes)
                ax.contourf(xs, xs, inside.astype(float), levels=[0.5, 1.5],
                            colors=[plt.get_cmap("Greys")(vals.max() / vmax)], zorder=1)
            else:
                ax.imshow(shown, origin="lower", extent=[-LIM, LIM, -LIM, LIM], cmap="Greys",
                          vmin=0, vmax=vmax, interpolation="bilinear", zorder=1)
            ax.contour(xs, xs, inside.astype(float), levels=[0.5], colors="black", linewidths=0.8)
            if n != 5:
                levels = np.linspace(vals.min(), vmax, 6)[1:-1]
                ax.contour(xs, xs, shown, levels=levels, colors="0.25", linewidths=0.5)
        ax.scatter(X[0], X[1], **parent_style)
        ax.set_title(f"RMPF, $n={n}$\n{LABELS[n]}")

    for ax in axes:
        ax.set_xlim(-LIM, LIM)
        ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_xlabel("$x_1$", labelpad=1)
        ax.tick_params(length=2.5, pad=1.5)
    axes[0].set_ylabel("$x_2$", labelpad=1)

    handles = [Line2D([], [], linestyle="none", marker="s", markersize=4.5, markerfacecolor="white",
                      markeredgecolor="black"),
               Line2D([], [], linestyle="none", marker="^", markersize=5, markerfacecolor="0.45",
                      markeredgecolor="black")]
    fig.legend(handles, ["resampled particles", "merged particles of the fixed-coefficient MPF"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.3, columnspacing=1.5)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.83, bottom=0.2, wspace=0.12)

    out = os.path.join(ROOT, "figures", "merging_density")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=400)
    print(f"Saved: {out}.pdf, .png")


if __name__ == "__main__":
    check_against_sampling()
    main()
