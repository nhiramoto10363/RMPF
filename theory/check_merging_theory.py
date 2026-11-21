"""
Numerical checks of the statements on the merging step used in the paper (Sect. 3-4 and Appendix).

Notation. A group of n parents is the d x n matrix X, P = I - 1 1^T / n, xbar = X 1 / n,
S = X P X^T, k = rank(S) = affine dimension of the parents (k <= min(n_distinct - 1, d)).
A child is x = X alpha with sum alpha = 1, sum alpha^2 = 1. RMPF uses the columns of a random
orthogonal matrix Omega with Omega 1 = 1 (Sakov and Oke 2008, Eqs. 17-19), drawn by
rmpf.filters.mean_preserving_orthogonal_batch.

Checks
  C1  every column of Omega satisfies the constraints and has the law of the coefficient sampler
      alpha = 1/n + sqrt((n-1)/n) * v, v uniform on the unit sphere of {1^T v = 0}
  C2  the children X Omega of a group have exactly the sample mean and covariance of the parents
  C3  conditional on X, a child has mean xbar and covariance S / n
      (kernel shrinkage with centre xbar and h^2 = 1 - 1/n in units of the group covariance S/(n-1))
  C4  law of a child for fixed X (Cambanis et al. 1981, Theorem 1 and Lemma 2):
      q = (x - xbar)^T S^+ (x - xbar) / a,  a = (n-1)/n
        k = n - 1  : q = 1               (child on an (n-2)-dimensional ellipsoid surface)
        k < n - 1  : q ~ Beta(k/2, (n-1-k)/2), i.e. density of x on the k-dim. affine hull
                     proportional to (1 - q)^((n-k-3)/2)
      including groups with duplicated parents, for which k < n - 1 even when n <= d + 1
  C5  normalising constant of that density (for k < n - 1)
  C6  probability that a group has no duplicated parent, systematic resampling + random permutation
      (as in RMPF) vs. multinomial resampling and the approximation exp(-n(n-1) / (2 N_eff))

Usage:  python theory/check_merging_theory.py   (prints PASS/FAIL per check; a few seconds)
"""
import os
import sys

import numpy as np
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rmpf.filters import mean_preserving_orthogonal_batch, systematic_resampling  # noqa: E402
from rmpf.rng import set_numba_seed  # noqa: E402

rng = np.random.default_rng(0)
set_numba_seed(0)
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def omega_columns(n, num):
    """num columns of independent Omega matrices (one column per matrix: independent draws)."""
    om = mean_preserving_orthogonal_batch(num, n)
    return om[np.arange(num), :, rng.integers(0, n, num)]          # (num, n)


def paper_sampler(n, num):
    u = rng.normal(size=(num, n))
    z = u - u.mean(1, keepdims=True)
    v = z / np.linalg.norm(z, axis=1, keepdims=True)
    return v * np.sqrt((n - 1) / n) + 1.0 / n


def ks(sample, cdf):
    return stats.kstest(sample, cdf).pvalue


# C1 ------------------------------------------------------------------------------------------
for n in (3, 10, 41):
    a = omega_columns(n, 20000)
    b = paper_sampler(n, 20000)
    err = max(np.abs(a.sum(1) - 1).max(), np.abs((a**2).sum(1) - 1).max())
    # alpha_1 - 1/n = sqrt((n-1)/n) * s * sqrt((n-1)/n), s = first coord. of a uniform point of S^{n-2}
    # (s + 1)/2 ~ Beta((n-2)/2, (n-2)/2)
    to_s = lambda al: (al[:, 0] - 1.0 / n) * n / (n - 1)
    p_ref = ks((to_s(a) + 1) / 2, stats.beta((n - 2) / 2, (n - 2) / 2).cdf)
    p_two = stats.ks_2samp(a[:, 0], b[:, 0]).pvalue
    report(f"C1 n={n}", err < 1e-12 and p_ref > 1e-3 and p_two > 1e-3,
           f"max constraint error {err:.1e}, KS p (vs theory) {p_ref:.3f}, KS p (vs Sect. 3.3 sampler) {p_two:.3f}")

# C2 ------------------------------------------------------------------------------------------
for n, d in ((3, 3), (40, 40), (100, 40)):
    X = rng.normal(size=(d, n))
    om = mean_preserving_orthogonal_batch(1, n)[0]
    Y = X @ om
    e_mean = np.abs(Y.mean(1) - X.mean(1)).max()
    e_cov = np.abs(np.cov(Y) - np.cov(X)).max()
    e_orth = np.abs(om @ om.T - np.eye(n)).max()
    report(f"C2 n={n} d={d}", max(e_mean, e_cov, e_orth) < 1e-10,
           f"|mean diff| {e_mean:.1e}, |cov diff| {e_cov:.1e}, |Omega Omega^T - I| {e_orth:.1e}")

# C3 ------------------------------------------------------------------------------------------
for n, d in ((3, 2), (10, 3), (45, 40)):
    X = rng.normal(size=(d, n)) * np.linspace(0.5, 2, d)[:, None]
    P = np.eye(n) - 1.0 / n
    S = X @ P @ X.T
    kids = omega_columns(n, 100000) @ X.T
    e_m = np.abs(kids.mean(0) - X.mean(1)).max() / np.sqrt(np.diag(S / n)).max()
    e_c = np.abs(np.cov(kids.T) - S / n).max() / np.abs(S / n).max()
    report(f"C3 n={n} d={d}", e_m < 0.02 and e_c < 0.02,
           f"mean error {e_m:.3f} sd, relative covariance error {e_c:.3f} (Monte Carlo, 1e5 draws)")


# C4 ------------------------------------------------------------------------------------------
def q_stat(X, kids):
    n = X.shape[1]
    P = np.eye(n) - 1.0 / n
    S = X @ P @ X.T
    r = kids - X.mean(1)
    k = np.linalg.matrix_rank(S, tol=1e-10 * max(1.0, np.abs(X).max() ** 2))   # absolute tolerance: S = 0 has rank 0
    Sp = np.linalg.pinv(S, rcond=1e-10) if k > 0 else np.zeros_like(S)
    return np.einsum("ij,jk,ik->i", r, Sp, r) / ((n - 1) / n), k


cases = [  # (description, d, n, number of distinct parents)
    ("surface, n = d+1", 2, 3, 3), ("edge, n = d+2", 2, 4, 4), ("uniform, n = d+3", 2, 5, 5),
    ("central, n = d+4", 2, 6, 6), ("surface, L96 n=10", 40, 10, 10), ("central, L96 n=45", 40, 45, 45),
    ("duplicates: n=3, 2 distinct", 2, 3, 2), ("duplicates: n=10, 4 distinct, d=40", 40, 10, 4),
    ("duplicates: n=45, 30 distinct, d=40", 40, 45, 30), ("all identical, n=5", 3, 5, 1),
]
for desc, d, n, nd in cases:
    base = rng.normal(size=(d, nd))
    X = base[:, np.r_[np.arange(nd), rng.integers(0, nd, n - nd)]]     # n parents, nd distinct
    q, k = q_stat(X, omega_columns(n, 20000) @ X.T)
    if k == 0:
        spread = np.abs(omega_columns(n, 1000) @ X.T - X[:, 0]).max()
        report(f"C4 {desc}", spread < 1e-12, f"k=0: every child is a copy (max deviation {spread:.1e})")
    elif k == n - 1:
        report(f"C4 {desc}", np.abs(q - 1).max() < 1e-8, f"k={k}=n-1: q = 1 for all children "
               f"(max |q-1| {np.abs(q - 1).max():.1e}); support is an (n-2)-dim. ellipsoid surface")
    else:
        p = ks(q, stats.beta(k / 2, (n - 1 - k) / 2).cdf)
        expo = (n - k - 3) / 2
        report(f"C4 {desc}", p > 1e-3, f"k={k}: q ~ Beta({k/2:g}, {(n-1-k)/2:g}), KS p {p:.3f}; "
               f"density exponent (n-k-3)/2 = {expo:g}")

# C5 ------------------------------------------------------------------------------------------
# density of the child on its k-dim. hull, in coordinates t = L (x - xbar) with S^+ = L^T L:
#   f(t) = C * (a - |t|^2)^beta,  beta = (n-k-3)/2,  C = Gamma((n-1)/2) / (pi^{k/2} Gamma((n-k-1)/2)) a^{-(n-3)/2}
# check: probability of the ball |t|^2 <= c a equals the Beta(k/2, (n-1-k)/2) cdf at c (radial law of C4)
for n, k in ((4, 2), (5, 2), (8, 3), (45, 40)):
    a = (n - 1) / n
    beta_ = (n - k - 3) / 2
    logC = special.gammaln((n - 1) / 2) - k / 2 * np.log(np.pi) - special.gammaln((n - k - 1) / 2) \
        - (n - 3) / 2 * np.log(a)
    # integral over |t|^2 <= c a of C (a - |t|^2)^beta dt = C * surface(S^{k-1}) * int_0^sqrt(ca) (a-s^2)^beta s^{k-1} ds
    c = 0.5
    s = np.linspace(0, np.sqrt(c * a), 200001)
    radial = np.trapezoid((a - s**2) ** beta_ * s ** (k - 1), s)
    mass = np.exp(logC) * 2 * np.pi ** (k / 2) / special.gamma(k / 2) * radial
    target = stats.beta(k / 2, (n - 1 - k) / 2).cdf(c)
    total = np.exp(logC) * np.pi ** (k / 2) * special.gamma(beta_ + 1) / special.gamma(beta_ + 1 + k / 2) \
        * a ** (beta_ + k / 2)
    report(f"C5 n={n} k={k}", abs(mass - target) < 1e-4 and abs(total - 1) < 1e-10,
           f"total mass {total:.12f}, P(q <= 0.5): density {mass:.6f} vs Beta cdf {target:.6f}")


# C6 ------------------------------------------------------------------------------------------
def no_dup_prob(weights, n, scheme, reps=2000):
    N = len(weights)
    ok = 0
    for _ in range(reps):
        if scheme == "systematic+perm":
            idx = systematic_resampling(weights, N)[rng.permutation(N)]
        else:
            idx = rng.choice(N, N, p=weights)
        groups = idx[: (N // n) * n].reshape(-1, n)
        ok += np.mean([len(np.unique(g)) == n for g in groups])
    return ok / reps


N = 1024
for ess_target in (50, 200, 800):
    # log-normal weights tuned to the target ESS
    for s in np.linspace(0.01, 5, 2000):
        w = np.exp(s * np.random.default_rng(1).normal(size=N))
        w /= w.sum()
        if 1 / np.sum(w**2) <= ess_target:
            break
    ess = 1 / np.sum(w**2)
    for n in (3, 10, 40):
        p_sys = no_dup_prob(w, n, "systematic+perm", 300)
        p_mul = no_dup_prob(w, n, "multinomial", 300)
        approx = np.exp(-n * (n - 1) / (2 * ess))
        print(f"[info] C6 N={N} ESS={ess:6.1f} n={n:3d}: P(no duplicate in a group) "
              f"systematic+perm {p_sys:.3f}, multinomial {p_mul:.3f}, exp(-n(n-1)/2ESS) {approx:.3f}")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
