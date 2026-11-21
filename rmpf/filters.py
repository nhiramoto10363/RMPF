"""
Filters compared in the paper.

  BootstrapPF       bootstrap particle filter (systematic resampling at every observation time)
  MPF               merging particle filter with fixed coefficients (Nakano et al. 2007)
  RMPF              randomized MPF: groups of n resampled parents are merged with random
                    mean-preserving orthogonal matrices (Sakov and Oke 2008)
  RegularizedPF     post-regularized PF with a Gaussian kernel (Musso et al. 2001)
  LiuWestPF         kernel-shrinkage PF (Liu and West 2001)
  EnKF              ensemble Kalman filter, square-root (ETKF) or perturbed-observation form,
                    no localization, multiplicative inflation

Common interface (one assimilation step = several ``forecast()`` calls followed by ``filtering(y)``):
  particles         (N, d) ensemble after the update
  estimate          state estimate at the observation time. For the particle filters this is the
                    importance-weighted mean *before* resampling (and merging / jittering); for the
                    EnKF it is the analysis mean.
  weights           normalized importance weights before resampling (uniform for the EnKF)
  log_lik_increment log((1/N) sum_i p(y_t | x_t^(i))), the increment of the log marginal likelihood
                    (particle filters only; NaN for the EnKF)

Every resampling step is systematic resampling at every observation time. Random numbers that are
drawn once and reused (MPF/RMPF merging matrices, RPF/LW jitter) are drawn when the filter is
constructed; see each class.
"""
import numpy as np
from numba import njit, prange

from .rng import draw_noise


# =============================================================================
# Shared kernels
# =============================================================================
@njit(fastmath=True)
def systematic_resampling(weights, num_particles):
    """Indices of systematic resampling (sorted in ascending order)."""
    indices = np.zeros(num_particles, dtype=np.int64)
    u0 = np.random.random() / num_particles
    cumulative = np.cumsum(weights)

    i, j = 0, 0
    while i < num_particles:
        u = u0 + i / num_particles
        left, right = j, num_particles - 1
        while left < right:
            mid = (left + right) // 2
            if u > cumulative[mid]:
                left = mid + 1
            else:
                right = mid
        j = left
        indices[i] = j
        i += 1

    return indices


@njit(fastmath=True)
def compute_weights_and_estimate(particles, loglikelihoods):
    """Normalized weights, weighted mean, and log((1/N) sum_i exp(loglikelihoods_i))."""
    max_ll = np.max(loglikelihoods)
    w_unn = np.exp(loglikelihoods - max_ll)
    w_sum = np.sum(w_unn)

    if w_sum <= 0:
        weights = np.ones_like(w_unn) / w_unn.size
    else:
        weights = w_unn / w_sum

    estimate = np.zeros(particles.shape[1])
    for i in range(particles.shape[0]):
        estimate += weights[i] * particles[i]

    return weights, estimate, max_ll + np.log(w_sum / w_unn.size)


class _ParticleFilterBase:
    """Holds the ensemble and the common forecast / weighting step."""

    def __init__(self, num_particles, dim_param, transition, observation, initial):
        if initial.shape != (num_particles, dim_param):
            raise ValueError("initial must have shape (num_particles, dim_param)")
        self.num_particles = num_particles
        self.dim_param = dim_param
        self.transition = transition
        self.observation = observation
        self.particles = initial
        self.estimate = None
        self.weights = None
        self.log_lik_increment = np.nan

    def forecast(self):
        self.particles = self.transition.forecast(self.particles)

    def _weight(self, obs):
        ll = self.observation.loglikelihoods(self.particles, obs)
        self.weights, self.estimate, self.log_lik_increment = compute_weights_and_estimate(self.particles, ll)


# =============================================================================
# Bootstrap PF
# =============================================================================
class BootstrapPF(_ParticleFilterBase):
    """Bootstrap PF. The estimate is the weighted mean before resampling, as for the other filters."""

    def filtering(self, obs):
        self._weight(obs)
        self.particles = self.particles[systematic_resampling(self.weights, self.num_particles)]


# =============================================================================
# MPF (fixed merging coefficients, Nakano et al. 2007)
# =============================================================================
@njit(fastmath=True)
def batch_resampling(weights, num_particles, n):
    """
    Parent indices for the MPF: n independent systematic resamplings of size N.

    Systematic resampling returns sorted indices, so the i-th members of the n sets would be (almost)
    the same parent. Each set is therefore randomly permuted before grouping. Group i consists of the
    i-th member of every set; the result is laid out as (N, n) in row-major order.
    """
    merged_idx = np.zeros(num_particles * n, dtype=np.int64)
    for k in range(n):
        idx = systematic_resampling(weights, num_particles)
        perm = np.random.permutation(num_particles)
        for i in range(num_particles):
            merged_idx[i * n + k] = idx[perm[i]]
    return merged_idx


@njit(parallel=True, fastmath=True)
def merge_fixed(parents, num_particles, n, alpha):
    """x_new_i = sum_j alpha_j x_parent(i, j) for parents laid out as (N * n, d)."""
    dim = parents.shape[1]
    groups = parents.reshape((num_particles, n, dim))
    merged = np.zeros((num_particles, dim))
    for i in prange(num_particles):
        for d in range(dim):
            v = 0.0
            for j in range(n):
                v += groups[i, j, d] * alpha[j]
            merged[i, d] = v
    return merged


class MPF(_ParticleFilterBase):
    """
    Merging particle filter with fixed coefficients alpha (sum alpha = 1, sum alpha^2 = 1).

    Each new particle merges n = len(alpha) parents drawn by n independent systematic resamplings.
    """

    def __init__(self, num_particles, dim_param, transition, observation, initial, alpha):
        super().__init__(num_particles, dim_param, transition, observation, initial)
        self.alpha = np.asarray(alpha, dtype=np.float64)
        if not (np.isclose(self.alpha.sum(), 1.0) and np.isclose(np.sum(self.alpha**2), 1.0)):
            raise ValueError("alpha must satisfy sum(alpha) = 1 and sum(alpha^2) = 1")
        self.n = len(self.alpha)

    def filtering(self, obs):
        self._weight(obs)
        idx = batch_resampling(self.weights, self.num_particles, self.n)
        self.particles = merge_fixed(self.particles[idx], self.num_particles, self.n, self.alpha)


# =============================================================================
# RMPF (random mean-preserving orthogonal merging)
# =============================================================================
@njit(fastmath=True)
def mean_preserving_orthogonal_batch(num_groups, n):
    """
    num_groups random orthogonal (n, n) matrices Omega with Omega 1 = 1 (Sakov and Oke 2008).

    Omega = 1 1^T / n + U Q U^T, where U is an orthonormal (Helmert) basis of the complement of 1 and
    Q is Haar-distributed on O(n-1) (QR of a Gaussian matrix with the sign of diag(R) fixed). Each
    column alpha satisfies sum alpha = 1 and sum alpha^2 = 1 and is uniformly distributed on that set.
    Uses the JIT-side RNG.
    """
    U = np.zeros((n, n - 1))
    for k in range(1, n):
        c = 1.0 / np.sqrt(k * (k + 1.0))
        for i in range(k):
            U[i, k - 1] = c
        U[k, k - 1] = -k * c
    Ut = np.ascontiguousarray(U.T)

    omegas = np.empty((num_groups, n, n))
    for g in range(num_groups):
        Z = np.random.normal(0.0, 1.0, (n - 1, n - 1))
        Q, R = np.linalg.qr(Z)
        for j in range(n - 1):
            if R[j, j] < 0:
                for i in range(n - 1):
                    Q[i, j] = -Q[i, j]
        omegas[g] = U @ np.ascontiguousarray(Q) @ Ut + 1.0 / n
    return omegas


@njit(fastmath=True)
def grouped_resampling(weights, num_particles, n):
    """
    One systematic resampling, randomly permuted and cut into groups of n: parent indices of length
    num_groups * n. If N is not a multiple of n, the last group is padded with the first parents and
    only N children are kept.
    """
    idx = systematic_resampling(weights, num_particles)
    perm = np.random.permutation(num_particles)
    num_groups = (num_particles + n - 1) // n
    out = np.empty(num_groups * n, dtype=np.int64)
    for i in range(num_groups * n):
        out[i] = idx[perm[i % num_particles]]
    return out


def random_merging_matrices(num_groups, n):
    """Transposed merging matrices Omega_g^T as a contiguous (num_groups, n, n) array."""
    return np.ascontiguousarray(mean_preserving_orthogonal_batch(num_groups, n).transpose(0, 2, 1))


class RMPF(_ParticleFilterBase):
    """
    Randomized merging particle filter.

    Resample N parents once, permute them, split them into groups of n, and replace each group by
    children = Omega_g^T parents_g. Each child is a random combination of its group with coefficients
    uniform on {sum alpha = 1, sum alpha^2 = 1}; the children of a group preserve its sample mean and
    covariance exactly.

    regenerate=False (default): the matrices Omega_g are drawn once at construction and reused at
    every assimilation step. True: drawn afresh at every step.
    """

    def __init__(self, num_particles, dim_param, transition, observation, initial, n, regenerate=False):
        if n < 2:
            raise ValueError("n must be >= 2")
        super().__init__(num_particles, dim_param, transition, observation, initial)
        self.n = n
        self.num_groups = (num_particles + n - 1) // n
        self.regenerate = regenerate
        self.omegas_t = random_merging_matrices(self.num_groups, n)

    def filtering(self, obs):
        self._weight(obs)
        parents = self.particles[grouped_resampling(self.weights, self.num_particles, self.n)]
        if self.regenerate:
            self.omegas_t = random_merging_matrices(self.num_groups, self.n)
        children = np.matmul(self.omegas_t, parents.reshape(self.num_groups, self.n, self.dim_param))
        self.particles = children.reshape(-1, self.dim_param)[:self.num_particles]


# =============================================================================
# Regularized PF (Musso et al. 2001) and Liu-West PF (Liu and West 2001)
# =============================================================================
def rpf_optimal_bandwidth(num_particles, dim):
    """h_opt = (4 / (d + 2))^{1/(d+4)} N^{-1/(d+4)} (Musso et al. 2001, Eq. 2.4), for whitened particles."""
    return (4.0 / (dim + 2.0)) ** (1.0 / (dim + 4.0)) * num_particles ** (-1.0 / (dim + 4.0))


def safe_cholesky(cov):
    """Cholesky factor; a scale-proportional ridge is added when the weights have degenerated."""
    d = cov.shape[0]
    jitter = 1e-10 * max(np.trace(cov) / d, 1e-12)
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(d))
        except np.linalg.LinAlgError:
            jitter *= 100.0
    return np.diag(np.sqrt(np.maximum(np.diag(cov), 0.0)))


class RegularizedPF(_ParticleFilterBase):
    """
    Post-regularized PF: weight -> resample -> x_i + h A eps_i, eps_i ~ N(0, I).

    A is the Cholesky factor of the weighted covariance before resampling and h = bandwidth_scale *
    h_opt (Musso et al. recommend bandwidth_scale = 1/2).

    regenerate=False (default): eps (N, d) is drawn once at construction and reused, as for the RMPF
    merging matrices. True: drawn afresh at every step. The jitter noise uses the chunked generator,
    so it does not depend on the number of threads.
    """

    def __init__(self, num_particles, dim_param, transition, observation, initial, bandwidth_scale=0.5,
                 regenerate=False, jitter_seed=None):
        super().__init__(num_particles, dim_param, transition, observation, initial)
        self.h = self._bandwidth(bandwidth_scale)
        if jitter_seed is None:
            jitter_seed = int(np.random.randint(0, 2**31 - 1))
        self.jitter_seed = jitter_seed
        self._jitter_step = 0
        self.regenerate = regenerate
        self.eps = None if regenerate else draw_noise(num_particles, dim_param, 1.0, jitter_seed, 0)

    def _bandwidth(self, bandwidth_scale):
        return bandwidth_scale * rpf_optimal_bandwidth(self.num_particles, self.dim_param)

    def _centers(self, sample):
        return sample

    def filtering(self, obs):
        self._weight(obs)
        anomalies = self.particles - self.estimate
        chol = safe_cholesky((anomalies * self.weights[:, None]).T @ anomalies)
        sample = self.particles[systematic_resampling(self.weights, self.num_particles)]

        if self.regenerate:
            eps = draw_noise(self.num_particles, self.dim_param, 1.0, self.jitter_seed, self._jitter_step)
            self._jitter_step += 1
        else:
            eps = self.eps
        self.particles = self._centers(sample) + self.h * (eps @ chol.T)


class LiuWestPF(RegularizedPF):
    """
    Liu-West kernel shrinkage: centers a x_i + (1 - a) m with a = sqrt(1 - h^2), so that the weighted
    mean and covariance are preserved. h (0 < h < 1) is used directly.
    """

    def __init__(self, num_particles, dim_param, transition, observation, initial, h=0.9,
                 regenerate=False, jitter_seed=None):
        if not 0.0 < h < 1.0:
            raise ValueError("Liu-West requires 0 < h < 1")
        super().__init__(num_particles, dim_param, transition, observation, initial, bandwidth_scale=h,
                         regenerate=regenerate, jitter_seed=jitter_seed)

    def _bandwidth(self, bandwidth_scale):
        return bandwidth_scale

    def _centers(self, sample):
        a = np.sqrt(1.0 - self.h**2)
        return a * sample + (1.0 - a) * self.estimate


# =============================================================================
# Ensemble Kalman filter
# =============================================================================
class EnKF:
    """
    EnKF without localization; multiplicative inflation of the forecast anomalies.

    perturbed_obs=False: ETKF with the symmetric square root (Bishop et al. 2001; Hunt et al. 2007),
    which preserves the mean. With X' (N x d) the forecast anomalies / sqrt(N-1) and
    S = R^{-1/2} H X'^T (p x N):
        mean:      x_a = x_f + X'^T S^T (I + S S^T)^{-1} R^{-1/2} (y - H x_f)
        anomalies: X'_a = T X', T = (I + S^T S)^{-1/2}
    computed from the thin SVD of S^T without forming N x N matrices (O(N p d)).

    perturbed_obs=True: stochastic EnKF (Burgers et al. 1998); each member is updated with y + e_i,
    e_i ~ N(0, R), centered over members. The perturbations use the chunked generator.

    The observation operator is the component selection of the observation model (obs_mask, or all
    components). An observation model with a method h(particles) is nonlinear: the update is then the
    EnKF of the augmented state [x, H(x)] (Evensen 2003; Nakano et al. 2007, Eq. 22), using the ensemble
    mean and anomalies of H(x_i), with the same inflation as the state anomalies. If a member becomes non-finite during the forecast, the filter keeps going with NaN
    and the run is recorded as diverged.
    """

    def __init__(self, num_particles, dim_param, transition, observation, initial, inflation=1.0,
                 perturbed_obs=False, obs_seed=None):
        if num_particles < 2:
            raise ValueError("EnKF requires num_particles >= 2")
        if initial.shape != (num_particles, dim_param):
            raise ValueError("initial must have shape (num_particles, dim_param)")
        self.num_particles = num_particles
        self.dim_param = dim_param
        self.transition = transition
        self.observation = observation
        self.particles = initial
        self.inflation = inflation
        mask = observation.obs_mask
        self.obs_idx = np.arange(dim_param) if mask is None else np.asarray(mask, dtype=np.int64)
        self.perturbed_obs = perturbed_obs
        if obs_seed is None:
            obs_seed = int(np.random.randint(0, 2**31 - 1))
        self.obs_seed = obs_seed
        self._obs_step = 0
        self.estimate = None
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self.log_lik_increment = np.nan

    def forecast(self):
        self.particles = self.transition.forecast(self.particles)

    def filtering(self, obs):
        N = self.num_particles
        sigma = self.observation.obs_noise_std
        mean_f = self.particles.mean(0)
        if not np.all(np.isfinite(mean_f)):
            self.estimate = mean_f
            return
        anom = (self.particles - mean_f) * self.inflation
        h = getattr(self.observation, "h", None)
        if h is None:
            St = anom[:, self.obs_idx] / (sigma * np.sqrt(N - 1.0))      # S^T: N x p
            innov = (obs[self.obs_idx] - mean_f[self.obs_idx]) / sigma    # R^{-1/2} (y - H x_f)
            hx = None
        else:
            hx = h(self.particles)
            hx_mean = hx.mean(0)
            hx = hx_mean + (hx - hx_mean) * self.inflation                # inflated H(x_i)
            St = (hx - hx_mean) / (sigma * np.sqrt(N - 1.0))
            innov = (obs[self.obs_idx] - hx_mean) / sigma

        if self.perturbed_obs:
            self._perturbed_obs_update(anom, mean_f, St, obs, sigma, hx)
            return

        V, s, U_t = np.linalg.svd(St, full_matrices=False)               # S^T = V diag(s) U^T
        s2 = s * s
        w_mean = V @ ((s / (1.0 + s2)) * (U_t @ innov))
        mean_a = mean_f + (anom.T @ w_mean) / np.sqrt(N - 1.0)
        f = 1.0 / np.sqrt(1.0 + s2) - 1.0                                 # T = I + V diag(f) V^T
        anom_a = anom + V @ (f[:, None] * (V.T @ anom))

        self.estimate = mean_a
        self.particles = mean_a + anom_a

    def _perturbed_obs_update(self, anom, mean_f, St, obs, sigma, hx=None):
        N, p = St.shape
        eps = draw_noise(N, p, 1.0, self.obs_seed, self._obs_step)
        self._obs_step += 1
        eps -= eps.mean(0)
        x = mean_f + anom                                                 # inflated members
        if hx is None:
            hx = x[:, self.obs_idx]
        D = obs[self.obs_idx] / sigma + eps - hx / sigma                  # R^{-1/2} (y + e_i - H x_i)
        G = np.linalg.solve(np.eye(p) + St.T @ St, D.T)
        M = anom.T @ St / np.sqrt(N - 1.0)                                # X'^T S^T
        self.particles = x + (M @ G).T
        self.estimate = self.particles.mean(0)
