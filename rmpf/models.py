"""
System and observation models: Lorenz-63 and Lorenz-96 with forward-Euler integration, additive
Gaussian process noise, and a Gaussian likelihood for direct (optionally masked) observations.
"""
import numpy as np
from numba import njit, prange

from .rng import draw_noise


# =============================================================================
# Lorenz-63
# =============================================================================
@njit(parallel=True, fastmath=True)
def lorenz63_forecast(params, dt, sigma, rho, beta, noise):
    """One Euler step x + dt f(x) + noise for every particle (noise is drawn by the caller)."""
    if params.ndim == 1:
        params = params.reshape(1, -1)

    num_particles = params.shape[0]
    X_new = np.zeros_like(params)

    for p in prange(num_particles):
        x = params[p, 0]
        y = params[p, 1]
        z = params[p, 2]

        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z

        X_new[p, 0] = x + dt * dx + noise[p, 0]
        X_new[p, 1] = y + dt * dy + noise[p, 1]
        X_new[p, 2] = z + dt * dz + noise[p, 2]

    return X_new


class Lorenz63Transition:
    """
    Lorenz-63 transition for one Euler step of length dt with N(0, proc_noise_std^2 I) noise.

    noise_seed=None draws the seed from the global np.random state, so the model is deterministic
    once np.random.seed() has been called.
    """

    def __init__(self, dt=0.01, sigma=10.0, rho=28.0, beta=8 / 3, proc_noise_std=0.01, noise_seed=None):
        self.dt = dt
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.proc_noise_std = proc_noise_std
        if noise_seed is None:
            noise_seed = int(np.random.randint(0, 2**31 - 1))
        self.noise_seed = noise_seed
        self._noise_step = 0

    def forecast(self, params):
        num_particles, dim = params.shape
        noise = draw_noise(num_particles, dim, self.proc_noise_std, self.noise_seed, self._noise_step)
        self._noise_step += 1
        return lorenz63_forecast(params, self.dt, self.sigma, self.rho, self.beta, noise)


# =============================================================================
# Lorenz-96
# =============================================================================
@njit(parallel=True, fastmath=True)
def lorenz96_forecast(params, dim, dt, F, noise, idx_ip1, idx_im1, idx_im2):
    """
    One Euler step x + dt f(x) + noise for every particle (noise is drawn by the caller).

    No array is allocated inside prange: under numba < 0.57 every allocation updates global counters
    atomically, which serialized the threads (about 10x slower on a 28-core node, slower still with
    more sockets). dx is written to the output row first and then updated, which gives results
    bit-identical to the earlier array-expression version (X[idx_ip1] - X[idx_im2]) * X[idx_im1] - X + F.
    """
    if params.ndim == 1:
        params = params.reshape(1, -1)

    num_particles = params.shape[0]
    X_new = np.empty_like(params)

    for p in prange(num_particles):
        for j in range(dim):
            X_new[p, j] = (params[p, idx_ip1[j]] - params[p, idx_im2[j]]) * params[p, idx_im1[j]] - params[p, j] + F
        for j in range(dim):
            X_new[p, j] = params[p, j] + dt * X_new[p, j] + noise[p, j]

    return X_new


class Lorenz96Transition:
    """Lorenz-96 transition for one Euler step of length dt with N(0, proc_noise_std^2 I) noise."""

    def __init__(self, dim=40, F=8.0, dt=0.005, proc_noise_std=0.5, noise_seed=None):
        self.dim = dim
        self.F = F
        self.dt = dt
        self.proc_noise_std = proc_noise_std
        if noise_seed is None:
            noise_seed = int(np.random.randint(0, 2**31 - 1))
        self.noise_seed = noise_seed
        self._noise_step = 0

        self.idx_ip1 = np.array([(i + 1) % dim for i in range(dim)], dtype=np.int64)
        self.idx_im1 = np.array([(i - 1) % dim for i in range(dim)], dtype=np.int64)
        self.idx_im2 = np.array([(i - 2) % dim for i in range(dim)], dtype=np.int64)

    def forecast(self, params):
        num_particles, dim = params.shape
        noise = draw_noise(num_particles, dim, self.proc_noise_std, self.noise_seed, self._noise_step)
        self._noise_step += 1
        return lorenz96_forecast(params, self.dim, self.dt, self.F, noise,
                                 self.idx_ip1, self.idx_im1, self.idx_im2)


# =============================================================================
# Observation model
# =============================================================================
@njit(parallel=True, fastmath=True)
def gaussian_loglik_vectorized(params, obs, obs_noise_std, obs_mask=None):
    """log N(obs[mask] | params[:, mask], obs_noise_std^2 I) for every particle (all components if mask is None)."""
    num_particles = params.shape[0]
    ll = np.zeros(num_particles)
    var = obs_noise_std**2
    log_term = -0.5 * np.log(2.0 * np.pi * var)

    if obs_mask is None:
        obs_size = obs.size
        for i in prange(num_particles):
            diff2 = 0.0
            for j in range(obs_size):
                diff = params[i, j] - obs[j]
                diff2 += diff * diff
            ll[i] = -0.5 * diff2 / var + obs_size * log_term
        return ll
    else:
        obs_mask_len = len(obs_mask)
        for i in prange(num_particles):
            diff2 = 0.0
            for j in range(obs_mask_len):
                idx = obs_mask[j]
                diff = params[i, idx] - obs[idx]
                diff2 += diff * diff
            ll[i] = -0.5 * diff2 / var + obs_mask_len * log_term
        return ll


class GaussianObservation:
    """
    Linear observation y = x[obs_mask] + e, e ~ N(0, obs_noise_std^2 I).

    obs_mask=None observes every component. The observation vector passed to loglikelihoods has the
    full state length; only the masked components are used.
    """

    def __init__(self, obs_noise_std, obs_mask=None):
        self.obs_noise_std = obs_noise_std
        self.obs_mask = None if obs_mask is None else np.asarray(obs_mask, dtype=np.int64)

    def loglikelihoods(self, params, obs):
        return gaussian_loglik_vectorized(params, obs, self.obs_noise_std, self.obs_mask)


@njit(parallel=True, fastmath=True)
def abs_gaussian_loglik_vectorized(params, obs, obs_noise_std, obs_mask):
    """log N(obs[mask] | |params[:, mask]|, obs_noise_std^2 I) for every particle."""
    num_particles = params.shape[0]
    ll = np.zeros(num_particles)
    var = obs_noise_std**2
    log_term = -0.5 * np.log(2.0 * np.pi * var)
    obs_mask_len = len(obs_mask)
    for i in prange(num_particles):
        diff2 = 0.0
        for j in range(obs_mask_len):
            idx = obs_mask[j]
            diff = abs(params[i, idx]) - obs[idx]
            diff2 += diff * diff
        ll[i] = -0.5 * diff2 / var + obs_mask_len * log_term
    return ll


class AbsGaussianObservation:
    """
    Nonlinear observation y = |x[obs_mask]| + e, e ~ N(0, obs_noise_std^2 I) (Nakano et al. 2007, Sect. 4.3).

    h(params) applies the observation operator to an ensemble (N x len(obs_mask)); the EnKFs use it.
    """

    def __init__(self, obs_noise_std, obs_mask):
        self.obs_noise_std = obs_noise_std
        self.obs_mask = np.asarray(obs_mask, dtype=np.int64)

    def loglikelihoods(self, params, obs):
        return abs_gaussian_loglik_vectorized(params, obs, self.obs_noise_std, self.obs_mask)

    def h(self, params):
        return np.abs(params[:, self.obs_mask])


# =============================================================================
# Truth trajectories (noise-free Euler integration, as in the paper)
# =============================================================================
@njit
def lorenz63_trajectory(x0, steps, dt, sigma, rho, beta, warmup_steps):
    state = x0.copy()
    for _ in range(warmup_steps):
        dx = sigma * (state[1] - state[0])
        dy = state[0] * (rho - state[2]) - state[1]
        dz = state[0] * state[1] - beta * state[2]
        state[0] += dt * dx
        state[1] += dt * dy
        state[2] += dt * dz

    X = np.zeros((steps, 3))
    X[0] = state
    for step in range(1, steps):
        dx = sigma * (X[step - 1, 1] - X[step - 1, 0])
        dy = X[step - 1, 0] * (rho - X[step - 1, 2]) - X[step - 1, 1]
        dz = X[step - 1, 0] * X[step - 1, 1] - beta * X[step - 1, 2]
        X[step, 0] = X[step - 1, 0] + dt * dx
        X[step, 1] = X[step - 1, 1] + dt * dy
        X[step, 2] = X[step - 1, 2] + dt * dz
    return X


@njit
def lorenz96_trajectory(x0, dim, F, dt, steps, warmup_steps):
    idx_ip1 = np.array([(i + 1) % dim for i in range(dim)], dtype=np.int64)
    idx_im1 = np.array([(i - 1) % dim for i in range(dim)], dtype=np.int64)
    idx_im2 = np.array([(i - 2) % dim for i in range(dim)], dtype=np.int64)

    state = x0.copy()
    for _ in range(warmup_steps):
        dx = np.zeros(dim)
        for i in range(dim):
            dx[i] = (state[idx_ip1[i]] - state[idx_im2[i]]) * state[idx_im1[i]] - state[i] + F
        new_state = np.zeros(dim)
        for i in range(dim):
            new_state[i] = state[i] + dt * dx[i]
        state = new_state

    X = np.zeros((steps, dim))
    X[0] = state
    for step in range(1, steps):
        dx = np.zeros(dim)
        for i in range(dim):
            dx[i] = (X[step - 1, idx_ip1[i]] - X[step - 1, idx_im2[i]]) * X[step - 1, idx_im1[i]] - X[step - 1, i] + F
        for i in range(dim):
            X[step, i] = X[step - 1, i] + dt * dx[i]
    return X
