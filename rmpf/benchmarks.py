"""
Benchmark settings (Lorenz-63, Lorenz-96), the method grid, and the filter factory.

A method is written "NAME" or "NAME:PARAM":
  PF            bootstrap PF
  MPF:k         MPF with the fixed coefficient set k of MPF_COEFFICIENTS (k = 1, 2, 3; numbering of Table 3)
  RMPF:n        RMPF with merge size n
  LW:h          Liu-West PF with h (0 < h < 1)
  RPF:c         regularized PF with bandwidth h = c * h_opt
  ETKF:lam      square-root EnKF with multiplicative inflation lam
  EnKF:lam      perturbed-observation EnKF with multiplicative inflation lam
"""
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .filters import EnKF, LiuWestPF, MPF, RMPF, BootstrapPF, RegularizedPF
from .models import AbsGaussianObservation, GaussianObservation, Lorenz63Transition, Lorenz96Transition

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed MPF coefficients (Table 3 of the manuscript). Set 2 is the one of Nakano et al. (2007):
# (3/4, (sqrt(13) + 1)/8, -(sqrt(13) - 1)/8).
MPF_COEFFICIENTS = {
    1: np.array([-0.13807119, 0.1607873, 0.97728388]),
    2: np.array([0.75, 0.5756939094329987, -0.32569390943299864]),
    3: np.array([2 / 3, 2 / 3, -1 / 3]),
}

# Particle numbers used during development (both benchmarks); the full grids are in Benchmark.full_counts.
DEV_COUNTS = [64, 1024, 4096]
SEEDS = list(range(42, 52))  # seeds 42-51 (ten seeds)
LW_BANDWIDTHS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
RPF_SCALES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
ENKF_INFLATIONS = [1.0, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5]


@dataclass
class Benchmark:
    key: str
    name: str
    dim: int
    dt: float
    obs_interval: int            # model steps per assimilation step
    proc_noise_std: float        # process noise of the filters (the truth is noise-free)
    obs_noise_std: float         # used both to generate the data and in the likelihood
    obs_mask: object             # observed components (None: all)
    warmup_steps: int            # assimilation steps excluded from the time averages
    merge_sizes: list
    full_counts: list
    enkf_max_count: int          # EnKF is run only up to this ensemble size
    obs_operator: str = "identity"   # "identity" (y = x + e) or "abs" (y = |x + e|)
    lik_std: float = None        # std of the Gaussian likelihood if it differs from obs_noise_std
    data_dir: str = field(init=False)

    def __post_init__(self):
        self.data_dir = os.path.join(ROOT, "data", self.key)

    def transition(self):
        if self.key == "lorenz63":
            return Lorenz63Transition(dt=self.dt, proc_noise_std=self.proc_noise_std)
        return Lorenz96Transition(dim=self.dim, dt=self.dt, proc_noise_std=self.proc_noise_std)

    def observation(self):
        std = self.obs_noise_std if self.lik_std is None else self.lik_std
        if self.obs_operator == "abs":
            return AbsGaussianObservation(std, self.obs_mask)
        return GaussianObservation(std, self.obs_mask)

    def initial_ensemble(self, observations, num_particles):
        if self.key == "lorenz63":
            return np.random.normal(observations[0], 2.0, (num_particles, self.dim))
        return np.random.normal(2.0, np.sqrt(2.0), (num_particles, self.dim))

    def build_models(self, observations, num_particles):
        """Transition, observation model and initial ensemble, drawn from the global np.random state in this order."""
        transition = self.transition()
        observation = self.observation()
        initial = self.initial_ensemble(observations, num_particles)
        return transition, observation, initial

    def load_data(self, max_steps=None):
        truth = np.loadtxt(os.path.join(self.data_dir, "true_states.csv"), delimiter=",", skiprows=1)
        obs = np.loadtxt(os.path.join(self.data_dir, "observations.csv"), delimiter=",", skiprows=1)
        T = min(len(truth), max_steps) if max_steps else len(truth)
        return obs[:T], truth[:T]


BENCHMARKS = {
    "l63": Benchmark(
        key="lorenz63", name="Lorenz-63", dim=3, dt=0.01, obs_interval=20,
        proc_noise_std=0.01, obs_noise_std=1.0, obs_mask=None, warmup_steps=100,
        merge_sizes=[3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 100],
        full_counts=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        enkf_max_count=8192,
    ),
    "l96": Benchmark(
        key="lorenz96", name="Lorenz-96", dim=40, dt=0.005, obs_interval=10,
        proc_noise_std=0.5, obs_noise_std=1.5, obs_mask=np.arange(0, 40, 2), warmup_steps=300,
        merge_sizes=[3, 5, 10, 20, 30, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 80, 100],
        full_counts=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        enkf_max_count=8192,
    ),
}

# Nonlinear observation (Nakano et al. 2007, Sect. 4.3): y = |x + e| on the data of l96 (same truth and
# noise), Gaussian likelihood around |x| with std 3; everything else as l96.
BENCHMARKS["l96abs"] = replace(BENCHMARKS["l96"], key="lorenz96_abs", name="Lorenz-96, |x| observed",
                               obs_operator="abs", lik_std=3.0)

# High-dimensional Lorenz-96 (d = 80, every second variable observed; data/lorenz96_d80 from
# generate_data.py --l96-dims 80). Particle filters need very large N here (with N = 16384 no filter
# tracked the truth); the comparison uses fixed parameters, one seed and 1000 steps
# (run_experiments.py l96d80 --full --seeds 42 --max-steps 1000, one job per method, see job.sh).
BENCHMARKS["l96d80"] = replace(BENCHMARKS["l96"], key="lorenz96_d80", name="Lorenz-96, d=80", dim=80,
                               obs_mask=np.arange(0, 80, 2), merge_sizes=[100],
                               full_counts=[2**16, 2**18, 2**20, 2**22], enkf_max_count=1024)


def lorenz96_with_dim(dim):
    """
    Lorenz-96 with state dimension dim and otherwise the settings of BENCHMARKS["l96"] (every second
    variable observed). dim = 40 is BENCHMARKS["l96"] itself; other dimensions use data/lorenz96_d<dim>
    (python generate_data.py --l96-dims ...).
    """
    base = BENCHMARKS["l96"]
    if dim == base.dim:
        return base
    return replace(base, key=f"lorenz96_d{dim}", name="Lorenz-96", dim=dim, obs_mask=np.arange(0, dim, 2))


def parse_method(method):
    name, _, param = method.partition(":")
    return name, (float(param) if param else 0.0)


def method_label(name, param):
    return name if name == "PF" else "%s:%g" % (name, param)


def method_grid(bench, counts):
    """All (method, N) pairs of the comparison. RMPF with n > N and EnKF beyond enkf_max_count are skipped."""
    methods = (["PF"] + ["MPF:%d" % k for k in MPF_COEFFICIENTS]
               + ["RMPF:%d" % n for n in bench.merge_sizes]
               + ["LW:%g" % h for h in LW_BANDWIDTHS] + ["RPF:%g" % c for c in RPF_SCALES]
               + ["%s:%g" % (m, lam) for m in ("ETKF", "EnKF") for lam in ENKF_INFLATIONS])
    grid = []
    for N in counts:
        for m in methods:
            name, param = parse_method(m)
            if name == "RMPF" and param > N:
                continue
            if name in ("ETKF", "EnKF") and N > bench.enkf_max_count:
                continue
            grid.append((m, N))
    return grid


def make_filter(method, num_particles, dim, transition, observation, initial):
    name, param = parse_method(method)
    common = dict(num_particles=num_particles, dim_param=dim, transition=transition,
                  observation=observation, initial=initial)
    if name == "PF":
        return BootstrapPF(**common)
    if name == "MPF":
        return MPF(alpha=MPF_COEFFICIENTS[int(param)], **common)
    if name == "RMPF":
        return RMPF(n=int(param), **common)
    if name == "LW":
        return LiuWestPF(h=param, **common)
    if name == "RPF":
        return RegularizedPF(bandwidth_scale=param, **common)
    if name == "ETKF":
        return EnKF(inflation=param, **common)
    if name == "EnKF":
        return EnKF(inflation=param, perturbed_obs=True, **common)
    raise ValueError(f"unknown method {method!r}")
