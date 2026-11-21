# RMPF: randomized merging particle filter

Code, data and results of the numerical experiments of

> **Improved Method for Merging Particle Filter Using Random Merging Coefficients**
> Naoki Hiramoto, Genta Ueno, Daisuke Murakami, *Nonlinear Processes in Geophysics* (egusphere-2026-4073)

Every table and figure of the paper can be regenerated from the CSV files in `results/` without running
any filter (see [Tables and figures](#tables-and-figures)); the commands that produced these CSV files
are listed in [Rerunning the experiments](#rerunning-the-experiments).

## Layout

```
rmpf/                   package
  rng.py                reproducible random numbers under numba parallelism
  models.py             Lorenz-63 / Lorenz-96 (forward Euler), Gaussian observation models (y = x + e,
                        y = |x| + e), truth trajectories
  filters.py            BootstrapPF, MPF, RMPF, RegularizedPF, LiuWestPF, EnKF (ETKF / perturbed obs.)
  benchmarks.py         benchmark settings, method grid, filter factory
  experiment.py         one filtering run and its summary (definitions of RMSE and timing)
  results.py            aggregation over seeds
data/<benchmark>/       truth and observations (generate_data.py; --check compares bitwise)
results/<benchmark>/    CSV outputs of the experiments of the paper (one row per run)
generate_data.py        data/
run_experiments.py      comparison of all methods (Sect. 5, Appendix C1-C3)
select_n.py             selection of the merging size n by PRS with validation on a held-out period (Sect. 6)
make_tables.py          LaTeX bodies of Tables 3, 4, B1, C1, C3 -> tables/
plot_rmse_vs_time.py    Fig. 2 and Fig. C1: RMSE against computation time per assimilation step
plot_select_n.py        Fig. 3: logZ(n) and held-out RMSE(n), grid and PRS selections
plot_rmse_vs_param.py   Fig. C2 and Table C2: RMSE of every run against the parameter (Lorenz-63)
theory/check_merging_theory.py   numerical checks of Propositions 1-3 and of the duplication statistics
theory/plot_merging_density.py   Fig. 1: law of a merged particle for d = 2
theory/duplication_stats.py      Table B1: duplicated particles in the merging groups during filtering
theory/outlier_merging.py        Appendix C4: one merging step of an ensemble with outlying particles
theory/outlier_table.py          Table C4
figures/, tables/       outputs of the plotting and table scripts
```

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions of the environment of superdome1 (Python 3.9.13, NumPy 1.21.4,
Numba 0.55.1, pandas 1.3.5). The experiments were run on two machines (see the table below). The numerical
results do not depend on the number of numba threads (`rmpf/rng.py`), but they are not bitwise identical
between software builds: the superdome1 runs used the Intel Distribution for Python (numpy linked to MKL,
Intel math libraries for numba), and a pip installation of the same versions reproduces the RMSEs of
`results/` only to within the run-to-run variability (in a test with 28 configurations of Lorenz-63 and
Lorenz-96, 2 were bitwise identical and the others differed by amounts typical of a change of seed; the
systems are chaotic). The computation times depend on the machine. The tables, and the CSV files written
next to the figures, are regenerated identically from `results/` with the pinned versions and with the
workstation environment; the figures agree up to anti-aliasing.

| machine | used for | software |
|---|---|---|
| superdome1 (Intel Xeon Platinum 8280L, 2.7 GHz; one socket = 28 cores per job) of the Supercomputer System for Data Assimilation, ISM | Sect. 5 (Tables 3, 4, Fig. 2), Appendix C1-C3 (computation times are reported) | Intel Distribution for Python 2022.3.1: Python 3.9.13, NumPy 1.21.4, Numba 0.55.1, pandas 1.3.5 |
| workstation (24 cores) | Sect. 6 (Fig. 3), Table B1, Appendix C4 (no computation times are reported, so the speed of the machine does not affect these results) | Python 3.11, NumPy 2.2.6, Numba 0.64.0, pandas 2.3.1, matplotlib 3.10.5 (also used for all figures) |

BLAS/LAPACK (used by RPF, LW, ETKF and EnKF) must be single-threaded (`OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`): a threaded BLAS changes the order of floating-point reductions, so that the results
of these methods would depend on the thread count.

## Tables and figures

All commands are run from the repository root (`export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`).

| paper | results | command |
|---|---|---|
| Fig. 1 | – | `python theory/plot_merging_density.py` |
| Fig. 2 | `results/lorenz63/experiments/`, `results/lorenz96/experiments/` | `python plot_rmse_vs_time.py --paper --exclude ETKF EnKF --output paper_rmse_vs_time` |
| Table 3 | same | `python make_tables.py` (`tables/table3_body.tex`) |
| Table 4 | `results/lorenz96/experiments/` | `python make_tables.py` (`tables/table_l96_time_body.tex`) |
| Table 5 | `results/lorenz96_d80/experiments/` | `python make_tables.py` (`tables/table_d80_body.tex`) |
| Fig. 3 | `results/lorenz96{,_d10,_d20,_d30}/select_n_dims/` | `python plot_select_n.py --l96-dims 10 20 30 40 --paper --output paper_select_n_dims` |
| Table B1 | `results/theory/` | `python make_tables.py` (`tables/table_duplicates_body.tex`) |
| Table C1 | `results/lorenz96_abs/experiments/` | `python make_tables.py` (`tables/tableC1_abs_body.tex`) |
| Fig. C1 | same | `python plot_rmse_vs_time.py --benchmarks l96abs --paper --exclude ETKF EnKF --output paper_l96abs_rmse_vs_time` |
| Fig. C2, Table C2 | `results/lorenz63/experiments/` | `python plot_rmse_vs_param.py --paper` (`tables/tableC2_median_body.tex`) |
| Table C3 | `results/lorenz{63,96}/experiments_smalln/` | `python make_tables.py` (`tables/tableC3_smalln_body.tex`) |
| Table C4 | `results/outlier/` | `python theory/outlier_table.py` (`tables/tableC4_outlier_body.tex`) |

The numbers quoted in the text are those of these tables and of the CSV files written next to the figures
(`figures/*_best.csv`, `figures/paper_select_n_dims.csv`, `figures/l63_rmse_vs_param_median.csv`) or of
`results/` directly. `theory/check_merging_theory.py` checks the propositions numerically (a few seconds).

## Rerunning the experiments

A rerun of a command skips the (method, N, seed) runs already present in its output directory, so an
interrupted run resumes; move the shipped CSV files away first to recompute them. Runs of another
machine must not be mixed into one directory, because the computation times would not be comparable.

| results | command | runs | paper |
|---|---|---|---|
| `lorenz63/experiments/` | `python run_experiments.py l63 --full` | 5190 | superdome1, 23 h (seeds 42-44: 7 h) |
| `lorenz96/experiments/` | `python run_experiments.py l96 --full` | 5390 | superdome1, 60 h (seeds 42-44: 18 h) |
| `lorenz96_d80/experiments/` | `python run_experiments.py l96d80 --full --seeds 42 --max-steps 1000 --methods M` for M = `PF`, `RMPF:100`, `RPF:1`, `LW:0.9` (one job each) | 4 × 4 | superdome1, 5-9 h each |
| `lorenz96_abs/experiments/` | `python run_experiments.py l96abs --full` | 5390 | superdome1, 60 h (seeds 42-44: 18 h) |
| `lorenz63/experiments_smalln/` | `python run_experiments.py l63 --counts 3 5 10 20 --subdir experiments_smalln` | 1650 | superdome1, 2 h |
| `lorenz96/experiments_smalln/` | `python run_experiments.py l96 --counts 3 5 10 20 40 --subdir experiments_smalln` | 2000 | superdome1, 1.7 h |
| `lorenz96{,_d<d>}/select_n_dims/` | for d = 10, 20, 30, 40: `python select_n.py l96 --dim d --budgets 32768 --grid 2 3 5 10 20 30 40 50 60 70 80 90 100 110 120 --eval-seeds 42 43 44 45 46 47 48 49 50 51 --seed 1 2 3 4 5 6 7 8 9 10` | 10 searches + 170-250 evaluation runs per d | workstation, 12 threads, 6-15 h per d |
| `theory/` | `python theory/duplication_stats.py` | | workstation, about 10 min |
| `outlier/` | `python theory/outlier_merging.py` | | workstation, about 5 min |

`results/outlier/outlier_*.csv.gz` (one row per repetition, method and scenario) is compressed; Table C4
uses only `summary_*.csv`. `python generate_data.py --l96-dims 10 20 30` writes all data files of `data/`
(`--check` regenerates them in memory and compares them bitwise with the shipped files).

## Methods

Every filter assimilates at every observation time and uses **systematic resampling**. The state
estimate of every particle filter is the importance-weighted mean **before** resampling (and before
merging or jittering); for the EnKF it is the analysis mean.

| name | method | parameter grid |
|---|---|---|
| `PF` | bootstrap PF | – |
| `MPF:k` | MPF with fixed coefficients (Nakano et al. 2007): n = 3 parents from 3 independent systematic resamplings, each randomly permuted before grouping | k = 1: α = (−0.138, 0.161, 0.977); 2: (0.75, 0.576, −0.326) (Nakano et al.); 3: (2/3, 2/3, −1/3) |
| `RMPF:n` | one systematic resampling, random permutation, groups of n parents; each group is replaced by Ωᵀ·parents with Ω a random orthogonal matrix with Ω1 = 1 (Sakov and Oke 2008). Each child has coefficients uniform on {Σα = 1, Σα² = 1}; the group's sample mean and covariance are preserved | L63: n = 3–10, 20, 40, 100; L96: n = 3, 5, 10, 20, 30, 35, 37–45, 80, 100 (n ≤ N) |
| `RPF:c` | post-regularized PF (Musso et al. 2001), Gaussian kernel, whitening by the Cholesky factor of the weighted covariance, h = c·h_opt | c = 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3 |
| `LW:h` | Liu and West (2001) kernel shrinkage (mean and covariance preserved) | h = 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 |
| `ETKF:λ` | square-root EnKF (symmetric, mean-preserving), no localization | λ = 1, 1.01, 1.02, 1.05, 1.1, 1.2, 1.5 (multiplicative inflation of forecast anomalies) |
| `EnKF:λ` | perturbed-observation EnKF (perturbations centered over members), no localization | same λ; run up to N = 8192 |

Random numbers that are drawn once and reused at every step: the RMPF matrices Ω and the RPF/LW
jitter ε (`regenerate=True` in `rmpf/filters.py` draws them afresh at every step instead).

## Benchmark settings

| | Lorenz-63 | Lorenz-96 |
|---|---|---|
| state dimension d | 3 (σ = 10, ρ = 28, β = 8/3) | 40 (F = 8) |
| integration | forward Euler, dt = 0.01 | forward Euler, dt = 0.005 |
| truth | noise-free, x₀ = (1, 1, 1), 1000 warm-up steps | noise-free, x₀ = 8 with x₂₀ = 8.008, 2000 warm-up steps |
| observation interval | 20 steps (Δt = 0.2) | 10 steps (Δt = 0.05) |
| observed components | all | even components (20) |
| observation error s.d. (data and likelihood) | 1.0 | 1.5 |
| observation times | 2500 | 2000 |
| process noise of the filters (additive, every model step) | N(0, 0.01² I) | N(0, 0.5² I) |
| initial ensemble | N(y₀, 2² I) | N(2, 2 I) |
| warm-up excluded from averages | 100 assimilation steps | 300 assimilation steps |
| particle numbers N (`--full`) | 32 … 65536 (powers of 2) | 64 … 65536 (powers of 2) |
| seeds | 42–51 (d = 80: 42) | 42–51 (d = 80: 42) |

## Definitions

* RMSE at time t: sqrt(mean over components of (estimate − truth)²), over all components (also the
  unobserved ones in Lorenz-96). The reported **rmse** of a run is the mean of this over the
  assimilation steps after the warm-up (`rmse_median`: the median).
* The climatological standard deviation of the truth (8.36 for Lorenz-63, 3.75 for Lorenz-96) is given
  as a reference: an RMSE close to or above it means that the filter does not track the truth. (The CSV
  files also contain a column `diverged`, rmse above this value or non-finite, which the tables and
  figures of the paper do not use.)
* Over seeds: mean, standard deviation and median. A configuration in which any seed became non-finite
  has mean +∞ (n.f. in the tables). The "best" parameter at a given N is the one with the smallest mean.
* Computation time: wall-clock time of one assimilation step (obs_interval forecasts + filtering),
  median over the steps after the warm-up.
* log-likelihood increment: log((1/N) Σᵢ p(y_t | x_t⁽ⁱ⁾)) of the forecast particles; its sum is the
  logZ used for the selection of n.

## Selection of n (`select_n.py`)

With the noise variances known and the particle budget N fixed, n is chosen by maximizing logZ over
the first half of the series with PRS (screening of M = 9 candidates under common random numbers with
successive halving, and a zoom on log2 n). The selected n and a grid of n are then run with N particles
and the evaluation seeds over the whole series, and the RMSE over the second half, which the search did
not use, is reported.

## License and citation

The code is licensed under the MIT License (`LICENSE`). The contents of `data/`, `results/`, `tables/`
and `figures/` are licensed under CC BY 4.0. If you use this repository, please cite the paper and the
archived version of this repository (`CITATION.cff`).
