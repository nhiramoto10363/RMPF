"""
One filtering run and its summary.

An assimilation step is obs_interval forecast steps followed by filtering (none before the first
observation). Per step we record the RMSE of the estimate, the spread of the updated ensemble, the
ESS of the importance weights, the log-likelihood increment, and the wall-clock time of the forecast
and of the filtering. Summaries exclude the first warmup_steps steps.

Definitions used in every table and figure:
  RMSE_t   = sqrt(mean_j (estimate_tj - truth_tj)^2) at each observation time t
  rmse     = mean of RMSE_t over t >= warmup (median: rmse_median)
  diverged = rmse is non-finite or exceeds the climatological standard deviation of the truth,
             sqrt(mean_j var_t(truth_tj))
"""
import os
import socket
import time

import numba
import numpy as np

from .benchmarks import make_filter, parse_method
from .rng import set_numba_seed


def run_filter(bench, method, num_particles, seed, obs, truth, progress=False):
    """Run one filter over the whole series; returns per-step arrays. All randomness follows from seed."""
    np.random.seed(seed)
    set_numba_seed(seed)
    transition, observation, initial = bench.build_models(obs, num_particles)
    pf = make_filter(method, num_particles, bench.dim, transition, observation, initial.copy())

    T = len(obs)
    out = {k: np.empty(T) for k in ("rmse", "spread", "ess", "log_lik", "forecast_time", "filter_time")}
    steps = range(T)
    if progress:
        from tqdm import tqdm
        steps = tqdm(steps, desc=f"{method} N={num_particles} s={seed}", mininterval=10)
    for t in steps:
        forecast_time = 0.0
        if t > 0:
            start = time.perf_counter()
            for _ in range(bench.obs_interval):
                pf.forecast()
            forecast_time = time.perf_counter() - start
        start = time.perf_counter()
        pf.filtering(obs[t])
        out["filter_time"][t] = time.perf_counter() - start
        out["forecast_time"][t] = forecast_time

        w = pf.weights
        out["rmse"][t] = np.sqrt(np.mean((pf.estimate - truth[t]) ** 2))
        out["spread"][t] = np.sqrt(np.mean((pf.particles - pf.particles.mean(0)) ** 2))
        out["ess"][t] = 1.0 / np.sum(w**2)
        out["log_lik"][t] = pf.log_lik_increment
    return out


def climatological_std(truth):
    return float(np.sqrt(np.mean(np.var(truth, axis=0))))


def summarize(bench, method, num_particles, seed, out, truth):
    k = min(bench.warmup_steps, len(truth))
    rmse = out["rmse"]
    clim = climatological_std(truth)
    mean_rmse = float(np.mean(rmse[k:]))
    step = out["forecast_time"] + out["filter_time"]
    nonfinite = ~np.isfinite(rmse)
    name, param = parse_method(method)
    return {
        "benchmark": bench.key, "method": name, "param": param, "num_particles": num_particles, "seed": seed,
        "num_steps": len(truth), "warmup_steps": k,
        "rmse": mean_rmse, "rmse_median": float(np.median(rmse[k:])),
        "spread": float(np.mean(out["spread"][k:])), "ess": float(np.mean(out["ess"][k:])),
        "log_lik": float(np.sum(out["log_lik"])),
        "climatological_std": clim,
        "diverged": bool((not np.isfinite(mean_rmse)) or mean_rmse > clim),
        "first_nonfinite_step": int(np.argmax(nonfinite)) if nonfinite.any() else -1,
        "median_step_time": float(np.median(step[k:])),
        "median_forecast_time": float(np.median(out["forecast_time"][k:])),
        "median_filter_time": float(np.median(out["filter_time"][k:])),
        "total_time": float(np.sum(step)),
        "host": socket.gethostname(), "numba_threads": numba.get_num_threads(),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "default")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
