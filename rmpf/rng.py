"""
Reproducible random numbers under numba parallelism.

Two independent RNG states are in play:
  * the Python-side ``np.random`` state (initial ensembles, per-run seeds), and
  * the numba JIT-side state used inside ``@njit`` functions (resampling, merging coefficients).
``np.random.seed`` only controls the former; ``set_numba_seed`` controls the latter.

Noise drawn inside ``prange`` loops would use per-thread RNG states and therefore depend on the
number of threads. ``generate_process_noise`` avoids this by splitting the particles into chunks
whose count depends only on N, seeding each chunk deterministically. Since the main thread takes
part in that loop, its JIT state is reset afterwards with ``reseed_main_thread``. With these two
rules the numerical results do not depend on the number of threads.
"""
import numpy as np
from numba import njit, prange


@njit
def set_numba_seed(seed):
    """Seed the JIT-side RNG (np.random inside @njit functions)."""
    np.random.seed(seed)


@njit(inline='always')
def _mix_seed(a, b):
    """Scrambled seed from two integers (splitmix64 finalizer), so adjacent chunks get uncorrelated streams."""
    x = np.uint64(a) * np.uint64(6364136223846793005) + np.uint64(b) * np.uint64(1442695040888963407)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xbf58476d1ce4e5b9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94d049bb133111eb)
    x ^= x >> np.uint64(31)
    return np.int64(x & np.uint64(0x7FFFFFFF))


def resolve_noise_chunks(num_particles, max_chunks=64, min_per_chunk=256):
    """
    Number of chunks for parallel noise generation, a function of N only (never of the thread count).

    max_chunks bounds the effective parallelism of generate_process_noise. Changing it changes the
    random numbers for N > max_chunks * min_per_chunk, so keep it at 64.
    """
    return max(1, min(max_chunks, num_particles // min_per_chunk))


@njit(parallel=True, fastmath=True)
def generate_process_noise(num_particles, dim, std, seed_base, step, n_chunks):
    """N(0, std^2) noise of shape (num_particles, dim), identical for any number of threads."""
    noise = np.empty((num_particles, dim))
    chunk = (num_particles + n_chunks - 1) // n_chunks

    for c in prange(n_chunks):
        np.random.seed(_mix_seed(seed_base + step * 1000003, c))
        start = c * chunk
        end = min(start + chunk, num_particles)
        if end > start:
            noise[start:end, :] = np.random.normal(0.0, std, (end - start, dim))

    return noise


@njit
def reseed_main_thread(seed_base, step):
    """
    Reset the main thread's JIT RNG state after generate_process_noise.

    Which chunk the main thread draws depends on the thread count, so without this the subsequent
    (serial) resampling would depend on it too.
    """
    np.random.seed(_mix_seed(seed_base + step * 7919, np.int64(0x5BF03635)))


def draw_noise(num_particles, dim, std, seed_base, step):
    """generate_process_noise followed by reseed_main_thread (the pattern used everywhere)."""
    noise = generate_process_noise(num_particles, dim, std, seed_base, step, resolve_noise_chunks(num_particles))
    reseed_main_thread(seed_base, step)
    return noise
