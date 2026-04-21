from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit


@dataclass(slots=True)
class WarmupResult:
    warmup_period: (
        int  # Index of the last sample before convergence (i.e. warmup length)
    )
    mu_ref: float  # Mean value used as the reference in the tail window
    sigma_ref: float  # Standard deviation of individual chains in the tail window
    is_converged: bool  # Whether the chain has converged (True if warmup is in the first (1-tail_fraction)% of samples)


@njit(cache=True)
def _smooth_and_find_last_exit(
    center: npt.NDArray[np.float64], window: int, mu_ref: float, sigma_th: float
) -> int:
    """
    Scans the sequence from the end, using a moving window to find the last point
    where the mean deviates more than sigma_ref from mu_ref.
    Returns the first sample (from start) after which the chain remains inside
    the convergence region.
    """
    num_steps = len(center)
    n_windows = num_steps - window + 1

    # Compute cumulative sum to efficiently calculate windowed mean
    cumsum = np.zeros(num_steps + 1, dtype=np.float64)
    cumsum[1:] = np.cumsum(center)

    # Scan backwards from the end of the sequence
    for t in range(n_windows - 1, -1, -1):
        window_mean = (cumsum[t + window] - cumsum[t]) / window
        # Check if the window mean deviates from the reference mean by more than the sigma threshold
        if abs(window_mean - mu_ref) > sigma_th:
            # Return the end of the window where the last deviation occurred
            return min(t + window, num_steps)

    # If never left the convergence region, warmup was over before the first sample
    return 0


def detect_warmup(
    data: npt.NDArray[np.float32],
    sigma_coef: float = 2.0,
    tail_fraction: float = 0.2,
    smooth_fraction: float = 0.02,
) -> WarmupResult:
    """
    Detects the warmup period of MCMC chains using a cross-chain mean deviation test.

    Warmup is defined as the last time (by index) at which the cross-chain mean
    leaves the region |mean - mu_ref| > sigma_ref, where mu_ref and sigma_ref
    are computed from the tail (last tail_fraction of samples).

    After this time, transient bias is assumed smaller than the individual chain noise.

    Args:
        data : [num_samples, num_steps]
        sigma_coef : Multiplier for the reference standard deviation
        tail_fraction : Fraction of the trajectory (from the end) to use as the reference "tail"
        smooth_fraction : Fractional window size for the running mean smoothing (default 0.02).
    """

    num_steps = data.shape[1]
    center = data.mean(axis=0, dtype=np.float64)  # [num_steps, ]

    tail_start = int(num_steps * (1.0 - tail_fraction))
    mu_ref = float(np.mean(center[tail_start:], dtype=np.float64))
    sigma_ref = float(np.std(data[:, tail_start:], axis=1, dtype=np.float64).mean())

    window = int(num_steps * smooth_fraction)
    warmup_period = _smooth_and_find_last_exit(
        center, window, mu_ref, sigma_coef * sigma_ref
    )

    return WarmupResult(
        warmup_period=warmup_period,
        mu_ref=mu_ref,
        sigma_ref=sigma_ref,
        is_converged=warmup_period < tail_start,
    )
