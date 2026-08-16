"""Numba-accelerated Monte Carlo kernels.

Numba is an optional dependency ([perf] extra). Import this module and check
NUMBA_AVAILABLE before requesting a numba backend; everything else in qpricer
works without a compiler.

The kernels consume pre-drawn standard normals so the random stream is shared
with the NumPy path. Their advantage is fusing exp, payoff and accumulation
into one allocation-free pass; fastmath relaxes summation order, so results
match NumPy statistically (~1e-12 relative), not bit-for-bit.
"""

import math

import numpy as np
from numpy.typing import NDArray

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    NUMBA_AVAILABLE = False


def _require_numba() -> None:
    if not NUMBA_AVAILABLE:
        raise ImportError("numba is not installed; install qpricer[perf]")


def payoff_sums_serial(
    z: NDArray[np.float64],
    spot: float,
    strike: float,
    sign: float,
    drift: float,
    diffusion: float,
    discount: float,
) -> tuple[float, float]:
    """Sum and sum-of-squares of discounted payoffs; sign=+1 call, -1 put."""
    _require_numba()
    return _payoff_sums_serial(z, spot, strike, sign, drift, diffusion, discount)


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _payoff_sums_serial(
        z: NDArray[np.float64],
        spot: float,
        strike: float,
        sign: float,
        drift: float,
        diffusion: float,
        discount: float,
    ) -> tuple[float, float]:
        total = 0.0
        total_sq = 0.0
        for i in range(len(z)):
            s_t = spot * math.exp(drift + diffusion * z[i])
            payoff = sign * (s_t - strike)
            if payoff < 0.0:
                payoff = 0.0
            value = discount * payoff
            total += value
            total_sq += value * value
        return total, total_sq
