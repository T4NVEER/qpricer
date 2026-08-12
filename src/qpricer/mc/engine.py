"""Monte Carlo pricing engine for European options under GBM.

European payoffs depend only on the terminal spot, so paths are sampled with
a single exact lognormal step: no time discretization and no discretization
bias. All randomness flows through an explicit numpy Generator.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from qpricer.market import MarketData

_Z_95 = 1.959963984540054  # two-sided 95% quantile of the standard normal


@dataclass(frozen=True, slots=True)
class MCResult:
    """Monte Carlo estimate with its statistical error."""

    price: float
    std_error: float
    n_paths: int

    def confidence_interval(self, z: float = _Z_95) -> tuple[float, float]:
        half_width = z * self.std_error
        return self.price - half_width, self.price + half_width


def sample_terminal_spots(
    market: MarketData,
    maturity: float,
    n_paths: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw terminal spots S_T = S0 * exp((r - q - vol^2/2) T + vol sqrt(T) Z)."""
    drift = (market.rate - market.dividend_yield - 0.5 * market.vol**2) * maturity
    diffusion = market.vol * math.sqrt(maturity)
    z = rng.standard_normal(n_paths)
    return market.spot * np.exp(drift + diffusion * z)
