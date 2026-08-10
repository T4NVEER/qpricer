"""Monte Carlo pricing engine for European options under GBM.

European payoffs depend only on the terminal spot, so paths are sampled with
a single exact lognormal step: no time discretization and no discretization
bias. All randomness flows through an explicit numpy Generator.
"""

import math

import numpy as np
from numpy.typing import NDArray

from qpricer.market import MarketData


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
