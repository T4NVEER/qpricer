"""Monte Carlo pricing engine for European options under GBM.

European payoffs depend only on the terminal spot, so paths are sampled with
a single exact lognormal step: no time discretization and no discretization
bias. All randomness flows through an explicit numpy Generator.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qpricer._validation import require_positive
from qpricer.analytic.black_scholes import deterministic_price
from qpricer.instruments import EuropeanOption, OptionType
from qpricer.market import MarketData

_Z_95 = 1.959963984540054  # two-sided 95% quantile of the standard normal

Backend = Literal["numpy", "numba", "numba_parallel"]


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


def mc_price(
    option: EuropeanOption,
    market: MarketData,
    n_paths: int,
    seed: int | None = None,
    batch_size: int | None = None,
    backend: Backend = "numpy",
) -> MCResult:
    """Price a European option by Monte Carlo.

    The estimator is the discounted sample mean of the terminal payoff; the
    reported standard error is the sample standard deviation / sqrt(n_paths).

    batch_size caps the number of paths simulated at once, keeping memory flat
    for large n_paths. The generator stream is consumed sequentially, so the
    result is bit-identical for any batch size given the same seed.

    backend selects the payoff accumulation: "numpy" (default), or "numba" /
    "numba_parallel" fused kernels (requires the [perf] extra). All backends
    share the same normal draws; numba results differ only by summation order.
    """
    require_positive("n_paths", n_paths)
    if batch_size is not None:
        require_positive("batch_size", batch_size)
    if option.maturity == 0.0 or market.vol == 0.0:
        return MCResult(price=deterministic_price(option, market), std_error=0.0, n_paths=n_paths)

    if backend != "numpy":
        from qpricer.mc import numba_kernels

        kernel = (
            numba_kernels.payoff_sums_serial
            if backend == "numba"
            else numba_kernels.payoff_sums_parallel
        )

    rng = np.random.default_rng(seed)
    discount = math.exp(-market.rate * option.maturity)
    drift = (market.rate - market.dividend_yield - 0.5 * market.vol**2) * option.maturity
    diffusion = market.vol * math.sqrt(option.maturity)
    sign = 1.0 if option.option_type is OptionType.CALL else -1.0
    batch = n_paths if batch_size is None else batch_size

    total = 0.0
    total_sq = 0.0
    remaining = n_paths
    while remaining > 0:
        n = min(batch, remaining)
        if backend == "numpy":
            spots = sample_terminal_spots(market, option.maturity, n, rng)
            discounted = discount * option.payoff(spots)
            total += float(discounted.sum())
            total_sq += float((discounted * discounted).sum())
        else:
            z = rng.standard_normal(n)
            batch_total, batch_sq = kernel(
                z, market.spot, option.strike, sign, drift, diffusion, discount
            )
            total += batch_total
            total_sq += batch_sq
        remaining -= n

    mean = total / n_paths
    variance = (total_sq - n_paths * mean * mean) / (n_paths - 1) if n_paths > 1 else 0.0
    std_error = math.sqrt(max(variance, 0.0) / n_paths)
    return MCResult(price=mean, std_error=std_error, n_paths=n_paths)
