"""Variance reduction techniques for the Monte Carlo engine.

Both estimators remain unbiased; they reshape the sampling so the same number
of paths yields a smaller standard error.
"""

import math

import numpy as np

from qpricer._validation import require_positive
from qpricer.analytic.black_scholes import deterministic_price
from qpricer.instruments import EuropeanOption
from qpricer.market import MarketData
from qpricer.mc.engine import MCResult


def mc_price_antithetic(
    option: EuropeanOption,
    market: MarketData,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Antithetic variates: simulate pairs (Z, -Z) and average within each pair.

    The payoff is monotone in Z, so paired draws are negatively correlated and
    the variance of the pair average drops below that of two independent draws.
    n_paths counts total paths and must be even.
    """
    require_positive("n_paths", n_paths)
    if n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic pairs, got {n_paths}")
    if option.maturity == 0.0 or market.vol == 0.0:
        return MCResult(price=deterministic_price(option, market), std_error=0.0, n_paths=n_paths)

    n_pairs = n_paths // 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_pairs)

    drift = (market.rate - market.dividend_yield - 0.5 * market.vol**2) * option.maturity
    diffusion = market.vol * math.sqrt(option.maturity)
    discount = math.exp(-market.rate * option.maturity)

    payoff_up = option.payoff(market.spot * np.exp(drift + diffusion * z))
    payoff_down = option.payoff(market.spot * np.exp(drift - diffusion * z))
    pair_means = discount * 0.5 * (payoff_up + payoff_down)

    price = float(pair_means.mean())
    std_error = float(pair_means.std(ddof=1) / math.sqrt(n_pairs))
    return MCResult(price=price, std_error=std_error, n_paths=n_paths)
