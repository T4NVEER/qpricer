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
from qpricer.mc.engine import MCResult, sample_terminal_spots


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


def mc_price_control_variate(
    option: EuropeanOption,
    market: MarketData,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Control variate: the discounted terminal spot, whose mean is known.

    E[e^{-rT} S_T] = S0 e^{-qT} under the risk-neutral measure. The estimator
    subtracts beta * (control sample mean - known mean), with beta chosen to
    minimize variance (regression coefficient of payoff on control). Highly
    effective for ITM options, where payoff and spot are strongly correlated.
    """
    require_positive("n_paths", n_paths)
    if option.maturity == 0.0 or market.vol == 0.0:
        return MCResult(price=deterministic_price(option, market), std_error=0.0, n_paths=n_paths)
    if n_paths < 2:
        raise ValueError("control variate needs at least 2 paths to estimate beta")

    rng = np.random.default_rng(seed)
    discount = math.exp(-market.rate * option.maturity)
    spots = sample_terminal_spots(market, option.maturity, n_paths, rng)

    payoffs = discount * option.payoff(spots)
    control = discount * spots
    control_mean_exact = market.spot * math.exp(-market.dividend_yield * option.maturity)

    cov = np.cov(payoffs, control, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])

    adjusted = payoffs - beta * (control - control_mean_exact)
    price = float(adjusted.mean())
    std_error = float(adjusted.std(ddof=1) / math.sqrt(n_paths))
    return MCResult(price=price, std_error=std_error, n_paths=n_paths)


def mc_price_antithetic_cv(
    option: EuropeanOption,
    market: MarketData,
    n_paths: int,
    seed: int | None = None,
) -> MCResult:
    """Antithetic pairs combined with the terminal-spot control variate.

    Both the payoff and the control are averaged within each antithetic pair;
    the control adjustment is then applied to the pair means.
    """
    require_positive("n_paths", n_paths)
    if n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even for antithetic pairs, got {n_paths}")
    if option.maturity == 0.0 or market.vol == 0.0:
        return MCResult(price=deterministic_price(option, market), std_error=0.0, n_paths=n_paths)
    if n_paths < 4:
        raise ValueError("need at least 2 antithetic pairs to estimate beta")

    n_pairs = n_paths // 2
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_pairs)

    drift = (market.rate - market.dividend_yield - 0.5 * market.vol**2) * option.maturity
    diffusion = market.vol * math.sqrt(option.maturity)
    discount = math.exp(-market.rate * option.maturity)

    spots_up = market.spot * np.exp(drift + diffusion * z)
    spots_down = market.spot * np.exp(drift - diffusion * z)

    pair_payoffs = discount * 0.5 * (option.payoff(spots_up) + option.payoff(spots_down))
    pair_controls = discount * 0.5 * (spots_up + spots_down)
    control_mean_exact = market.spot * math.exp(-market.dividend_yield * option.maturity)

    cov = np.cov(pair_payoffs, pair_controls, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1])

    adjusted = pair_payoffs - beta * (pair_controls - control_mean_exact)
    price = float(adjusted.mean())
    std_error = float(adjusted.std(ddof=1) / math.sqrt(n_pairs))
    return MCResult(price=price, std_error=std_error, n_paths=n_paths)
