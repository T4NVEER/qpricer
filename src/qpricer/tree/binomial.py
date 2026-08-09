"""Cox-Ross-Rubinstein binomial tree for European options."""

import math

import numpy as np

from qpricer._validation import require_positive
from qpricer.instruments import EuropeanOption, OptionType
from qpricer.market import MarketData


def _crr_params(market: MarketData, maturity: float, steps: int) -> tuple[float, float, float]:
    """Up factor, risk-neutral up probability and per-step discount factor."""
    dt = maturity / steps
    u = math.exp(market.vol * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((market.rate - market.dividend_yield) * dt)
    p = (growth - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0, 1); increase steps")
    return u, p, math.exp(-market.rate * dt)


def crr_price_loop(option: EuropeanOption, market: MarketData, steps: int) -> float:
    """Reference CRR pricer with explicit Python loops.

    Kept as the ground truth for the vectorized implementation and as a
    baseline for benchmarks; O(steps^2) node updates.
    """
    require_positive("steps", steps)
    u, p, disc = _crr_params(market, option.maturity, steps)

    # Terminal option values at nodes j = 0..steps (j up-moves).
    values = []
    for j in range(steps + 1):
        spot_t = market.spot * u ** (2 * j - steps)
        if option.option_type is OptionType.CALL:
            values.append(max(spot_t - option.strike, 0.0))
        else:
            values.append(max(option.strike - spot_t, 0.0))

    for step in range(steps, 0, -1):
        values = [disc * (p * values[j + 1] + (1.0 - p) * values[j]) for j in range(step)]
    return values[0]


def crr_price(option: EuropeanOption, market: MarketData, steps: int) -> float:
    """CRR pricer with vectorized backward induction.

    Same lattice as crr_price_loop; each induction step collapses the value
    vector with one fused NumPy expression instead of a Python loop.
    """
    require_positive("steps", steps)
    u, p, disc = _crr_params(market, option.maturity, steps)

    exponents = np.arange(-steps, steps + 1, 2, dtype=np.float64)
    terminal_spots = market.spot * np.exp(np.log(u) * exponents)
    values = option.payoff(terminal_spots)

    for _ in range(steps):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
    return float(values[0])
