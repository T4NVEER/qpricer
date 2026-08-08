"""Implied volatility inversion via Newton's method."""

import math

from qpricer.analytic.black_scholes import call_price, put_price, vega
from qpricer.instruments import OptionType

_TOL = 1e-10
_MAX_NEWTON_ITER = 50


def implied_vol(
    target_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Invert Black-Scholes for volatility.

    Uses Newton's method with the Brenner-Subrahmanyam approximation as the
    initial guess. Vega-based Newton converges quadratically near the root.
    """
    price_fn = call_price if option_type is OptionType.CALL else put_price

    # Brenner-Subrahmanyam ATM approximation as starting point.
    vol = max(math.sqrt(2.0 * math.pi / maturity) * target_price / spot, 1e-4)

    for _ in range(_MAX_NEWTON_ITER):
        diff = price_fn(spot, strike, maturity, rate, vol, dividend_yield) - target_price
        if abs(diff) < _TOL:
            return vol
        v = vega(spot, strike, maturity, rate, vol, dividend_yield)
        if v < 1e-12:
            break  # flat vega: Newton step unreliable
        vol -= diff / v
        if vol <= 0.0:
            break
    raise RuntimeError("Newton iteration failed to converge")
