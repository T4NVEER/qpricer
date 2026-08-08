"""Implied volatility inversion via Newton's method with bisection fallback."""

import math
from collections.abc import Callable

from qpricer.analytic.black_scholes import call_price, put_price, vega
from qpricer.instruments import OptionType

_TOL = 1e-10
_VOL_TOL = 1e-12
_MAX_NEWTON_ITER = 50
_MAX_BISECT_ITER = 200
_VOL_LO, _VOL_HI = 1e-9, 5.0


def _no_arbitrage_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float,
) -> tuple[float, float]:
    fwd_leg = spot * math.exp(-dividend_yield * maturity)
    strike_leg = strike * math.exp(-rate * maturity)
    if option_type is OptionType.CALL:
        return max(fwd_leg - strike_leg, 0.0), fwd_leg
    return max(strike_leg - fwd_leg, 0.0), strike_leg


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

    Newton's method (quadratic convergence, vega as derivative) with the
    Brenner-Subrahmanyam approximation as initial guess; falls back to
    bisection when vega is too flat for a reliable Newton step (deep ITM/OTM).
    """
    lower, upper = _no_arbitrage_bounds(spot, strike, maturity, rate, option_type, dividend_yield)
    if not lower < target_price < upper:
        raise ValueError(
            f"price {target_price} violates no-arbitrage bounds ({lower:.6g}, {upper:.6g})"
        )

    price_fn = call_price if option_type is OptionType.CALL else put_price

    def objective(vol: float) -> float:
        return price_fn(spot, strike, maturity, rate, vol, dividend_yield) - target_price

    # Brenner-Subrahmanyam ATM approximation as starting point.
    vol = max(math.sqrt(2.0 * math.pi / maturity) * target_price / spot, 1e-4)

    for _ in range(_MAX_NEWTON_ITER):
        diff = objective(vol)
        if abs(diff) < _TOL:
            return vol
        v = vega(spot, strike, maturity, rate, vol, dividend_yield)
        if v < 1e-12:
            break  # flat vega: fall back to bisection
        vol -= diff / v
        if not _VOL_LO < vol < _VOL_HI:
            break
    return _bisect(objective)


def _bisect(objective: Callable[[float], float]) -> float:
    """Bisection on [lo, hi]; the BS price is strictly increasing in vol.

    Converges on the vol interval width rather than the price residual: deep
    OTM prices can sit below any absolute price tolerance across a wide vol
    range, which would otherwise trigger a premature exit.
    """
    lo, hi = _VOL_LO, _VOL_HI
    if objective(lo) > 0.0 or objective(hi) < 0.0:
        raise RuntimeError("implied vol not bracketed by [1e-9, 5.0]")
    for _ in range(_MAX_BISECT_ITER):
        if hi - lo < _VOL_TOL:
            break
        mid = 0.5 * (lo + hi)
        if objective(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
