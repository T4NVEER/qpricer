"""Black-Scholes closed-form prices and Greeks for European vanilla options."""

import math

_SQRT_2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


def d1_d2(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Black-Scholes d1/d2. Requires vol > 0 and maturity > 0."""
    sigma_sqrt_t = vol * math.sqrt(maturity)
    if sigma_sqrt_t <= 0.0:
        raise ValueError("d1/d2 undefined for vol * sqrt(maturity) == 0")
    d1 = (
        math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * maturity
    ) / sigma_sqrt_t
    return d1, d1 - sigma_sqrt_t


def call_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """European call price under Black-Scholes."""
    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return spot * math.exp(-dividend_yield * maturity) * norm_cdf(d1) - strike * math.exp(
        -rate * maturity
    ) * norm_cdf(d2)


def put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """European put price under Black-Scholes."""
    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return strike * math.exp(-rate * maturity) * norm_cdf(-d2) - spot * math.exp(
        -dividend_yield * maturity
    ) * norm_cdf(-d1)
