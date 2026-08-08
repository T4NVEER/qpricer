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


def call_delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    d1, _ = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return math.exp(-dividend_yield * maturity) * norm_cdf(d1)


def put_delta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    d1, _ = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return math.exp(-dividend_yield * maturity) * (norm_cdf(d1) - 1.0)


def gamma(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Second derivative w.r.t. spot; identical for calls and puts."""
    d1, _ = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return math.exp(-dividend_yield * maturity) * norm_pdf(d1) / (spot * vol * math.sqrt(maturity))


def vega(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Sensitivity to vol (per unit of vol, not per 1%); identical for calls and puts."""
    d1, _ = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return spot * math.exp(-dividend_yield * maturity) * norm_pdf(d1) * math.sqrt(maturity)


def call_theta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Time decay per year (calendar convention: dV/dt, typically negative)."""
    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    df_q = math.exp(-dividend_yield * maturity)
    df_r = math.exp(-rate * maturity)
    diffusion = -spot * df_q * norm_pdf(d1) * vol / (2.0 * math.sqrt(maturity))
    return (
        diffusion
        - rate * strike * df_r * norm_cdf(d2)
        + dividend_yield * spot * df_q * norm_cdf(d1)
    )


def put_theta(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    df_q = math.exp(-dividend_yield * maturity)
    df_r = math.exp(-rate * maturity)
    diffusion = -spot * df_q * norm_pdf(d1) * vol / (2.0 * math.sqrt(maturity))
    return (
        diffusion
        + rate * strike * df_r * norm_cdf(-d2)
        - dividend_yield * spot * df_q * norm_cdf(-d1)
    )


def call_rho(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    """Sensitivity to the risk-free rate (per unit of rate)."""
    _, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return strike * maturity * math.exp(-rate * maturity) * norm_cdf(d2)


def put_rho(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    dividend_yield: float = 0.0,
) -> float:
    _, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend_yield)
    return -strike * maturity * math.exp(-rate * maturity) * norm_cdf(-d2)
