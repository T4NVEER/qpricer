"""Put-call parity: C - P = S * exp(-qT) - K * exp(-rT)."""

import math

from hypothesis import given
from hypothesis import strategies as st

from qpricer.analytic.black_scholes import call_price, put_price

reasonable = {
    "spot": st.floats(min_value=1.0, max_value=1e4),
    "strike": st.floats(min_value=1.0, max_value=1e4),
    "maturity": st.floats(min_value=0.01, max_value=30.0),
    "rate": st.floats(min_value=-0.05, max_value=0.15),
    "vol": st.floats(min_value=0.01, max_value=2.0),
    "dividend_yield": st.floats(min_value=0.0, max_value=0.10),
}


@given(**reasonable)
def test_put_call_parity(
    spot: float, strike: float, maturity: float, rate: float, vol: float, dividend_yield: float
) -> None:
    c = call_price(spot, strike, maturity, rate, vol, dividend_yield)
    p = put_price(spot, strike, maturity, rate, vol, dividend_yield)
    lhs = c - p
    rhs = spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)
    assert math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-9 * max(spot, strike))
