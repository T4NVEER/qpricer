import math

from qpricer.analytic.black_scholes import call_delta, gamma, put_delta, vega

PARAMS = {"spot": 100.0, "strike": 100.0, "maturity": 1.0, "rate": 0.05, "vol": 0.2}


def test_call_delta_bounds_and_atm() -> None:
    d = call_delta(**PARAMS)
    assert 0.0 < d < 1.0
    assert d > 0.5  # ATM forward > strike when r > 0


def test_put_delta_bounds() -> None:
    d = put_delta(**PARAMS)
    assert -1.0 < d < 0.0


def test_delta_parity() -> None:
    # call_delta - put_delta = exp(-qT)
    q = 0.02
    diff = call_delta(**PARAMS, dividend_yield=q) - put_delta(**PARAMS, dividend_yield=q)
    assert math.isclose(diff, math.exp(-q * PARAMS["maturity"]))


def test_deep_itm_call_delta_near_one() -> None:
    assert call_delta(spot=300.0, strike=100.0, maturity=0.5, rate=0.05, vol=0.2) > 0.999


def test_gamma_positive_and_peaks_near_atm() -> None:
    atm = gamma(**PARAMS)
    itm = gamma(**{**PARAMS, "spot": 60.0})
    otm = gamma(**{**PARAMS, "spot": 160.0})
    assert atm > 0.0
    assert atm > itm
    assert atm > otm


def test_vega_positive_and_scales_with_spot() -> None:
    v = vega(**PARAMS)
    assert v > 0.0
    doubled = vega(**{**PARAMS, "spot": 200.0, "strike": 200.0})
    assert math.isclose(doubled, 2.0 * v, rel_tol=1e-12)
