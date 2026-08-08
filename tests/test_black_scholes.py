import math

import pytest

from qpricer import EuropeanOption, MarketData, OptionType
from qpricer.analytic.black_scholes import call_price, d1_d2, norm_cdf, norm_pdf, price, put_price


def test_norm_cdf_known_values() -> None:
    assert norm_cdf(0.0) == 0.5
    assert math.isclose(norm_cdf(1.96), 0.9750021, abs_tol=1e-6)
    assert math.isclose(norm_cdf(-1.96), 0.0249979, abs_tol=1e-6)


def test_norm_pdf_known_values() -> None:
    assert math.isclose(norm_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))
    assert math.isclose(norm_pdf(1.0), 0.2419707, abs_tol=1e-6)


def test_price_dispatches_call_and_put() -> None:
    market = MarketData(spot=42.0, rate=0.10, vol=0.20)
    call = EuropeanOption(strike=40.0, maturity=0.5, option_type=OptionType.CALL)
    put = EuropeanOption(strike=40.0, maturity=0.5, option_type=OptionType.PUT)
    assert math.isclose(price(call, market), 4.759422, abs_tol=1e-6)
    assert math.isclose(price(put, market), 0.808599, abs_tol=1e-6)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_zero_maturity_price_is_intrinsic(option_type: OptionType) -> None:
    market = MarketData(spot=110.0, rate=0.05, vol=0.2)
    opt = EuropeanOption(strike=100.0, maturity=0.0, option_type=option_type)
    expected = 10.0 if option_type is OptionType.CALL else 0.0
    assert price(opt, market) == expected


def test_zero_vol_price_is_discounted_forward_payoff() -> None:
    market = MarketData(spot=100.0, rate=0.05, vol=0.0)
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    forward = 100.0 * math.exp(0.05)
    expected = math.exp(-0.05) * (forward - 100.0)
    assert math.isclose(price(opt, market), expected, rel_tol=1e-12)


def test_zero_vol_price_matches_positive_vol_limit() -> None:
    market_limit = MarketData(spot=100.0, rate=0.05, vol=1e-8)
    market_zero = MarketData(spot=100.0, rate=0.05, vol=0.0)
    opt = EuropeanOption(strike=80.0, maturity=1.0, option_type=OptionType.CALL)
    assert math.isclose(price(opt, market_zero), price(opt, market_limit), rel_tol=1e-9)


def test_d1_d2_atm() -> None:
    # ATM, r=q=0: d1 = sigma*sqrt(T)/2, d2 = -d1.
    d1, d2 = d1_d2(spot=100.0, strike=100.0, maturity=1.0, rate=0.0, vol=0.2)
    assert math.isclose(d1, 0.1)
    assert math.isclose(d2, -0.1)


def test_call_price_reference_value() -> None:
    # Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call = 4.76.
    assert math.isclose(
        call_price(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20),
        4.759422,
        abs_tol=1e-6,
    )


def test_call_price_with_dividend_yield() -> None:
    # Continuous dividends lower the call price.
    base = call_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2)
    with_div = call_price(
        spot=100.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2, dividend_yield=0.03
    )
    assert with_div < base


def test_put_price_reference_value() -> None:
    # Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 -> put = 0.81.
    assert math.isclose(
        put_price(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20),
        0.808599,
        abs_tol=1e-6,
    )


def test_deep_itm_put_close_to_discounted_intrinsic() -> None:
    p = put_price(spot=10.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2)
    assert math.isclose(p, 100.0 * math.exp(-0.05) - 10.0, rel_tol=1e-6)


def test_d1_d2_reference_value() -> None:
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5.
    d1, d2 = d1_d2(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20)
    assert math.isclose(d1, 0.7693, abs_tol=1e-4)
    assert math.isclose(d2, 0.6278, abs_tol=1e-4)
