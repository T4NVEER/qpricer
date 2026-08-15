import pytest

from qpricer import EuropeanOption, MarketData, OptionType, mc_price
from qpricer.analytic import black_scholes as bs
from qpricer.mc.variance_reduction import (
    mc_price_antithetic,
    mc_price_antithetic_cv,
    mc_price_control_variate,
)

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2, dividend_yield=0.01)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def test_antithetic_within_3_sigma_of_black_scholes() -> None:
    res = mc_price_antithetic(CALL, MARKET, n_paths=200_000, seed=11)
    assert abs(res.price - bs.price(CALL, MARKET)) < 3.0 * res.std_error


def test_antithetic_reduces_standard_error() -> None:
    n = 200_000
    plain = mc_price(CALL, MARKET, n_paths=n, seed=11)
    anti = mc_price_antithetic(CALL, MARKET, n_paths=n, seed=11)
    assert anti.std_error < plain.std_error


def test_antithetic_rejects_odd_paths() -> None:
    with pytest.raises(ValueError, match="even"):
        mc_price_antithetic(CALL, MARKET, n_paths=1001, seed=11)


ITM_CALL = EuropeanOption(strike=80.0, maturity=1.0, option_type=OptionType.CALL)


def test_control_variate_within_3_sigma_of_black_scholes() -> None:
    res = mc_price_control_variate(CALL, MARKET, n_paths=200_000, seed=13)
    assert abs(res.price - bs.price(CALL, MARKET)) < 3.0 * res.std_error


def test_control_variate_reduces_standard_error() -> None:
    n = 200_000
    plain = mc_price(CALL, MARKET, n_paths=n, seed=13)
    cv = mc_price_control_variate(CALL, MARKET, n_paths=n, seed=13)
    assert cv.std_error < plain.std_error


def test_control_variate_strongest_for_itm_option() -> None:
    # ITM payoff is nearly linear in S_T, so the spot control removes most variance.
    n = 200_000
    plain = mc_price(ITM_CALL, MARKET, n_paths=n, seed=13)
    cv = mc_price_control_variate(ITM_CALL, MARKET, n_paths=n, seed=13)
    assert cv.std_error < 0.25 * plain.std_error


def test_combined_within_3_sigma_of_black_scholes() -> None:
    res = mc_price_antithetic_cv(CALL, MARKET, n_paths=200_000, seed=17)
    assert abs(res.price - bs.price(CALL, MARKET)) < 3.0 * res.std_error


def test_combined_beats_each_single_technique() -> None:
    n = 200_000
    anti = mc_price_antithetic(CALL, MARKET, n_paths=n, seed=17)
    cv = mc_price_control_variate(CALL, MARKET, n_paths=n, seed=17)
    both = mc_price_antithetic_cv(CALL, MARKET, n_paths=n, seed=17)
    assert both.std_error < anti.std_error
    assert both.std_error < cv.std_error


def test_combined_rejects_odd_paths() -> None:
    with pytest.raises(ValueError, match="even"):
        mc_price_antithetic_cv(CALL, MARKET, n_paths=999, seed=17)
