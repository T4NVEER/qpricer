import pytest

from qpricer import EuropeanOption, MarketData, OptionType, mc_price
from qpricer.analytic import black_scholes as bs
from qpricer.mc.variance_reduction import mc_price_antithetic

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
