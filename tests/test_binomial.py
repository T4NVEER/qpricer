import math

import pytest

from qpricer import EuropeanOption, MarketData, OptionType
from qpricer.analytic import black_scholes as bs
from qpricer.tree.binomial import crr_price, crr_price_loop

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2, dividend_yield=0.01)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
PUT = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT)


@pytest.mark.parametrize("option", [CALL, PUT], ids=["call", "put"])
def test_loop_pricer_close_to_black_scholes(option: EuropeanOption) -> None:
    approx = crr_price_loop(option, MARKET, steps=1000)
    exact = bs.price(option, MARKET)
    assert math.isclose(approx, exact, rel_tol=1e-3)


def test_single_step_tree_matches_hand_calculation() -> None:
    market = MarketData(spot=100.0, rate=0.05, vol=0.2)
    option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    u = math.exp(0.2)
    d = 1.0 / u
    p = (math.exp(0.05) - d) / (u - d)
    expected = math.exp(-0.05) * p * (100.0 * u - 100.0)
    assert math.isclose(crr_price_loop(option, market, steps=1), expected, rel_tol=1e-12)


def test_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        crr_price_loop(CALL, MARKET, steps=0)


@pytest.mark.parametrize("option", [CALL, PUT], ids=["call", "put"])
@pytest.mark.parametrize("steps", [1, 2, 17, 250])
def test_vectorized_matches_loop_reference(option: EuropeanOption, steps: int) -> None:
    assert math.isclose(
        crr_price(option, MARKET, steps),
        crr_price_loop(option, MARKET, steps),
        rel_tol=1e-12,
    )
