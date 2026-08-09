import math

import pytest

from qpricer.analytic.black_scholes import call_price, put_price
from qpricer.analytic.implied_vol import implied_vol
from qpricer.instruments import OptionType


def test_recovers_known_vol_atm_call() -> None:
    target = call_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2)
    iv = implied_vol(
        target, spot=100.0, strike=100.0, maturity=1.0, rate=0.05, option_type=OptionType.CALL
    )
    assert math.isclose(iv, 0.2, abs_tol=1e-8)


def test_deep_otm_falls_back_to_bisection() -> None:
    # Deep OTM: tiny vega makes pure Newton unreliable.
    params = {"spot": 100.0, "strike": 300.0, "maturity": 0.25, "rate": 0.02}
    target = call_price(**params, vol=0.35)
    iv = implied_vol(target, **params, option_type=OptionType.CALL)
    assert math.isclose(iv, 0.35, abs_tol=1e-6)


def test_price_above_upper_bound_rejected() -> None:
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(
            120.0,
            spot=100.0,
            strike=100.0,
            maturity=1.0,
            rate=0.05,
            option_type=OptionType.CALL,
        )


def test_price_below_intrinsic_rejected() -> None:
    # Deep ITM put priced below its lower bound.
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_vol(
            10.0,
            spot=50.0,
            strike=100.0,
            maturity=1.0,
            rate=0.0,
            option_type=OptionType.PUT,
        )


# Grid is restricted to |log-moneyness| / (vol * sqrt(T)) below ~4.5: beyond that
# the extrinsic value approaches double-precision noise and vol is unrecoverable
# from the price by any solver.
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [80.0, 90.0, 100.0, 110.0, 125.0])
@pytest.mark.parametrize("maturity", [0.25, 1.0, 5.0])
@pytest.mark.parametrize("vol", [0.1, 0.2, 0.8])
def test_round_trip_across_grid(
    option_type: OptionType, strike: float, maturity: float, vol: float
) -> None:
    params = {"spot": 100.0, "strike": strike, "maturity": maturity, "rate": 0.03}
    price_fn = call_price if option_type is OptionType.CALL else put_price
    target = price_fn(**params, vol=vol, dividend_yield=0.01)
    iv = implied_vol(target, **params, option_type=option_type, dividend_yield=0.01)
    assert math.isclose(iv, vol, rel_tol=1e-6)
