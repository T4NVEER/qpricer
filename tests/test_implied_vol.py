import math

import pytest

from qpricer.analytic.black_scholes import call_price
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
