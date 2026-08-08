import math

from qpricer.analytic.black_scholes import call_price
from qpricer.analytic.implied_vol import implied_vol
from qpricer.instruments import OptionType


def test_recovers_known_vol_atm_call() -> None:
    target = call_price(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, vol=0.2)
    iv = implied_vol(
        target, spot=100.0, strike=100.0, maturity=1.0, rate=0.05, option_type=OptionType.CALL
    )
    assert math.isclose(iv, 0.2, abs_tol=1e-8)
