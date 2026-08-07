import numpy as np
import pytest

from qpricer.instruments import EuropeanOption, OptionType


def test_call_payoff() -> None:
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    spots = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(opt.payoff(spots), [0.0, 0.0, 20.0])


def test_put_payoff() -> None:
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.PUT)
    spots = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(opt.payoff(spots), [20.0, 0.0, 0.0])


def test_payoff_scalar_input() -> None:
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    assert opt.payoff(np.float64(110.0)) == 10.0


def test_zero_maturity_allowed() -> None:
    assert EuropeanOption(strike=100.0, maturity=0.0, option_type=OptionType.PUT).maturity == 0.0


@pytest.mark.parametrize("strike", [0.0, -5.0])
def test_non_positive_strike_rejected(strike: float) -> None:
    with pytest.raises(ValueError, match="strike"):
        EuropeanOption(strike=strike, maturity=1.0, option_type=OptionType.CALL)


def test_negative_maturity_rejected() -> None:
    with pytest.raises(ValueError, match="maturity"):
        EuropeanOption(strike=100.0, maturity=-1.0, option_type=OptionType.CALL)
