import math

import numpy as np
import pytest

from qpricer import EuropeanOption, MarketData, OptionType
from qpricer.analytic import black_scholes as bs
from qpricer.mc.engine import mc_price, sample_terminal_spots

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2, dividend_yield=0.01)


def test_sampler_lognormal_moments() -> None:
    # E[S_T] = S0 * exp((r - q) T); Var[log S_T] = vol^2 T.
    rng = np.random.default_rng(7)
    spots = sample_terminal_spots(MARKET, maturity=1.0, n_paths=2_000_000, rng=rng)

    expected_mean = 100.0 * math.exp(0.04)
    assert math.isclose(spots.mean(), expected_mean, rel_tol=5e-4)
    assert math.isclose(np.log(spots).std(ddof=1), 0.2, rel_tol=5e-3)


def test_sampler_seed_reproducibility() -> None:
    a = sample_terminal_spots(MARKET, 1.0, 100, np.random.default_rng(42))
    b = sample_terminal_spots(MARKET, 1.0, 100, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_sampler_all_positive() -> None:
    spots = sample_terminal_spots(MARKET, 5.0, 10_000, np.random.default_rng(0))
    assert (spots > 0.0).all()


def test_mc_result_confidence_interval() -> None:
    from qpricer.mc.engine import MCResult

    res = MCResult(price=10.0, std_error=0.5, n_paths=1000)
    lo, hi = res.confidence_interval()
    assert math.isclose(hi - lo, 2.0 * 1.959963984540054 * 0.5)
    assert math.isclose((lo + hi) / 2.0, 10.0)


def test_mc_price_returns_finite_estimate() -> None:
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    res = mc_price(opt, MARKET, n_paths=50_000, seed=1)
    assert res.price > 0.0
    assert res.std_error > 0.0
    assert res.n_paths == 50_000


def test_mc_price_deterministic_limits() -> None:
    expired = EuropeanOption(strike=90.0, maturity=0.0, option_type=OptionType.CALL)
    res = mc_price(expired, MARKET, n_paths=10)
    assert res.price == 10.0
    assert res.std_error == 0.0


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_mc_price_within_3_sigma_of_black_scholes(option_type: OptionType) -> None:
    # Statistical test: BS price should lie inside price +/- 3 SE. With a
    # fixed seed this is deterministic; ~99.7% of seeds would pass.
    opt = EuropeanOption(strike=105.0, maturity=1.0, option_type=option_type)
    res = mc_price(opt, MARKET, n_paths=500_000, seed=2024)
    exact = bs.price(opt, MARKET)
    assert abs(res.price - exact) < 3.0 * res.std_error


def test_mc_std_error_shrinks_as_sqrt_n() -> None:
    opt = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    se_small = mc_price(opt, MARKET, n_paths=10_000, seed=3).std_error
    se_large = mc_price(opt, MARKET, n_paths=1_000_000, seed=3).std_error
    ratio = se_small / se_large
    assert 8.0 < ratio < 12.0  # sqrt(100) = 10 expected
