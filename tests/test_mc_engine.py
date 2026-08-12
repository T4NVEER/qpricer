import math

import numpy as np

from qpricer import MarketData
from qpricer.mc.engine import sample_terminal_spots

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
