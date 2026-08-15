import math

from qpricer import EuropeanOption, MarketData, OptionType, mc_price
from qpricer.mc.convergence import error_study, paths_for_relative_error

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def test_rmse_decreases_with_paths() -> None:
    points = error_study(CALL, MARKET, path_counts=[1_000, 16_000, 256_000], n_repeats=10, seed=23)
    rmses = [p.rmse for p in points]
    assert rmses[0] > rmses[1] > rmses[2]


def test_rmse_matches_reported_standard_error() -> None:
    # The reported SE should estimate the actual sampling error within ~2x.
    points = error_study(CALL, MARKET, path_counts=[10_000], n_repeats=30, seed=29)
    point = points[0]
    assert 0.5 < point.rmse / point.mean_std_error < 2.0


def test_paths_for_relative_error_hits_target() -> None:
    target = 0.01
    n = paths_for_relative_error(CALL, MARKET, target_rel_error=target, seed=31)
    result = mc_price(CALL, MARKET, n, seed=37)
    achieved = result.std_error / result.price
    assert achieved < target * 1.1


def test_tighter_target_needs_quadratically_more_paths() -> None:
    n_1pct = paths_for_relative_error(CALL, MARKET, 0.01, seed=31)
    n_half_pct = paths_for_relative_error(CALL, MARKET, 0.005, seed=31)
    assert math.isclose(n_half_pct / n_1pct, 4.0, rel_tol=0.01)
