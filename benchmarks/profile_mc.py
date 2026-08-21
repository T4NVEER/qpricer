"""Profile the NumPy MC path to find hotspots.

Prints the top functions by internal time and a coarse stage breakdown
(RNG draw / exp / payoff+reduction) measured directly. cProfile cannot see
inside NumPy ufuncs, so the stage timing is the actionable output.
"""

import cProfile
import pstats
import time

import numpy as np

from qpricer import EuropeanOption, MarketData, OptionType, mc_price

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
N_PATHS = 10_000_000
SEED = 7


def stage_breakdown() -> None:
    rng = np.random.default_rng(SEED)

    start = time.perf_counter()
    z = rng.standard_normal(N_PATHS)
    t_rng = time.perf_counter() - start

    start = time.perf_counter()
    spots = 100.0 * np.exp(0.03 + 0.2 * z)
    t_exp = time.perf_counter() - start

    start = time.perf_counter()
    payoff = np.maximum(spots - 100.0, 0.0)
    discounted = 0.95 * payoff
    total = discounted.sum()
    total_sq = (discounted * discounted).sum()
    t_reduce = time.perf_counter() - start

    del total, total_sq
    total_t = t_rng + t_exp + t_reduce
    print(f"\nstage breakdown at {N_PATHS:,} paths:")
    for name, t in [("rng draw", t_rng), ("exp/drift", t_exp), ("payoff+reduce", t_reduce)]:
        print(f"  {name:>14}: {t * 1e3:8.1f} ms  ({t / total_t:5.1%})")


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    mc_price(CALL, MARKET, N_PATHS, seed=SEED)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    stats.print_stats(8)

    stage_breakdown()


if __name__ == "__main__":
    main()
