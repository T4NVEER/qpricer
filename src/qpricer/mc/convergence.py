"""Convergence analysis of the Monte Carlo estimator against Black-Scholes."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qpricer._validation import require_positive
from qpricer.analytic import black_scholes as bs
from qpricer.instruments import EuropeanOption
from qpricer.market import MarketData
from qpricer.mc.engine import mc_price


@dataclass(frozen=True, slots=True)
class ConvergencePoint:
    """RMSE of independent MC estimates at one path count."""

    n_paths: int
    rmse: float
    mean_std_error: float


def error_study(
    option: EuropeanOption,
    market: MarketData,
    path_counts: Sequence[int],
    n_repeats: int = 20,
    seed: int | None = None,
) -> list[ConvergencePoint]:
    """Measure MC pricing error against the closed form at each path count.

    Runs n_repeats independent estimates per count (child seeds spawned from
    one SeedSequence) and reports the RMSE around the Black-Scholes price
    alongside the average reported standard error, which should track it.
    """
    require_positive("n_repeats", n_repeats)
    exact = bs.price(option, market)
    seed_seq = np.random.SeedSequence(seed)

    points = []
    for n_paths in path_counts:
        child_seeds = seed_seq.spawn(n_repeats)
        errors = []
        std_errors = []
        for child in child_seeds:
            result = mc_price(option, market, n_paths, seed=int(child.generate_state(1)[0]))
            errors.append(result.price - exact)
            std_errors.append(result.std_error)
        rmse = math.sqrt(sum(e * e for e in errors) / n_repeats)
        points.append(
            ConvergencePoint(
                n_paths=n_paths,
                rmse=rmse,
                mean_std_error=sum(std_errors) / n_repeats,
            )
        )
    return points


def paths_for_relative_error(
    option: EuropeanOption,
    market: MarketData,
    target_rel_error: float,
    seed: int | None = None,
    pilot_paths: int = 100_000,
) -> int:
    """Path count for the standard error to reach target_rel_error * price.

    A pilot run estimates the payoff standard deviation, then n is solved from
    SE(n) = sd / sqrt(n) <= target_rel_error * price.
    """
    require_positive("target_rel_error", target_rel_error)
    pilot = mc_price(option, market, pilot_paths, seed=seed)
    if pilot.price <= 0.0:
        raise ValueError("pilot price is zero; relative error target is undefined")
    payoff_sd = pilot.std_error * math.sqrt(pilot_paths)
    return math.ceil((payoff_sd / (target_rel_error * pilot.price)) ** 2)
