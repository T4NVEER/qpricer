import math

import numpy as np
import pytest

from qpricer import EuropeanOption, MarketData, OptionType, mc_price
from qpricer.mc import numba_kernels

pytestmark = pytest.mark.skipif(not numba_kernels.NUMBA_AVAILABLE, reason="numba not installed")

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)


def _kernel_price(option: EuropeanOption, market: MarketData, n_paths: int, seed: int) -> float:
    drift = (market.rate - market.dividend_yield - 0.5 * market.vol**2) * option.maturity
    diffusion = market.vol * math.sqrt(option.maturity)
    discount = math.exp(-market.rate * option.maturity)
    sign = 1.0 if option.option_type is OptionType.CALL else -1.0
    z = np.random.default_rng(seed).standard_normal(n_paths)
    total, _ = numba_kernels.payoff_sums_serial(
        z, market.spot, option.strike, sign, drift, diffusion, discount
    )
    return total / n_paths


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_serial_kernel_matches_numpy_engine(option_type: OptionType) -> None:
    opt = EuropeanOption(strike=95.0, maturity=1.0, option_type=option_type)
    n, seed = 100_000, 41
    kernel = _kernel_price(opt, MARKET, n, seed)
    numpy_ref = mc_price(opt, MARKET, n, seed=seed).price
    # Same normals; fastmath only reorders the summation.
    assert math.isclose(kernel, numpy_ref, rel_tol=1e-10)


def test_parallel_kernel_matches_serial() -> None:
    numba_kernels.warm_up()
    z = np.random.default_rng(43).standard_normal(200_000)
    args = (z, 100.0, 105.0, 1.0, 0.03, 0.2, 0.95)
    total_s, sq_s = numba_kernels.payoff_sums_serial(*args)
    total_p, sq_p = numba_kernels.payoff_sums_parallel(*args)
    assert math.isclose(total_s, total_p, rel_tol=1e-10)
    assert math.isclose(sq_s, sq_p, rel_tol=1e-10)


@pytest.mark.parametrize("backend", ["numba", "numba_parallel"])
def test_engine_backends_match_numpy(backend: str) -> None:
    n, seed = 200_000, 47
    numpy_res = mc_price(CALL, MARKET, n, seed=seed)
    numba_res = mc_price(CALL, MARKET, n, seed=seed, backend=backend)  # type: ignore[arg-type]
    assert math.isclose(numba_res.price, numpy_res.price, rel_tol=1e-10)
    assert math.isclose(numba_res.std_error, numpy_res.std_error, rel_tol=1e-8)


def test_engine_backend_batching_consistent() -> None:
    full = mc_price(CALL, MARKET, 100_000, seed=53, backend="numba")
    batched = mc_price(CALL, MARKET, 100_000, seed=53, backend="numba", batch_size=9_973)
    assert math.isclose(full.price, batched.price, rel_tol=1e-12)
