"""Benchmark MC pricing throughput: NumPy vs numba vs numba_parallel.

Writes reports/benchmark_backends.csv and prints a table. The JIT is warmed
up first so compile time is excluded; each cell is the median of repeats.
"""

import csv
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from qpricer import EuropeanOption, MarketData, OptionType, mc_price
from qpricer.mc import numba_kernels

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
PATH_COUNTS = [100_000, 1_000_000, 10_000_000]
BACKENDS = ["numpy", "numba", "numba_parallel"]
REPEATS = 7
SEED = 7

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def median_seconds(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    numba_kernels.warm_up()
    REPORTS.mkdir(exist_ok=True)

    rows = []
    for n_paths in PATH_COUNTS:
        base_time = None
        for backend in BACKENDS:
            t = median_seconds(
                lambda: mc_price(CALL, MARKET, n_paths, seed=SEED, backend=backend)  # noqa: B023
            )
            if backend == "numpy":
                base_time = t
            assert base_time is not None
            rows.append(
                {
                    "n_paths": n_paths,
                    "backend": backend,
                    "median_ms": round(t * 1e3, 3),
                    "speedup_vs_numpy": round(base_time / t, 2),
                }
            )

    with (REPORTS / "benchmark_backends.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'N':>12}  {'backend':>16}  {'median ms':>10}  {'speedup':>8}")
    for row in rows:
        print(
            f"{row['n_paths']:>12,}  {row['backend']:>16}  "
            f"{row['median_ms']:>10}  {row['speedup_vs_numpy']:>8}x"
        )


if __name__ == "__main__":
    main()
