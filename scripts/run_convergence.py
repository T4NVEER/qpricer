"""Convergence study: MC error vs path count, and paths needed for 1% error.

Writes reports/convergence.csv and reports/convergence.png, and prints the
summary quoted in the README. Requires the [analysis] extra (matplotlib).
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qpricer import EuropeanOption, MarketData, OptionType
from qpricer.mc.convergence import error_study, paths_for_relative_error

BLUE = "#2a78d6"
AQUA = "#1baf7a"
GRAY = "#8a8a8a"

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
CALL = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
PATH_COUNTS = [1_000, 4_000, 16_000, 64_000, 256_000, 1_024_000]
N_REPEATS = 30
SEED = 20240712

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    points = error_study(CALL, MARKET, PATH_COUNTS, n_repeats=N_REPEATS, seed=SEED)

    with (REPORTS / "convergence.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_paths", "rmse", "mean_std_error"])
        for p in points:
            writer.writerow([p.n_paths, f"{p.rmse:.6f}", f"{p.mean_std_error:.6f}"])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    n = [p.n_paths for p in points]
    ax.loglog(
        n, [p.rmse for p in points], color=BLUE, marker="o", lw=1.8, ms=5, label="observed RMSE"
    )
    ax.loglog(
        n,
        [p.mean_std_error for p in points],
        color=AQUA,
        marker="s",
        lw=1.8,
        ms=5,
        ls="--",
        label="mean reported SE",
    )
    ref = [points[0].rmse * (n[0] / x) ** 0.5 for x in n]
    ax.loglog(n, ref, color=GRAY, ls=":", lw=1.4, label=r"$O(N^{-1/2})$ reference")
    ax.set_xlabel("number of paths $N$")
    ax.set_ylabel("pricing error")
    ax.set_title("Monte Carlo error vs Black-Scholes (ATM call)")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "convergence.png", dpi=150)

    print(f"{'N':>10}  {'RMSE':>10}  {'mean SE':>10}")
    for p in points:
        print(f"{p.n_paths:>10}  {p.rmse:>10.5f}  {p.mean_std_error:>10.5f}")

    n_1pct = paths_for_relative_error(CALL, MARKET, 0.01, seed=SEED)
    print(f"\npaths for 1% relative standard error (ATM call): {n_1pct:,}")


if __name__ == "__main__":
    main()
