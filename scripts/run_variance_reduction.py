"""Variance reduction effectiveness across moneyness.

Writes reports/variance_reduction.csv and reports/variance_reduction.png.
Ratios are variance reduction factors vs plain MC at equal path count: a
factor of R means plain MC needs R times more paths for the same error.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qpricer import EuropeanOption, MarketData, OptionType, variance_ratios

BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"

MARKET = MarketData(spot=100.0, rate=0.05, vol=0.2)
STRIKES = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
N_PATHS = 400_000
SEED = 20240712

REPORTS = Path(__file__).resolve().parent.parent / "reports"
TECHNIQUES = ["antithetic", "control_variate", "antithetic_cv"]
COLORS = {"antithetic": BLUE, "control_variate": AQUA, "antithetic_cv": YELLOW}


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    rows = []
    for strike in STRIKES:
        call = EuropeanOption(strike=strike, maturity=1.0, option_type=OptionType.CALL)
        ratios = variance_ratios(call, MARKET, N_PATHS, seed=SEED)
        rows.append({"strike": strike, **{k: round(v, 2) for k, v in ratios.items()}})

    with (REPORTS / "variance_reduction.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["strike", *TECHNIQUES])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(STRIKES))
    width = 0.27
    for i, tech in enumerate(TECHNIQUES):
        ax.bar(
            x + (i - 1) * width,
            [row[tech] for row in rows],
            width * 0.93,
            color=COLORS[tech],
            label=tech.replace("_", " + ") if tech == "antithetic_cv" else tech.replace("_", " "),
        )
    ax.set_yscale("log")
    ax.axhline(1.0, color="#8a8a8a", lw=1.0, ls=":")
    ax.set_xticks(x, [f"{int(k)}" for k in STRIKES])
    ax.set_xlabel("strike (spot = 100)")
    ax.set_ylabel("variance reduction factor vs plain MC")
    ax.set_title("Variance reduction by technique and moneyness (1y call)")
    ax.grid(True, axis="y", which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(REPORTS / "variance_reduction.png", dpi=150)

    header = f"{'strike':>8}" + "".join(f"{t:>18}" for t in TECHNIQUES)
    print(header)
    for row in rows:
        print(f"{row['strike']:>8}" + "".join(f"{row[t]:>18}" for t in TECHNIQUES))


if __name__ == "__main__":
    main()
