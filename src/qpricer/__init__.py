"""qpricer: European option pricing by analytic, lattice and Monte Carlo methods."""

from qpricer.analytic.black_scholes import price as bs_price
from qpricer.analytic.implied_vol import implied_vol
from qpricer.instruments import EuropeanOption, OptionType
from qpricer.market import MarketData
from qpricer.mc.engine import MCResult, mc_price
from qpricer.mc.variance_reduction import (
    mc_price_antithetic,
    mc_price_antithetic_cv,
    mc_price_control_variate,
    variance_ratios,
)
from qpricer.tree.binomial import crr_price

__version__ = "0.1.0"

__all__ = [
    "EuropeanOption",
    "MCResult",
    "MarketData",
    "OptionType",
    "__version__",
    "bs_price",
    "crr_price",
    "implied_vol",
    "mc_price",
    "mc_price_antithetic",
    "mc_price_antithetic_cv",
    "mc_price_control_variate",
    "variance_ratios",
]
