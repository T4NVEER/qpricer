"""qpricer: European option pricing by analytic, lattice and Monte Carlo methods."""

from qpricer.analytic.black_scholes import price as bs_price
from qpricer.analytic.implied_vol import implied_vol
from qpricer.instruments import EuropeanOption, OptionType
from qpricer.market import MarketData
from qpricer.mc.engine import MCResult, mc_price
from qpricer.tree.binomial import crr_price

__version__ = "0.0.1"

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
]
