"""Instrument definitions and payoffs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from qpricer._validation import require_non_negative, require_positive


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True, slots=True)
class EuropeanOption:
    """European vanilla option. Maturity is in years from valuation date."""

    strike: float
    maturity: float
    option_type: OptionType

    def __post_init__(self) -> None:
        require_positive("strike", self.strike)
        require_non_negative("maturity", self.maturity)

    def payoff(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        """Terminal payoff, vectorized over spot prices."""
        if self.option_type is OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)
