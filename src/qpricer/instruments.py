"""Instrument definitions and payoffs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


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
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity < 0.0:
            raise ValueError(f"maturity must be non-negative, got {self.maturity}")

    def payoff(self, spot: NDArray[np.float64]) -> NDArray[np.float64]:
        """Terminal payoff, vectorized over spot prices."""
        if self.option_type is OptionType.CALL:
            return np.maximum(spot - self.strike, 0.0)
        return np.maximum(self.strike - spot, 0.0)
