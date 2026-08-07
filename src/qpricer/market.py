"""Market data inputs shared by all pricing engines."""

from dataclasses import dataclass

from qpricer._validation import require_non_negative, require_positive


@dataclass(frozen=True, slots=True)
class MarketData:
    """Flat-parameter market snapshot for a single underlying.

    Rates may be negative; volatility of zero is allowed so engines can be
    exercised against deterministic limits.
    """

    spot: float
    rate: float
    vol: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        require_positive("spot", self.spot)
        require_non_negative("vol", self.vol)
