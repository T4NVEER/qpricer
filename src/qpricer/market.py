"""Market data inputs shared by all pricing engines."""

from dataclasses import dataclass


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
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol < 0.0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")
