import pytest

from qpricer.market import MarketData


def test_valid_market_data() -> None:
    md = MarketData(spot=100.0, rate=0.05, vol=0.2, dividend_yield=0.01)
    assert md.spot == 100.0
    assert md.dividend_yield == 0.01


def test_default_dividend_yield_is_zero() -> None:
    assert MarketData(spot=100.0, rate=0.05, vol=0.2).dividend_yield == 0.0


def test_negative_rate_allowed() -> None:
    assert MarketData(spot=100.0, rate=-0.01, vol=0.2).rate == -0.01


def test_zero_vol_allowed() -> None:
    assert MarketData(spot=100.0, rate=0.05, vol=0.0).vol == 0.0


@pytest.mark.parametrize("spot", [0.0, -1.0])
def test_non_positive_spot_rejected(spot: float) -> None:
    with pytest.raises(ValueError, match="spot"):
        MarketData(spot=spot, rate=0.05, vol=0.2)


def test_negative_vol_rejected() -> None:
    with pytest.raises(ValueError, match="vol"):
        MarketData(spot=100.0, rate=0.05, vol=-0.2)


def test_frozen() -> None:
    md = MarketData(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(AttributeError):
        md.spot = 101.0  # type: ignore[misc]
