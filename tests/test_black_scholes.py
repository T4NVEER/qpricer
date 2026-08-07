import math

from qpricer.analytic.black_scholes import d1_d2, norm_cdf, norm_pdf


def test_norm_cdf_known_values() -> None:
    assert norm_cdf(0.0) == 0.5
    assert math.isclose(norm_cdf(1.96), 0.9750021, abs_tol=1e-6)
    assert math.isclose(norm_cdf(-1.96), 0.0249979, abs_tol=1e-6)


def test_norm_pdf_known_values() -> None:
    assert math.isclose(norm_pdf(0.0), 1.0 / math.sqrt(2.0 * math.pi))
    assert math.isclose(norm_pdf(1.0), 0.2419707, abs_tol=1e-6)


def test_d1_d2_atm() -> None:
    # ATM, r=q=0: d1 = sigma*sqrt(T)/2, d2 = -d1.
    d1, d2 = d1_d2(spot=100.0, strike=100.0, maturity=1.0, rate=0.0, vol=0.2)
    assert math.isclose(d1, 0.1)
    assert math.isclose(d2, -0.1)


def test_d1_d2_reference_value() -> None:
    # Hull, Options Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5.
    d1, d2 = d1_d2(spot=42.0, strike=40.0, maturity=0.5, rate=0.10, vol=0.20)
    assert math.isclose(d1, 0.7693, abs_tol=1e-4)
    assert math.isclose(d2, 0.6278, abs_tol=1e-4)
