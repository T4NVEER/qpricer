"""Verify analytic Greeks against central finite differences of the price."""

import math
from collections.abc import Callable

import pytest

from qpricer.analytic import black_scholes as bs

S, K, T, R, VOL, Q = 105.0, 100.0, 0.75, 0.04, 0.25, 0.015
EPS = 1e-5


def central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize(
    ("price_fn", "greek_fn"),
    [(bs.call_price, bs.call_delta), (bs.put_price, bs.put_delta)],
    ids=["call", "put"],
)
def test_delta_matches_fd(price_fn: Callable[..., float], greek_fn: Callable[..., float]) -> None:
    fd = central_diff(lambda s: price_fn(s, K, T, R, VOL, Q), S, S * EPS)
    assert math.isclose(greek_fn(S, K, T, R, VOL, Q), fd, rel_tol=1e-6)


def test_gamma_matches_fd() -> None:
    fd = central_diff(lambda s: bs.call_delta(s, K, T, R, VOL, Q), S, S * EPS)
    assert math.isclose(bs.gamma(S, K, T, R, VOL, Q), fd, rel_tol=1e-6)


def test_vega_matches_fd() -> None:
    fd = central_diff(lambda v: bs.call_price(S, K, T, R, v, Q), VOL, EPS)
    assert math.isclose(bs.vega(S, K, T, R, VOL, Q), fd, rel_tol=1e-6)


@pytest.mark.parametrize(
    ("price_fn", "greek_fn"),
    [(bs.call_price, bs.call_theta), (bs.put_price, bs.put_theta)],
    ids=["call", "put"],
)
def test_theta_matches_fd(price_fn: Callable[..., float], greek_fn: Callable[..., float]) -> None:
    # Theta is dV/dt = -dV/dT for a fixed calendar date.
    fd = -central_diff(lambda t: price_fn(S, K, t, R, VOL, Q), T, EPS)
    assert math.isclose(greek_fn(S, K, T, R, VOL, Q), fd, rel_tol=1e-6)


@pytest.mark.parametrize(
    ("price_fn", "greek_fn"),
    [(bs.call_price, bs.call_rho), (bs.put_price, bs.put_rho)],
    ids=["call", "put"],
)
def test_rho_matches_fd(price_fn: Callable[..., float], greek_fn: Callable[..., float]) -> None:
    fd = central_diff(lambda r: price_fn(S, K, T, r, VOL, Q), R, EPS)
    assert math.isclose(greek_fn(S, K, T, R, VOL, Q), fd, rel_tol=1e-6)
