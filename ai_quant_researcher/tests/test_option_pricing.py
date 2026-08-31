"""Black-Scholes, used for one thing only — specs/10-options-research.md D1a.

This model never moves money. Entry prices come from the vendor's bid and ask,
settlement comes from the underlying's close, and both are quoted numbers a
broker statement could be reconciled against. What the model produces is the
equity curve *between* those two events, because a book that only moves on cash
flows has a beta of zero by construction and would walk into `alpha.py`
claiming the entire return of a short put spread as alpha.

So the tests here check the arithmetic, and one of them checks the wall: the
realised P&L of a trade must not move when the model does.
"""

from __future__ import annotations

import math

import pytest

from aqr.options.pricing import black_scholes, intrinsic

YEAR = 1.0


def test_a_call_at_expiry_is_worth_its_intrinsic_value() -> None:
    assert black_scholes("call", spot=110.0, strike=100.0, years=0.0, iv=0.2) == pytest.approx(10.0)
    assert black_scholes("call", spot=90.0, strike=100.0, years=0.0, iv=0.2) == pytest.approx(0.0)


def test_a_put_at_expiry_is_worth_its_intrinsic_value() -> None:
    assert black_scholes("put", spot=90.0, strike=100.0, years=0.0, iv=0.2) == pytest.approx(10.0)
    assert black_scholes("put", spot=110.0, strike=100.0, years=0.0, iv=0.2) == pytest.approx(0.0)


def test_an_at_the_money_call_matches_the_closed_form() -> None:
    """S=K, r=0: the price is S(2N(sigma*sqrt(T)/2) - 1)."""
    got = black_scholes("call", spot=100.0, strike=100.0, years=YEAR, iv=0.20)
    expected = 100.0 * (2 * 0.5 * math.erfc(-(0.10) / math.sqrt(2)) - 1)
    assert got == pytest.approx(expected, rel=1e-9)


def test_put_call_parity_holds_at_zero_rates() -> None:
    """C - P = S - K. If this breaks, every spread is mispriced in one direction."""
    call = black_scholes("call", spot=105.0, strike=100.0, years=0.5, iv=0.25)
    put = black_scholes("put", spot=105.0, strike=100.0, years=0.5, iv=0.25)
    assert call - put == pytest.approx(5.0, abs=1e-9)


def test_value_is_never_below_intrinsic_and_never_above_the_underlying() -> None:
    for spot in (80.0, 100.0, 120.0):
        value = black_scholes("call", spot=spot, strike=100.0, years=0.25, iv=0.3)
        assert intrinsic("call", spot=spot, strike=100.0) <= value <= spot


def test_a_zero_volatility_option_is_worth_its_intrinsic_value() -> None:
    """Not a crash, and not a NaN: sigma*sqrt(T) is the denominator of d1."""
    assert black_scholes("call", spot=110.0, strike=100.0, years=1.0, iv=0.0) == pytest.approx(10.0)
    assert black_scholes("put", spot=110.0, strike=100.0, years=1.0, iv=0.0) == pytest.approx(0.0)


def test_time_value_decays_towards_expiry() -> None:
    far = black_scholes("put", spot=100.0, strike=100.0, years=0.50, iv=0.2)
    near = black_scholes("put", spot=100.0, strike=100.0, years=0.05, iv=0.2)
    assert far > near > 0


def test_a_call_gains_and_a_put_loses_as_spot_rises() -> None:
    """The property the curve exists for: the mark must carry the position's
    direction. A short put spread that cannot lose money when SPY falls is not
    a risk measurement."""
    up = black_scholes("call", spot=105.0, strike=100.0, years=0.25, iv=0.2)
    down = black_scholes("call", spot=95.0, strike=100.0, years=0.25, iv=0.2)
    assert up > down
    assert black_scholes("put", spot=105.0, strike=100.0, years=0.25, iv=0.2) < black_scholes(
        "put", spot=95.0, strike=100.0, years=0.25, iv=0.2
    )


def test_negative_time_is_refused() -> None:
    """A mark dated after the expiry is a bug in the caller's calendar, and
    silently clamping it would hide the day the position should have settled."""
    with pytest.raises(ValueError, match="years"):
        black_scholes("call", spot=100.0, strike=100.0, years=-0.1, iv=0.2)


def test_intrinsic_is_exact_at_the_strike() -> None:
    assert intrinsic("call", spot=100.0, strike=100.0) == 0.0
    assert intrinsic("put", spot=100.0, strike=100.0) == 0.0
