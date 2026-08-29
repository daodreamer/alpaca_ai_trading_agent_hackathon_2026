"""Session VWAP and volume MA — ADR 0007 D5.

VWAP is the one MVP indicator whose state is tied to the trading day rather than
to a bar count, so most of what matters here is when it resets and which bars it
is allowed to see.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from alphagate.core.errors import InvariantViolation
from alphagate.core.indicators import SessionVwap, VolumeSma, compute_series
from alphagate.core.time_model import SessionKind
from tests.core.indicators.synthetic import bar_at, close_bars, ohlc_bars

NEXT_DAY = date(2026, 8, 14)

# typical price (h+l+c)/3 = 10 then 12
TWO_BARS = [("10", "11", "9", "10"), ("11", "13", "11", "12")]


def test_session_vwap_is_volume_weighted_typical_price() -> None:
    series = compute_series(SessionVwap(), ohlc_bars(TWO_BARS))

    assert series.values == pytest.approx([10.0, 11.0])


def test_session_vwap_weights_by_volume_not_by_bar_count() -> None:
    bars = (
        bar_at(0, ("10", "11", "9", "10"), volume="100"),
        bar_at(1, ("11", "13", "11", "12"), volume="300"),
    )
    series = compute_series(SessionVwap(), bars)

    assert series.values[1] == pytest.approx((10 * 100 + 12 * 300) / 400)


def test_session_vwap_resets_when_the_trading_day_changes() -> None:
    bars = (
        bar_at(0, ("10", "11", "9", "10")),
        bar_at(1, ("11", "13", "11", "12")),
        bar_at(2, ("20", "21", "19", "20"), session_date=NEXT_DAY),
    )
    series = compute_series(SessionVwap(), bars)

    assert series.values == pytest.approx([10.0, 11.0, 20.0])


def test_session_vwap_ignores_bars_outside_the_configured_sessions() -> None:
    """Default is REGULAR only: a pre-market print must not move the session VWAP."""
    bars = (
        bar_at(0, ("10", "11", "9", "10"), session=SessionKind.PRE),
        bar_at(1, ("11", "13", "11", "12")),
    )
    series = compute_series(SessionVwap(), bars)

    assert series.values[0] is None
    assert series.values[1] == pytest.approx(12.0)


def test_session_vwap_can_be_configured_to_include_extended_hours() -> None:
    bars = (
        bar_at(0, ("10", "11", "9", "10"), session=SessionKind.PRE),
        bar_at(1, ("11", "13", "11", "12")),
    )
    series = compute_series(SessionVwap(sessions=(SessionKind.PRE, SessionKind.REGULAR)), bars)

    assert series.values == pytest.approx([10.0, 11.0])


def test_session_vwap_needs_at_least_one_participating_session() -> None:
    with pytest.raises(InvariantViolation, match="session"):
        SessionVwap(sessions=())


def test_session_vwap_is_undefined_while_no_volume_has_traded() -> None:
    """A volume-weighted average of nothing is not zero."""
    bars = (
        bar_at(0, ("10", "11", "9", "10"), volume="0"),
        bar_at(1, ("11", "13", "11", "12"), volume="0"),
        bar_at(2, ("11", "13", "11", "12"), volume="500"),
    )
    series = compute_series(SessionVwap(), bars)

    assert series.values[:2] == (None, None)
    assert series.values[2] == pytest.approx(12.0)


def test_a_zero_volume_bar_does_not_move_an_established_vwap() -> None:
    bars = (
        bar_at(0, ("10", "11", "9", "10")),
        bar_at(1, ("99", "99", "99", "99"), volume="0"),
    )
    series = compute_series(SessionVwap(), bars)

    assert series.values == pytest.approx([10.0, 10.0])


def test_session_vwap_ignores_a_provider_supplied_bar_vwap() -> None:
    """ADR 0007 D5: the provider's aggregate covers a different trade population."""
    plain = bar_at(0, ("10", "11", "9", "10"))
    with_vendor_vwap = bar_at(0, ("10", "11", "9", "10")).revised(vwap="10.9")

    assert (
        compute_series(SessionVwap(), (plain,)).values
        == compute_series(SessionVwap(), (with_vendor_vwap,)).values
    )


def test_session_vwap_does_not_lose_small_terms_to_a_large_running_total() -> None:
    """ADR 0007 D2 — the session is re-summed with `math.fsum`, not accumulated.

    One enormous term followed by many tiny ones: with a running `+=` every tiny
    term falls off the bottom of the total and is gone for good. The expected
    value is computed in exact `Decimal` arithmetic, so this compares against the
    real answer rather than against another float pipeline.
    """
    light_count = 2000
    heavy = bar_at(0, ("100000000", "100000000", "100000000", "100000000"), volume="100000000")
    light = [
        bar_at(index, ("0.1", "0.1", "0.1", "0.1"), volume="1")
        for index in range(1, light_count + 1)
    ]

    series = compute_series(SessionVwap(), (heavy, *light))

    # Each light term is 0.1, far below half an ulp of the 1e16 running total, so
    # a `+=` loses every one of them. 2000 of them are worth 200.
    notional = Decimal("100000000") * Decimal("100000000") + Decimal("0.1") * light_count
    volume = Decimal("100000000") + light_count
    assert series.values[-1] == pytest.approx(float(notional / volume), rel=1e-15)


def test_session_vwap_lies_between_the_session_low_and_high() -> None:
    rows = [
        ("10", "12", "9", "11"),
        ("11", "15", "10", "14"),
        ("14", "14", "8", "9"),
        ("9", "10", "8", "10"),
    ]
    series = compute_series(SessionVwap(), ohlc_bars(rows))

    assert all(8.0 <= value <= 15.0 for value in series.values if value is not None)


# -- volume moving average --------------------------------------------------


def test_volume_sma_averages_the_last_n_volumes() -> None:
    bars = tuple(
        bar_at(index, ("100", "100", "100", "100"), volume=volume)
        for index, volume in enumerate(["100", "200", "300", "400"])
    )
    series = compute_series(VolumeSma(period=3), bars)

    assert series.values[:2] == (None, None)
    assert series.values[2:] == pytest.approx([200.0, 300.0])


def test_volume_sma_of_a_zero_volume_series_is_zero() -> None:
    bars = tuple(bar_at(index, ("100", "100", "100", "100"), volume="0") for index in range(5))
    series = compute_series(VolumeSma(period=2), bars)

    assert series.values[1:] == pytest.approx([0.0] * 4)


def test_volume_sma_ignores_price_entirely() -> None:
    volumes = ["10", "20", "30", "40", "50"]
    flat = tuple(
        bar_at(index, ("100", "100", "100", "100"), volume=volume)
        for index, volume in enumerate(volumes)
    )
    swinging = tuple(
        bar_at(index, ("100", str(100 + index), "50", "60"), volume=volume)
        for index, volume in enumerate(volumes)
    )

    assert compute_series(VolumeSma(period=3), flat).values == pytest.approx(
        compute_series(VolumeSma(period=3), swinging).values
    )


def test_close_bars_helper_produces_a_usable_default_volume() -> None:
    """Guards the fixtures the other modules rely on."""
    series = compute_series(VolumeSma(period=2), close_bars(["10", "11", "12"]))

    assert series.values[1:] == pytest.approx([1000.0, 1000.0])
