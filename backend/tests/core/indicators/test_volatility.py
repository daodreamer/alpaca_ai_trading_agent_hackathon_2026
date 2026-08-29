"""ATR — ADR 0007 D5.

The interesting part of ATR is not the smoothing but the true range: it needs
the *previous* close, which is why the first bar of a series contributes nothing
and the warmup is one bar longer than the period.
"""

from __future__ import annotations

import pytest

from alphagate.core.indicators import Atr, compute_series
from tests.core.indicators.synthetic import close_bars, ohlc_bars

# open, high, low, close
WORKED = [
    ("9.5", "10", "9", "9.5"),
    ("10.5", "11", "10", "10.5"),  # TR = max(1, |11-9.5|, |10-9.5|)   = 1.5
    ("10", "10.5", "9.5", "10"),  # TR = max(1, |10.5-10.5|, |9.5-10.5|) = 1.0
    ("11.5", "12", "11", "11.5"),  # TR = max(1, |12-10|, |11-10|)     = 2.0
]


def test_atr_warmup_needs_one_more_bar_than_its_period() -> None:
    assert Atr(period=14).warmup == 15
    assert compute_series(Atr(period=2), ohlc_bars(WORKED[:2])).values == (None, None)


def test_atr_seeds_with_the_mean_of_the_first_n_true_ranges_then_smooths_by_wilder() -> None:
    series = compute_series(Atr(period=2), ohlc_bars(WORKED))

    assert series.values[:2] == (None, None)
    assert series.values[2:] == pytest.approx([1.25, 1.625])


def test_atr_ignores_the_first_bars_own_range() -> None:
    """No previous close, so no true range — the seed must not depend on it."""
    widened = [("9.5", "40", "1", "9.5"), *WORKED[1:]]

    assert compute_series(Atr(period=2), ohlc_bars(widened)).values == pytest.approx(
        compute_series(Atr(period=2), ohlc_bars(WORKED)).values
    )


def test_atr_of_a_constant_range_is_that_range() -> None:
    rows = [("10", "11", "9", "10")] * 20
    series = compute_series(Atr(period=5), ohlc_bars(rows))

    assert series.values[5:] == pytest.approx([2.0] * 15)


def test_true_range_counts_a_gap_beyond_the_bars_own_high_low() -> None:
    """A bar that opens far above the previous close has a range larger than h-l."""
    rows = [("10", "10", "10", "10"), ("20", "21", "19", "20")]
    series = compute_series(Atr(period=1), ohlc_bars(rows))

    assert series.values[1] == pytest.approx(11.0)  # max(2, |21-10|, |19-10|)


def test_atr_of_a_series_that_never_moves_is_zero() -> None:
    series = compute_series(Atr(period=3), close_bars(["10"] * 8))

    assert series.values[3:] == pytest.approx([0.0] * 5)


def test_atr_is_never_negative() -> None:
    rows = [
        ("10", "12", "9", "11"),
        ("11", "11.5", "8", "8.5"),
        ("8.5", "14", "8.5", "13"),
        ("13", "13", "6", "6.5"),
        ("6.5", "9", "6", "8"),
    ]
    series = compute_series(Atr(period=2), ohlc_bars(rows))

    assert all(value >= 0.0 for value in series.values[2:] if value is not None)
