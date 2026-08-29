"""RSI and MACD — ADR 0007 D4/D5.

RSI is pinned twice: on tiny hand-calculable sequences, and against Wilder's
published worked example, which is the reference every charting package agrees
with. If those two ever disagree, the averaging method has drifted.
"""

from __future__ import annotations

import pytest

from alphagate.core.indicators import Ema, Macd, Rsi, compute_series
from tests.core.indicators.synthetic import close_bars

# -- RSI --------------------------------------------------------------------


def test_rsi_warmup_needs_one_more_bar_than_its_period() -> None:
    """n changes require n+1 closes."""
    assert Rsi(period=14).warmup == 15
    assert compute_series(Rsi(period=2), close_bars(["10", "11"])).values == (None, None)


def test_rsi_seeds_with_the_mean_of_the_first_n_changes_then_smooths_by_wilder() -> None:
    """changes +1, +1 then -1 at n=2: seed 100, then avg_gain = avg_loss = 0.5."""
    series = compute_series(Rsi(period=2), close_bars(["10", "11", "12", "11"]))

    assert series.values[:2] == (None, None)
    assert series.values[2:] == pytest.approx([100.0, 50.0])


def test_rsi_is_one_hundred_while_nothing_has_fallen() -> None:
    series = compute_series(Rsi(period=3), close_bars(["10", "11", "12", "13", "14"]))

    assert series.values[3:] == pytest.approx([100.0, 100.0])


def test_rsi_is_zero_while_nothing_has_risen() -> None:
    series = compute_series(Rsi(period=3), close_bars(["14", "13", "12", "11", "10"]))

    assert series.values[3:] == pytest.approx([0.0, 0.0])


def test_rsi_of_a_flat_series_is_the_neutral_fifty_not_none() -> None:
    """Both averages are zero: a real neutral reading, not missing data."""
    series = compute_series(Rsi(period=3), close_bars(["25"] * 6))

    assert series.values[3:] == pytest.approx([50.0] * 3)


def test_rsi_stays_inside_zero_to_one_hundred() -> None:
    closes = ["10", "12", "11", "15", "9", "9", "20", "3", "7", "7", "12", "11"]
    series = compute_series(Rsi(period=4), close_bars(closes))

    assert all(0.0 <= value <= 100.0 for value in series.values[4:] if value is not None)


# Wilder's worked example ("New Concepts in Technical Trading Systems", 1978), as
# reproduced in the StockCharts RSI reference table. Closes carry 4 decimals; the
# published RSI column carries 2, so this asserts to 0.01 absolute rather than to
# ADR 0005 D7's 1e-6 relative — the reference itself is not more precise.
WILDER_CLOSES = [
    "44.3389", "44.0902", "44.1497", "43.6124", "44.3278", "44.8264", "45.0955",
    "45.4245", "45.8433", "46.0826", "45.8931", "46.0328", "45.6140", "46.2820",
    "46.2820", "46.0028", "46.0328", "46.4116", "46.2222", "45.6439", "46.2122",
    "46.2521", "45.7137", "46.4515", "45.7835", "45.3548", "44.0288", "44.1783",
    "44.2181", "44.5672", "43.4205", "42.6628", "43.1314",
]  # fmt: skip

WILDER_RSI14 = [
    70.53, 66.32, 66.55, 69.41, 66.36, 57.97, 62.93, 63.26, 56.06, 62.38,
    54.71, 50.42, 39.99, 41.46, 41.87, 45.46, 37.30, 33.08, 37.77,
]  # fmt: skip


def test_rsi_matches_wilders_published_worked_example() -> None:
    series = compute_series(Rsi(period=14), close_bars(WILDER_CLOSES))

    assert series.values[:14] == (None,) * 14
    assert series.values[14:] == pytest.approx(WILDER_RSI14, abs=0.01)


# -- MACD -------------------------------------------------------------------


def test_macd_defaults_are_twelve_twenty_six_nine() -> None:
    indicator = Macd()

    assert (indicator.fast, indicator.slow, indicator.signal_period) == (12, 26, 9)
    assert indicator.warmup == 26
    assert indicator.signal_warmup == 34


def test_macd_line_is_the_difference_of_its_two_emas() -> None:
    closes = ["10", "12", "11", "15", "9", "14", "20", "13", "17", "17", "12", "11"]
    bars = close_bars(closes)

    macd = compute_series(Macd(fast=3, slow=5, signal=2), bars).values
    fast = compute_series(Ema(period=3), bars).values
    slow = compute_series(Ema(period=5), bars).values

    for index in range(4, len(closes)):
        point, quick, slowly = macd[index], fast[index], slow[index]
        assert point is not None
        assert quick is not None
        assert slowly is not None
        assert point.macd == pytest.approx(quick - slowly)


def test_macd_emits_its_line_before_the_signal_exists() -> None:
    """Line at slow-1; signal and histogram only at slow+signal-2."""
    closes = [str(value) for value in range(10, 30)]
    series = compute_series(Macd(fast=2, slow=3, signal=2), close_bars(closes))

    assert series.values[:2] == (None, None)

    early = series.values[2]
    assert early is not None
    assert early.signal is None
    assert early.histogram is None

    later = series.values[3]
    assert later is not None
    assert later.signal is not None
    assert later.histogram == pytest.approx(later.macd - later.signal)


def test_macd_of_a_linear_ramp_is_a_constant_lag_difference() -> None:
    """Both EMAs settle at x - (n-1)/2, so the line settles at (slow-fast)/2."""
    closes = [str(value) for value in range(1, 1001)]
    series = compute_series(Macd(fast=12, slow=26, signal=9), close_bars(closes))

    last = series.values[-1]
    assert last is not None
    assert last.macd == pytest.approx((26 - 12) / 2, rel=1e-9)
    assert last.signal == pytest.approx((26 - 12) / 2, rel=1e-9)
    assert last.histogram == pytest.approx(0.0, abs=1e-9)


def test_macd_of_a_constant_series_is_zero_throughout() -> None:
    series = compute_series(Macd(fast=2, slow=3, signal=2), close_bars(["80"] * 12))

    for value in series.values[3:]:
        assert value is not None
        assert value.macd == pytest.approx(0.0)
        assert value.signal == pytest.approx(0.0)
        assert value.histogram == pytest.approx(0.0)


def test_macd_signal_seeds_with_the_sma_of_the_first_signal_period_macd_values() -> None:
    """A ramp gives a constant MACD line, so the seed is that constant."""
    closes = [str(value) for value in range(1, 12)]
    series = compute_series(Macd(fast=2, slow=3, signal=2), close_bars(closes))

    first_signalled = series.values[3]
    assert first_signalled is not None
    assert first_signalled.signal == pytest.approx(0.5)
