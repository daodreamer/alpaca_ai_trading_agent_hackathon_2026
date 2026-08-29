"""SMA and EMA — ADR 0007 D5.

Every expectation here is hand-calculable from the stated formula; the numbers
are written out rather than derived in the test, so a change in the averaging
convention fails loudly instead of quietly agreeing with itself.
"""

from __future__ import annotations

import pytest

from alphagate.core.indicators import Ema, PriceSource, Sma, compute_series
from tests.core.indicators.synthetic import close_bars, ohlc_bars

# -- SMA --------------------------------------------------------------------


def test_sma_is_the_arithmetic_mean_of_the_last_n_closes() -> None:
    series = compute_series(Sma(period=3), close_bars(["1", "2", "3", "4", "5"]))

    assert series.values[:2] == (None, None)
    assert series.values[2:] == pytest.approx([2.0, 3.0, 4.0])


def test_sma_warmup_is_the_period() -> None:
    assert Sma(period=3).warmup == 3
    assert compute_series(Sma(period=3), close_bars(["1", "2"])).values == (None, None)


def test_sma_of_a_constant_series_is_that_constant() -> None:
    series = compute_series(Sma(period=4), close_bars(["50"] * 10))

    assert series.values[3:] == pytest.approx([50.0] * 7)


def test_sma_of_a_linear_ramp_lags_by_half_the_window() -> None:
    """For x[i] = i the mean of the last n values is exactly i - (n-1)/2."""
    closes = [str(value) for value in range(1, 31)]
    series = compute_series(Sma(period=5), close_bars(closes))

    assert series.values[4:] == pytest.approx([float(i) - 2.0 for i in range(5, 31)])


def test_sma_reads_the_configured_price_source() -> None:
    bars = ohlc_bars([("10", "14", "8", "12"), ("12", "16", "10", "14")])
    series = compute_series(Sma(period=2, source=PriceSource.HL2), bars)

    assert series.values[1] == pytest.approx((11.0 + 13.0) / 2)


def test_sma_period_one_is_the_price_itself() -> None:
    series = compute_series(Sma(period=1), close_bars(["7", "9", "8"]))

    assert series.values == pytest.approx([7.0, 9.0, 8.0])


# -- EMA --------------------------------------------------------------------


def test_ema_seeds_with_the_sma_of_the_first_period_values() -> None:
    """alpha = 2/(3+1) = 0.5; seed = mean(1,2,3) = 2."""
    series = compute_series(Ema(period=3), close_bars(["1", "2", "3", "10", "10"]))

    assert series.values[:2] == (None, None)
    assert series.values[2:] == pytest.approx([2.0, 6.0, 8.0])


def test_ema_warmup_is_the_period() -> None:
    assert Ema(period=20).warmup == 20
    assert compute_series(Ema(period=3), close_bars(["1", "2"])).values == (None, None)


def test_ema_differs_from_sma_once_it_leaves_the_seed() -> None:
    closes = ["1", "2", "3", "10", "10"]
    ema = compute_series(Ema(period=3), close_bars(closes)).values
    sma = compute_series(Sma(period=3), close_bars(closes)).values

    assert ema[2] == pytest.approx(sma[2])  # the seed, by construction
    assert ema[3] != pytest.approx(sma[3])


def test_ema_of_a_constant_series_is_that_constant() -> None:
    series = compute_series(Ema(period=5), close_bars(["42"] * 30))

    assert series.values[4:] == pytest.approx([42.0] * 26)


def test_ema_of_a_linear_ramp_converges_to_the_documented_lag() -> None:
    """For x[i] = i the EMA settles at i - (1 - alpha)/alpha = i - (n-1)/2."""
    closes = [str(value) for value in range(1, 201)]
    series = compute_series(Ema(period=10), close_bars(closes))

    assert series.values[-1] == pytest.approx(200.0 - 4.5, rel=1e-9)


def test_ema_period_one_is_the_price_itself() -> None:
    """alpha = 2/2 = 1, so the recursion keeps no history."""
    series = compute_series(Ema(period=1), close_bars(["7", "9", "8"]))

    assert series.values == pytest.approx([7.0, 9.0, 8.0])


def test_ema_responds_to_a_step_geometrically() -> None:
    """After a step the remaining error halves every bar at alpha = 0.5."""
    series = compute_series(Ema(period=3), close_bars(["1", "1", "1", "9", "9", "9", "9"]))

    assert series.values[2:] == pytest.approx([1.0, 5.0, 7.0, 8.0, 8.5])
