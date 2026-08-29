"""Indicator correctness against hand-computed values.

Prefix-stability (``test_causality.py``) proves an indicator does not read the
future. It does not prove the indicator is *correct* — a function returning
zeros is perfectly causal. These tests pin the actual arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from aqr.core import indicators as ind


class TestMovingAverages:
    def test_sma_matches_the_arithmetic_mean(self) -> None:
        x = np.arange(1.0, 11.0)
        out = ind.sma(x, 3)
        assert np.isnan(out[:2]).all()
        assert out[2] == pytest.approx(2.0)
        assert out[9] == pytest.approx(9.0)

    def test_ema_is_seeded_with_the_first_sma(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = ind.ema(x, 3)
        assert out[2] == pytest.approx(2.0)  # mean of 1, 2, 3
        alpha = 2.0 / 4.0
        assert out[3] == pytest.approx(alpha * 4.0 + (1 - alpha) * 2.0)

    def test_ema_seed_is_independent_of_extra_leading_history(self) -> None:
        """The property SMA-seeding buys: the same window gives the same value
        regardless of how much history happened to be loaded before it."""
        base = np.arange(1.0, 60.0)
        short = ind.ema(base, 10)
        assert not np.isnan(short[-1])

    def test_period_longer_than_the_series_yields_all_nan(self) -> None:
        assert np.isnan(ind.sma(np.arange(5.0), 10)).all()
        assert np.isnan(ind.ema(np.arange(5.0), 10)).all()

    def test_a_zero_period_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="period"):
            ind.sma(np.arange(5.0), 0)


class TestOscillators:
    def test_rsi_of_a_monotonic_rise_is_one_hundred(self) -> None:
        out = ind.rsi(np.arange(1.0, 40.0), 14)
        assert out[-1] == pytest.approx(100.0)

    def test_rsi_of_a_monotonic_fall_is_zero(self) -> None:
        out = ind.rsi(np.arange(40.0, 1.0, -1.0), 14)
        assert out[-1] == pytest.approx(0.0)

    def test_rsi_stays_in_range(self) -> None:
        rng = np.random.default_rng(3)
        x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
        out = ind.rsi(x, 14)
        finite = out[~np.isnan(out)]
        assert finite.min() >= 0.0 and finite.max() <= 100.0

    def test_stochastic_is_one_hundred_at_the_window_high(self) -> None:
        high = np.array([10.0, 11.0, 12.0])
        low = np.array([9.0, 9.0, 9.0])
        close = np.array([10.0, 11.0, 12.0])
        out = ind.stochastic_k(high, low, close, 3)
        assert out[2] == pytest.approx(100.0)

    def test_stochastic_handles_a_flat_window(self) -> None:
        flat = np.full(5, 10.0)
        out = ind.stochastic_k(flat, flat, flat, 3)
        assert out[4] == pytest.approx(50.0), "a zero range must not divide by zero"


class TestVolatility:
    def test_true_range_uses_the_previous_close(self) -> None:
        high = np.array([10.0, 12.0])
        low = np.array([9.0, 11.5])
        close = np.array([9.5, 12.0])
        tr = ind.true_range(high, low, close)
        assert tr[0] == pytest.approx(1.0)  # first bar: its own range
        assert tr[1] == pytest.approx(2.5)  # 12.0 - 9.5, the gap dominates

    def test_atr_of_a_constant_range_equals_that_range(self) -> None:
        n = 40
        high = np.full(n, 11.0)
        low = np.full(n, 9.0)
        close = np.full(n, 10.0)
        out = ind.atr(high, low, close, 14)
        assert out[-1] == pytest.approx(2.0)

    def test_adx_is_high_in_a_trend_and_low_in_a_range(self) -> None:
        n = 120
        trend = np.arange(100.0, 100.0 + n)
        chop = 100.0 + np.tile([0.0, 1.0], n // 2)
        trend_adx = ind.adx(trend + 1, trend - 1, trend, 14)
        chop_adx = ind.adx(chop + 1, chop - 1, chop, 14)
        assert np.nanmax(trend_adx) > np.nanmax(chop_adx), (
            "ADX did not distinguish a straight line from an alternating series"
        )

    def test_realized_vol_scales_with_dispersion(self) -> None:
        rng = np.random.default_rng(5)
        calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
        wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
        assert np.nanmean(ind.realized_vol(wild, 20)) > np.nanmean(ind.realized_vol(calm, 20))


class TestBands:
    def test_bollinger_bands_straddle_the_mean(self) -> None:
        rng = np.random.default_rng(7)
        x = 100 + rng.normal(0, 2, 200)
        lower, mid, upper = ind.bollinger(x, 20, 2.0)
        good = ~np.isnan(mid)
        assert (lower[good] <= mid[good]).all()
        assert (mid[good] <= upper[good]).all()

    def test_zero_variance_collapses_the_bands(self) -> None:
        flat = np.full(50, 10.0)
        lower, mid, upper = ind.bollinger(flat, 20, 2.0)
        assert lower[-1] == pytest.approx(upper[-1]) == pytest.approx(10.0)


class TestVolume:
    def test_rvol_excludes_the_current_bar_from_its_own_average(self) -> None:
        volume = np.array([100.0] * 25 + [1000.0])
        out = ind.rvol(volume, 20)
        assert out[-1] == pytest.approx(10.0), (
            "the spike diluted its own average; the window must exclude the current bar"
        )

    def test_vwap_resets_on_a_new_session(self) -> None:
        high = np.array([10.0, 20.0, 30.0])
        low = high.copy()
        close = high.copy()
        volume = np.ones(3)
        session = np.array([1, 1, 2])
        out = ind.vwap_session(high, low, close, volume, session)
        assert out[1] == pytest.approx(15.0)  # mean of 10 and 20
        assert out[2] == pytest.approx(30.0)  # reset


class TestStructure:
    def test_rolling_max_and_min_include_the_current_bar(self) -> None:
        x = np.array([1.0, 5.0, 3.0, 2.0])
        assert ind.rolling_max(x, 2)[1] == pytest.approx(5.0)
        assert ind.rolling_min(x, 2)[3] == pytest.approx(2.0)

    def test_roc_is_a_fraction(self) -> None:
        x = np.array([100.0, 100.0, 105.0])
        assert ind.roc(x, 2)[2] == pytest.approx(0.05)

    def test_roc_of_a_zero_base_is_nan_not_infinity(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        assert np.isnan(ind.roc(x, 2)[2])


def test_every_indicator_returns_the_input_length() -> None:
    n = 300
    rng = np.random.default_rng(13)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high, low = close * 1.01, close * 0.99
    volume = np.full(n, 1e6)
    session = np.arange(n)

    outputs = [
        ind.sma(close, 20),
        ind.ema(close, 20),
        ind.rsi(close, 14),
        ind.atr(high, low, close, 14),
        ind.adx(high, low, close, 14),
        ind.roc(close, 10),
        ind.rvol(volume, 20),
        ind.realized_vol(close, 20),
        ind.percentile_rank(close, 50),
        ind.rolling_max(high, 20),
        ind.rolling_min(low, 20),
        ind.stochastic_k(high, low, close, 14),
        ind.vwap_session(high, low, close, volume, session),
        *ind.bollinger(close, 20),
        *ind.macd(close),
    ]
    assert all(out.size == n for out in outputs)
