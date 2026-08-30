"""The synthetic generator annualises against the timeframe it is asked for.

The regime table holds *annual* drifts and vols, scaled to per-bar steps by
``1 / annual_bars``. A hard-coded 252 on hourly bars gives every bar a day's
worth of variance -- 6.5 times too much per bar, 2.55 times too much once
re-annualised -- and every vol-sensitive feature tested on synthetic intraday
data would have been calibrated against a market that does not exist.

``None`` now derives the factor from ``bars_per_year(timeframe)``; on daily
bars that is exactly 252 and the generated series is bit-for-bit the old one.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aqr.data.bars import bars_per_year
from aqr.data.providers import SyntheticProvider

START = datetime(2015, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 1, tzinfo=UTC)


def test_daily_bars_are_unchanged() -> None:
    default = SyntheticProvider().load("SPY", START, END, "1D")
    explicit = SyntheticProvider(annual_bars=252).load("SPY", START, END, "1D")
    np.testing.assert_array_equal(default.close, explicit.close)
    np.testing.assert_array_equal(default.event_time, explicit.event_time)


def test_intraday_bars_carry_intraday_sized_variance() -> None:
    hourly = SyntheticProvider().load("SPY", START, END, "1h")
    logret = np.diff(np.log(hourly.close))
    annualised = float(logret.std()) * np.sqrt(bars_per_year("1h"))

    # The regime table spans 9%..32% annual vol; a mixture lands inside it.
    assert 0.07 < annualised < 0.40, (
        f"annualised vol {annualised:.3f} is outside the regime table -- "
        "the per-bar scaling is wrong for this timeframe"
    )


def test_an_explicit_annual_bars_still_wins() -> None:
    bars = SyntheticProvider(annual_bars=100).load("SPY", START, END, "1h")
    logret = np.diff(np.log(bars.close))
    annualised = float(logret.std()) * np.sqrt(100.0)
    assert 0.07 < annualised < 0.40


def test_the_regime_labels_cover_the_same_bars() -> None:
    provider = SyntheticProvider()
    bars = provider.load("SPY", START, END, "4h")
    assert len(provider.regimes("SPY", START, END, "4h")) == len(bars)


@pytest.mark.parametrize("timeframe", ["1D", "4h", "1h"])
def test_every_timeframe_generates(timeframe: str) -> None:
    assert len(SyntheticProvider().load("SPY", START, END, timeframe)) > 0
