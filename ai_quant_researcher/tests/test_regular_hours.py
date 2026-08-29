"""Keeping intraday bars inside the regular session.

Measured, not assumed. An Alpaca 15-minute pull of SPY returned 42,482 bars
across roughly 715 sessions -- 59 per session, where a 9:30-16:00 session holds
26. The first bar of each day was stamped 09:00Z (04:00 New York) and the last
23:45Z (19:45). Alpaca serves extended hours by default and offers no flag to
say otherwise, so the filtering has to happen here.

Why it cannot be left in. Pre- and post-market bars are thin, their spreads are
several times the regular session's, and the backtester's next-bar-open fill
treats a 100-share 04:15 print as a price anyone could have transacted at. A
strategy tested on those bars is measured against a market that was not open.

IBKR has ``useRTH``, which the IBKR adapter already sets. This is the same
decision, applied where the vendor does not offer it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from aqr.data.bars import Bars
from aqr.data.sessions import REGULAR_CLOSE_ET, REGULAR_OPEN_ET, regular_hours_only


def _intraday(times_utc: list[str], timeframe: str = "15m") -> Bars:
    stamps = np.array(
        [int(datetime.fromisoformat(t).replace(tzinfo=UTC).timestamp()) for t in times_utc],
        dtype=np.int64,
    )
    n = len(times_utc)
    return Bars(
        symbol="SPY",
        timeframe=timeframe,
        event_time=stamps,
        open=np.full(n, 100.0),
        high=np.full(n, 101.0),
        low=np.full(n, 99.0),
        close=np.full(n, 100.5),
        volume=np.full(n, 1e5),
    )


class TestWinterSession:
    """November to March, New York is UTC-5: the session is 14:30-21:00Z."""

    def test_a_premarket_bar_is_dropped(self) -> None:
        bars = _intraday(["2024-01-02T09:00:00", "2024-01-02T15:00:00"])
        assert len(regular_hours_only(bars)) == 1

    def test_an_after_hours_bar_is_dropped(self) -> None:
        bars = _intraday(["2024-01-02T15:00:00", "2024-01-02T23:45:00"])
        kept = regular_hours_only(bars)
        assert len(kept) == 1
        assert kept.timestamps[0].hour == 15

    def test_the_opening_bar_is_kept(self) -> None:
        # 14:30Z is 09:30 New York in winter: the first bar of the session.
        assert len(regular_hours_only(_intraday(["2024-01-02T14:30:00"]))) == 1

    def test_the_bar_starting_at_the_close_is_dropped(self) -> None:
        """A bar stamped 21:00Z covers 16:00-16:15 New York, which is after the
        bell. The last bar of the session starts at 20:45."""
        bars = _intraday(["2024-01-02T20:45:00", "2024-01-02T21:00:00"])
        kept = regular_hours_only(bars)
        assert len(kept) == 1
        assert kept.timestamps[0].hour == 20


class TestSummerSession:
    """March to November, New York is UTC-4: the session is 13:30-20:00Z."""

    def test_the_session_shifts_with_daylight_saving(self) -> None:
        # 13:30Z is 09:30 New York in July, and 08:30 in January.
        july = _intraday(["2024-07-02T13:30:00"])
        january = _intraday(["2024-01-02T13:30:00"])
        assert len(regular_hours_only(july)) == 1
        assert len(regular_hours_only(january)) == 0

    def test_the_summer_close_is_an_hour_earlier_in_utc(self) -> None:
        bars = _intraday(["2024-07-02T19:45:00", "2024-07-02T20:00:00"])
        assert len(regular_hours_only(bars)) == 1


class TestSessionShape:
    def test_a_full_day_of_fifteen_minute_bars_leaves_twenty_six(self) -> None:
        """6.5 hours divided by 15 minutes. The number Alpaca should have
        returned instead of 59."""
        stamps = [f"2024-01-02T{h:02d}:{m:02d}:00" for h in range(9, 24) for m in (0, 15, 30, 45)]
        kept = regular_hours_only(_intraday(stamps))
        assert len(kept) == 26

    def test_a_full_day_of_hourly_bars_leaves_seven(self) -> None:
        # 09:30-16:00 touches seven hourly buckets: 09:30 and 10:00 through 15:00.
        stamps = [f"2024-01-02T{h:02d}:30:00" for h in range(9, 24)]
        assert len(regular_hours_only(_intraday(stamps, "1h"))) == 7


class TestDailyBarsAreLeftAlone:
    def test_a_daily_series_passes_through_unchanged(self) -> None:
        """A daily bar has no time of day to be inside or outside a session, and
        filtering one on its stamp would delete the entire series."""
        stamps = ["2024-01-02T00:00:00", "2024-01-03T00:00:00", "2024-01-04T00:00:00"]
        bars = _intraday(stamps, "1D")
        assert len(regular_hours_only(bars)) == 3

    def test_a_weekly_series_passes_through_unchanged(self) -> None:
        bars = _intraday(["2024-01-02T00:00:00", "2024-01-09T00:00:00"], "1W")
        assert len(regular_hours_only(bars)) == 2


class TestPreservation:
    def test_prices_and_volumes_follow_their_bars(self) -> None:
        bars = _intraday(["2024-01-02T09:00:00", "2024-01-02T15:00:00"])
        kept = regular_hours_only(bars)
        assert kept.close[0] == pytest.approx(100.5)
        assert kept.symbol == "SPY"
        assert kept.timeframe == "15m"

    def test_point_in_time_stamps_are_kept(self) -> None:
        bars = _intraday(["2024-01-02T09:00:00", "2024-01-02T15:00:00"])
        kept = regular_hours_only(bars)
        assert kept.available_time.size == len(kept)

    def test_an_empty_series_stays_empty(self) -> None:
        empty = _intraday(["2024-01-02T15:00:00"]).slice(0, 0)
        assert len(regular_hours_only(empty)) == 0


class TestConstants:
    def test_the_session_is_the_new_york_equity_session(self) -> None:
        assert REGULAR_OPEN_ET == (9, 30)
        assert REGULAR_CLOSE_ET == (16, 0)
