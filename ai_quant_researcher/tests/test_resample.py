from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from aqr.data.bars import Bars
from aqr.data.resample import aggregate_bars


def _hourly(n: int, *, start: datetime | None = None) -> Bars:
    start = start or datetime(2024, 1, 2, 15, tzinfo=UTC)
    stamps = [int((start + timedelta(hours=i)).timestamp()) for i in range(n)]
    close = np.arange(100.0, 100.0 + n)
    return Bars(
        symbol="TEST",
        timeframe="1h",
        event_time=np.array(stamps, dtype=np.int64),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.ones(n) * 1_000,
    )


def test_four_hour_bars_aggregate_four_consecutive_hours() -> None:
    bars = _hourly(8)
    out = aggregate_bars(bars, target_timeframe="4h", factor=4)
    assert len(out) == 2
    assert out.timeframe == "4h"
    assert out.open[0] == 100.0
    assert out.close[0] == 103.0
    assert out.high[0] == 104.0
    assert out.low[0] == 99.0
    assert out.volume[0] == 4_000


def test_overnight_break_starts_a_new_group() -> None:
    day1 = _hourly(4)
    day2 = _hourly(4, start=datetime(2024, 1, 3, 15, tzinfo=UTC))
    merged = Bars(
        symbol="TEST",
        timeframe="1h",
        event_time=np.concatenate([day1.event_time, day2.event_time]),
        open=np.concatenate([day1.open, day2.open]),
        high=np.concatenate([day1.high, day2.high]),
        low=np.concatenate([day1.low, day2.low]),
        close=np.concatenate([day1.close, day2.close]),
        volume=np.concatenate([day1.volume, day2.volume]),
    )
    out = aggregate_bars(merged, target_timeframe="4h", factor=4)
    assert len(out) == 2


def test_trailing_bars_of_a_session_are_kept_as_a_short_final_bar() -> None:
    """A US session is 6 hourly bars, which does not divide by 4.

    Dropping the remainder would discard 19:00 and 20:00 UTC every day -- the
    last two hours of trading, including the close -- and the 4h close would
    silently be a 14:00 ET price instead of the session close.
    """
    session = _hourly(6)
    out = aggregate_bars(session, target_timeframe="4h", factor=4)
    assert len(out) == 2
    # First bar is the full group of four.
    assert out.open[0] == 100.0
    assert out.close[0] == 103.0
    # Second bar is the two-hour remainder, and it carries the session close.
    assert out.open[1] == 104.0
    assert out.close[1] == 105.0
    assert out.volume[1] == 2_000
    assert out.high[1] == 106.0
    assert out.low[1] == 103.0


def test_no_bar_is_dropped_across_two_six_bar_sessions() -> None:
    day1 = _hourly(6)
    day2 = _hourly(6, start=datetime(2024, 1, 3, 15, tzinfo=UTC))
    merged = Bars(
        symbol="TEST",
        timeframe="1h",
        event_time=np.concatenate([day1.event_time, day2.event_time]),
        open=np.concatenate([day1.open, day2.open]),
        high=np.concatenate([day1.high, day2.high]),
        low=np.concatenate([day1.low, day2.low]),
        close=np.concatenate([day1.close, day2.close]),
        volume=np.concatenate([day1.volume, day2.volume]),
    )
    out = aggregate_bars(merged, target_timeframe="4h", factor=4)
    assert len(out) == 4
    # Every source bar is accounted for exactly once.
    assert out.volume.sum() == merged.volume.sum()
    # A group never spans the overnight break.
    assert out.close[1] == day1.close[-1]
    assert out.open[2] == day2.open[0]
