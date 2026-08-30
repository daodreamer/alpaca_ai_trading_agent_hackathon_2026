"""Restricting intraday bars to the regular session.

Measured, not assumed. An Alpaca 15-minute pull of SPY returned 42,482 bars
across roughly 715 sessions -- 59 per session, where 09:30-16:00 holds 26. The
first bar of each day was stamped 09:00Z (04:00 in New York) and the last
23:45Z. Alpaca serves extended hours by default and has no parameter to say
otherwise, so the filtering happens here.

Why they cannot simply be left in. Pre- and post-market bars are thin and their
spreads are a multiple of the regular session's, but the backtester treats every
bar alike: its next-bar-open fill would take a 100-share 04:15 print as a price
anyone could have transacted at, and the annualisation factor inferred from bar
spacing would count those bars as tradeable time. A strategy tested on them is
measured against a market that was not open.

IBKR offers ``useRTH`` and the IBKR adapter sets it. This is the same decision,
applied where the vendor does not offer it.

The session is computed from the US eastern timezone rather than a fixed UTC
offset, because the offset moves twice a year: 09:30 New York is 14:30Z in
January and 13:30Z in July. A hard-coded offset is right for half the year,
which is the worst kind of wrong -- it silently deletes the first half-hour of
every summer session and admits the last half-hour of every summer evening.

Half-days -- the 13:00 closes around Thanksgiving and Christmas -- are not
modelled. Keeping bars from an early-closed afternoon admits a handful of
genuinely quiet ones; excluding a whole real session would be worse, and neither
needs a holiday calendar to be maintained for a research cache.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np

from aqr.data.bars import Bars

__all__ = ["EXCHANGE_TZ", "REGULAR_CLOSE_ET", "REGULAR_OPEN_ET", "regular_hours_only"]

EXCHANGE_TZ = ZoneInfo("America/New_York")

REGULAR_OPEN_ET = (9, 30)
REGULAR_CLOSE_ET = (16, 0)

# Timeframes with no time of day to be inside or outside a session. Filtering a
# daily bar on its stamp would delete the entire series.
_SESSIONLESS = frozenset({"1D", "1W"})


def _minutes(hour_minute: tuple[int, int]) -> int:
    return hour_minute[0] * 60 + hour_minute[1]


def regular_hours_only(bars: Bars) -> Bars:
    """Drop bars whose period begins outside 09:30-16:00 New York.

    A bar is kept when it *starts* inside the session. The 15-minute bar stamped
    at the close covers 16:00-16:15, which is after the bell, so the last bar
    kept is the one starting at 15:45.
    """
    if bars.timeframe in _SESSIONLESS or len(bars) == 0:
        return bars

    open_at, close_at = _minutes(REGULAR_OPEN_ET), _minutes(REGULAR_CLOSE_ET)
    keep = np.fromiter(
        (
            open_at
            <= _minute_of_day(int(stamp))
            < close_at
            for stamp in bars.event_time
        ),
        dtype=bool,
        count=len(bars),
    )
    index = np.flatnonzero(keep)
    if index.size == len(bars):
        return bars
    return Bars(
        symbol=bars.symbol,
        timeframe=bars.timeframe,
        event_time=bars.event_time[index],
        open=bars.open[index],
        high=bars.high[index],
        low=bars.low[index],
        close=bars.close[index],
        volume=bars.volume[index],
        available_time=bars.available_time[index],
    )


def _minute_of_day(epoch_seconds: int) -> int:
    """Minutes past midnight in New York, honouring daylight saving.

    Converted per bar rather than by a fixed offset: 09:30 New York is 14:30Z in
    January and 13:30Z in July, and a hard-coded offset is correct for half the
    year -- which silently deletes the first half-hour of every summer session
    and admits the last half-hour of every summer evening.
    """
    local = datetime.fromtimestamp(epoch_seconds, tz=UTC).astimezone(EXCHANGE_TZ)
    return local.hour * 60 + local.minute
