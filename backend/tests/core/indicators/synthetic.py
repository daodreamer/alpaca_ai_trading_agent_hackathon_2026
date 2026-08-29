"""Bar factories for indicator tests.

Indicators are bar-indexed (ADR 0007 D3), so these build a contiguous run of 1m
bars and nothing else: no calendar, no session grid. Prices are passed as text
because that is how a price enters the exact domain (ADR 0005 D2).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")

SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)

Ohlc = tuple[str, str, str, str]
"""(open, high, low, close), as text."""


def bar_at(
    index: int,
    ohlc: Ohlc,
    *,
    volume: str = "1000",
    session_date: date = SESSION_DATE,
    session: SessionKind = SessionKind.REGULAR,
    is_final: bool = True,
    symbol: Ticker = AAPL,
    timeframe: Timeframe = Timeframe.M1,
    feed: Feed = Feed.SYNTHETIC,
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
) -> Bar:
    """One bar, `index` minutes after the session open."""
    start = SESSION_OPEN + timedelta(minutes=index)
    open_, high, low, close = ohlc
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        start_time_utc=start,
        end_time_utc=start + timedelta(minutes=1),
        session_date=session_date,
        session=session,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        source="test",
        feed=feed,
        adjustment_mode=adjustment_mode,
        is_final=is_final,
    )


def close_bars(
    closes: Sequence[str],
    *,
    volume: str = "1000",
    session_date: date = SESSION_DATE,
    session: SessionKind = SessionKind.REGULAR,
) -> tuple[Bar, ...]:
    """A run of flat bars whose only interesting field is the close.

    open == high == low == close, which is a legal bar and keeps the arithmetic
    of a close-driven indicator visible in the test data.
    """
    return tuple(
        bar_at(
            index,
            (close, close, close, close),
            volume=volume,
            session_date=session_date,
            session=session,
        )
        for index, close in enumerate(closes)
    )


def ohlc_bars(rows: Sequence[Ohlc], *, volume: str = "1000") -> tuple[Bar, ...]:
    return tuple(bar_at(index, row, volume=volume) for index, row in enumerate(rows))
