"""Bars and candidates for level-engine tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.level_engine import LevelCandidate
from alphagate.core.levels import LevelKind, LevelSource
from alphagate.core.structure import SwingKind, SwingPoint, SwingStatus
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")

SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)


def start_of(index: int) -> datetime:
    return SESSION_OPEN + timedelta(minutes=index)


def minute_bar(
    index: int,
    *,
    high: str,
    low: str,
    close: str | None = None,
    volume: str = "1000",
    is_final: bool = True,
) -> Bar:
    start = start_of(index)
    settle = close if close is not None else low
    return Bar(
        symbol=AAPL,
        timeframe=Timeframe.M1,
        start_time_utc=start,
        end_time_utc=start + timedelta(minutes=1),
        session_date=SESSION_DATE,
        session=SessionKind.REGULAR,
        open=low,
        high=high,
        low=low,
        close=settle,
        volume=volume,
        source="test",
        feed=Feed.SYNTHETIC,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        is_final=is_final,
    )


def daily_bar(day: date, *, high: str, low: str, is_final: bool = True) -> Bar:
    open_utc = datetime(day.year, day.month, day.day, 13, 30, tzinfo=UTC)
    return Bar(
        symbol=AAPL,
        timeframe=Timeframe.D1,
        start_time_utc=open_utc,
        end_time_utc=open_utc + timedelta(hours=6, minutes=30),
        session_date=day,
        session=SessionKind.REGULAR,
        open=low,
        high=high,
        low=low,
        close=high,
        volume="1000000",
        source="test",
        feed=Feed.SYNTHETIC,
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        is_final=is_final,
    )


def daily_series(rows: Sequence[tuple[date, str, str]]) -> tuple[Bar, ...]:
    return tuple(daily_bar(day, high=high, low=low) for day, high, low in rows)


def swing(
    index: int,
    *,
    price: str,
    kind: SwingKind = SwingKind.HIGH,
    status: SwingStatus = SwingStatus.CONFIRMED,
) -> SwingPoint:
    """A pivot at minute `index`, detected two bars later.

    Two bars rather than one because a confirmed swing cannot be detected on its
    own pivot bar (ADR 0008 D3), and the helper should not be able to build one
    that the domain would refuse.
    """
    return SwingPoint(
        kind=kind,
        status=status,
        bar_start_utc=start_of(index),
        detected_at_utc=start_of(index + 2),
        price=price,
    )


def candidate(
    price: str,
    *,
    kind: LevelKind = LevelKind.RESISTANCE,
    source: LevelSource = LevelSource.SWING_HIGH,
    index: int = 0,
) -> LevelCandidate:
    return LevelCandidate(price=price, kind=kind, source=source, observed_at_utc=start_of(index))
