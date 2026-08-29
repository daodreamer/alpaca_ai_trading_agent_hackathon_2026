"""Bar factories and a fast-warming profile for trend tests.

A trend test is usually "this shape of closes produces this state", so the
helpers take the closes and fill in everything else. `FAST` shrinks every
composed indicator to the smallest period that still exercises its logic, which
keeps a test fixture a dozen bars long instead of eighty.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, Timeframe
from alphagate.core.trend_engine import TrendConfig

AAPL: Ticker = ticker("AAPL")

SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)

FAST = TrendConfig(
    fast_period=2,
    slow_period=3,
    slope_lookback=1,
    atr_period=2,
    rsi_period=2,
    macd_fast=2,
    macd_slow=3,
    macd_signal=2,
)
"""Every period at its floor, so warmup is measured in bars, not screenfuls."""


def start_of(index: int) -> datetime:
    return SESSION_OPEN + timedelta(minutes=index)


def bar_at(
    index: int,
    *,
    open_: str,
    high: str,
    low: str,
    close: str,
    volume: str = "1000",
    is_final: bool = True,
    symbol: Ticker = AAPL,
    timeframe: Timeframe = Timeframe.M1,
    feed: Feed = Feed.SYNTHETIC,
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
) -> Bar:
    start = start_of(index)
    return Bar(
        symbol=symbol,
        timeframe=timeframe,
        start_time_utc=start,
        end_time_utc=start + timedelta(minutes=1),
        session_date=SESSION_DATE,
        session=SessionKind.REGULAR,
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


def bars_from(closes: Sequence[str | Decimal], *, margin: str = "0.5") -> tuple[Bar, ...]:
    """One bar per close, each spanning `close ± margin` and opening at its close.

    The list of closes is therefore read directly as the price path, and the
    highs and lows follow it — which is what the swing detector needs to see a
    shape at all.
    """
    step = Decimal(margin)
    bars: list[Bar] = []
    for index, close in enumerate(closes):
        middle = Decimal(close)
        bars.append(
            bar_at(
                index,
                open_=str(middle),
                high=str(middle + step),
                low=str(middle - step),
                close=str(middle),
            )
        )
    return tuple(bars)


def ramp(start: int, count: int, step: int = 1) -> tuple[str, ...]:
    """A straight line of closes — the least ambiguous trend there is."""
    return tuple(str(start + step * index) for index in range(count))


def flat(price: int, count: int) -> tuple[str, ...]:
    return tuple(str(price) for _ in range(count))
