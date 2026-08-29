"""Bar factories for structure tests.

A structure test is usually "this shape of highs and lows produces these swings",
so the helpers take the shape and fill in everything else. `pivot_bars` builds
bars whose high/low straddle a given centre by a fixed margin, which keeps a
zig-zag readable as a list of numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")

SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)


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


def swing_bars(
    centres: Sequence[str | Decimal],
    *,
    margin: str = "1",
    volumes: Sequence[str] | None = None,
) -> tuple[Bar, ...]:
    """Bars shaped around a path of centre prices.

    Each bar spans `centre ± margin`, opens and closes at its centre. A list of
    centres is therefore read directly as the swing path: the highs follow the
    centres, and so do the lows.
    """
    step = Decimal(margin)
    bars: list[Bar] = []
    for index, centre in enumerate(centres):
        middle = Decimal(centre)
        volume = volumes[index] if volumes is not None else "1000"
        bars.append(
            bar_at(
                index,
                open_=str(middle),
                high=str(middle + step),
                low=str(middle - step),
                close=str(middle),
                volume=volume,
            )
        )
    return tuple(bars)


def closing_bars(closes: Sequence[str], *, volumes: Sequence[str] | None = None) -> tuple[Bar, ...]:
    """Bars with no range at all: open == high == low == close.

    For break tests, where what matters is where the close lands and nothing
    should accidentally count as a wick.
    """
    return tuple(
        bar_at(
            index,
            open_=close,
            high=close,
            low=close,
            close=close,
            volume=volumes[index] if volumes is not None else "1000",
        )
        for index, close in enumerate(closes)
    )
