"""Factories for domain tests.

Deliberately thin: they fill in valid defaults so each test states only the
field it is actually exercising. They are not builders with behaviour.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.identifiers import Ticker, ticker
from alphagate.core.time_model import SessionKind, Timeframe

AAPL: Ticker = ticker("AAPL")

# 2026-08-13 is a Thursday; US regular session 13:30–20:00 UTC (EDT).
SESSION_DATE = date(2026, 8, 13)
SESSION_OPEN = datetime(2026, 8, 13, 13, 30, tzinfo=UTC)
SESSION_CLOSE = datetime(2026, 8, 13, 20, 0, tzinfo=UTC)


def make_bar(**overrides: Any) -> Bar:
    defaults: dict[str, Any] = {
        "symbol": AAPL,
        "timeframe": Timeframe.H1,
        "start_time_utc": SESSION_OPEN,
        "end_time_utc": SESSION_OPEN + timedelta(hours=1),
        "session_date": SESSION_DATE,
        "session": SessionKind.REGULAR,
        "open": "100.00",
        "high": "105.00",
        "low": "99.00",
        "close": "104.00",
        "volume": "1000",
        "source": "fake",
        "feed": Feed.SYNTHETIC,
        "adjustment_mode": AdjustmentMode.UNADJUSTED,
        "is_final": True,
    }
    return Bar(**{**defaults, **overrides})
