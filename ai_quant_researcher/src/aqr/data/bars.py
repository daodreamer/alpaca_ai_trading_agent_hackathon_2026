"""Point-in-time OHLCV containers.

Every record carries two timestamps, per the architecture's look-ahead section:

``event_time``      when the bar closed in the market.
``available_time``  when our system could first have known about it.

For exchange bars the two are the same modulo feed latency. For anything
restated after the fact -- fundamentals, revised macro prints, late-corrected
bars -- they differ, and only ``available_time`` may drive a decision. The
loader filters on ``available_time`` and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from aqr.seal import current as current_seal

Array = NDArray[np.float64]

_FIELDS = ("open", "high", "low", "close", "volume")


def ensure_utc(ts: datetime) -> datetime:
    """Reject naive datetimes rather than guessing a timezone."""
    if ts.tzinfo is None:
        raise ValueError(f"timestamp must be tz-aware: {ts!r}")
    return ts.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Bars:
    """An immutable, chronologically sorted OHLCV series for one symbol.

    Arrays are parallel and equal-length. ``available_time`` defaults to
    ``event_time`` (an exchange bar is knowable when it closes) but may be
    supplied explicitly for delayed or restated sources.
    """

    symbol: str
    timeframe: str
    event_time: NDArray[np.int64]
    open: Array
    high: Array
    low: Array
    close: Array
    volume: Array
    available_time: NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    """When each bar became knowable. Defaults to ``event_time``: an exchange bar
    is knowable when it closes. Supply it explicitly for delayed or restated
    sources, where the two genuinely differ."""

    def __post_init__(self) -> None:
        n = self.event_time.size
        for name in _FIELDS:
            arr = getattr(self, name)
            if arr.size != n:
                raise ValueError(f"{self.symbol}: {name} has {arr.size} rows, expected {n}")
        if self.available_time.size == 0 and n:
            object.__setattr__(self, "available_time", self.event_time.copy())
        if self.available_time.size != n:
            raise ValueError(f"{self.symbol}: available_time has {self.available_time.size} rows")
        if n and np.any(np.diff(self.event_time) <= 0):
            raise ValueError(f"{self.symbol}: event_time must be strictly increasing")
        if n and np.any(self.available_time < self.event_time):
            raise ValueError(f"{self.symbol}: a bar cannot be available before it happened")
        if np.any(self.high < self.low):
            raise ValueError(f"{self.symbol}: high < low")
        # The provenance sensor. Here rather than in the providers because this
        # is the one chokepoint every series must pass through, including the
        # slices walk-forward carves and any provider not yet written. See
        # ``aqr.seal``.
        current_seal().observe(self.symbol, self.event_time)

    def __len__(self) -> int:
        return int(self.event_time.size)

    @property
    def timestamps(self) -> list[datetime]:
        return [datetime.fromtimestamp(int(t), tz=UTC) for t in self.event_time]

    @property
    def session(self) -> NDArray[np.int64]:
        """Date ordinal per bar, for session-scoped aggregates such as VWAP."""
        return (self.event_time // 86_400).astype(np.int64)

    def slice(self, start: int, stop: int) -> Bars:
        """Positional slice. Used by walk-forward to carve folds."""
        return Bars(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_time=self.event_time[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
            available_time=self.available_time[start:stop],
        )

    def as_of(self, when: datetime) -> Bars:
        """The series as it was knowable at ``when``.

        Filters on ``available_time``, never on ``event_time``. A bar that had
        happened but had not yet reached us is correctly excluded.
        """
        cutoff = int(ensure_utc(when).timestamp())
        keep = self.available_time <= cutoff
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            return self.slice(0, 0)
        return Bars(
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_time=self.event_time[idx],
            open=self.open[idx],
            high=self.high[idx],
            low=self.low[idx],
            close=self.close[idx],
            volume=self.volume[idx],
            available_time=self.available_time[idx],
        )

    def index_of(self, when: datetime) -> int:
        """Index of the first bar at or after ``when``; ``len(self)`` if none."""
        return self.index_of_ts(int(ensure_utc(when).timestamp()))

    def index_of_ts(self, epoch_seconds: int) -> int:
        """Index of the first bar at or after a raw UTC epoch timestamp."""
        return int(np.searchsorted(self.event_time, epoch_seconds, side="left"))


def bars_per_year(timeframe: str) -> float:
    """Trading bars in a year, for costs that accrue over calendar time.

    Derived from the trading calendar, not from 365 days: a daily strategy holds
    a position for 252 bars a year, and financing a short for "20 bars" means 20
    trading days, not 20 calendar days.
    """
    per_session = {
        "1m": 390.0,
        "5m": 78.0,
        "15m": 26.0,
        "30m": 13.0,
        "1h": 6.5,
        "4h": 1.625,
        "1D": 1.0,
        "1W": 0.2,
    }
    if timeframe not in per_session:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    return 252.0 * per_session[timeframe]


def bar_duration(timeframe: str) -> timedelta:
    """Nominal duration of one bar. Used to place fills at the next bar's open."""
    table = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
    }
    if timeframe not in table:
        raise ValueError(f"unsupported timeframe {timeframe!r}; known: {sorted(table)}")
    return table[timeframe]
