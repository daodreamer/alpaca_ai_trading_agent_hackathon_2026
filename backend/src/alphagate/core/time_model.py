"""Timeframes, sessions and the trading-calendar port — ADR 0004.

Every instant in the Domain is timezone-aware and normalized to UTC. Session
boundaries are never hardcoded here: they come from a `TradingCalendar`, so
holidays, half-days and DST are one implementation's problem instead of every
engine's.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Protocol, Self

from alphagate.core.errors import InvariantViolation

__all__ = [
    "SessionKind",
    "SessionWindow",
    "Timeframe",
    "TradingCalendar",
    "ensure_utc",
]


def ensure_utc(value: datetime, *, field: str) -> datetime:
    """Require an aware datetime and normalize it to UTC.

    A naive datetime is rejected rather than assumed local or assumed UTC:
    guessing is how a bar ends up an hour out twice a year.
    """
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise InvariantViolation(f"{field}: datetime must carry a timezone, got {value!r}")
    if value.tzinfo is UTC:
        return value
    return value.astimezone(UTC)


class Timeframe(Enum):
    """Canonical bar timeframes (specs/04-domain-model.md).

    Declaration order is ascending duration; `ladder`-style comparisons are not
    provided until something needs them.
    """

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1D"
    W1 = "1W"

    @property
    def code(self) -> str:
        return self.value

    @property
    def is_intraday(self) -> bool:
        """Whether the bar grid is anchored inside a single session (ADR 0004 D4)."""
        return self not in (Timeframe.D1, Timeframe.W1)

    @property
    def nominal_duration(self) -> timedelta:
        """The grid step for this timeframe.

        This is an *upper bound* on a bar's actual span, not the span itself: a
        session-close bar is truncated, and a `1D` bar covers a 6.5-hour regular
        session rather than 24 hours (ADR 0004 D5/D6).
        """
        return _NOMINAL_DURATIONS[self]

    @classmethod
    def from_code(cls, code: str) -> Self:
        """Parse a canonical code. Case-sensitive: `1m` is a minute, `1M` is nothing."""
        for timeframe in cls:
            if timeframe.value == code:
                return timeframe
        supported = ", ".join(tf.value for tf in cls)
        raise InvariantViolation(f"unsupported timeframe {code!r}; supported: {supported}")


_NOMINAL_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.M1: timedelta(minutes=1),
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.M30: timedelta(minutes=30),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
    Timeframe.W1: timedelta(days=7),
}


class SessionKind(Enum):
    """Which segment of the trading day a window or bar belongs to."""

    PRE = "PRE"
    REGULAR = "REGULAR"
    POST = "POST"
    OVERNIGHT = "OVERNIGHT"


@dataclasses.dataclass(frozen=True, slots=True)
class SessionWindow:
    """A half-open `[open_utc, close_utc)` slice of one trading day."""

    kind: SessionKind
    open_utc: datetime
    close_utc: datetime
    session_date: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "open_utc", ensure_utc(self.open_utc, field="open_utc"))
        object.__setattr__(self, "close_utc", ensure_utc(self.close_utc, field="close_utc"))
        if self.close_utc <= self.open_utc:
            raise InvariantViolation(
                f"session close ({self.close_utc.isoformat()}) must follow "
                f"open ({self.open_utc.isoformat()})"
            )

    @property
    def duration(self) -> timedelta:
        return self.close_utc - self.open_utc

    def contains(self, instant: datetime) -> bool:
        """Half-open membership: the close instant belongs to the next window."""
        moment = ensure_utc(instant, field="instant")
        return self.open_utc <= moment < self.close_utc


class TradingCalendar(Protocol):
    """Port for exchange session data (ADR 0004 D9).

    The Domain depends on this interface only. Choosing the concrete source —
    vendored schedule, maintained package, or provider endpoint — is deferred to
    its own ADR before Phase 2.
    """

    def sessions_for(self, exchange: str, day: date) -> Sequence[SessionWindow]:
        """Every session window on `day`, in chronological order.

        Empty when the exchange is closed that day.
        """
        ...

    def trading_day_of(self, exchange: str, instant: datetime) -> date:
        """The trading day an instant is attributed to.

        Not the UTC calendar date: an overnight or pre-market print belongs to
        the session it precedes.
        """
        ...

    def next_session_open(self, exchange: str, instant: datetime) -> datetime:
        """The first regular-session open strictly after `instant`."""
        ...
