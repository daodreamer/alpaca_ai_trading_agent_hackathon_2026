"""`TradingCalendar` backed by `pandas-market-calendars` — ADR 0006.

The library is pinned exactly (see `pyproject.toml`); its version is part of the
rebuildability inputs in `specs/12-storage.md`, so an upgrade is a
rebuild-triggering change rather than a routine bump.

This module is the only place in the backend that imports it. `pandas` and
friends are on the Domain's forbidden-import list, enforced by
`tests/test_domain_boundaries.py`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Final

import pandas_market_calendars as mcal

from alphagate.core.errors import CalendarHorizonExceeded, UnknownExchange
from alphagate.core.time_model import SessionKind, SessionWindow, ensure_utc

if TYPE_CHECKING:
    from alphagate.core.clock import Clock

__all__ = ["PandasMarketCalendar"]

_EXCHANGE_ALIASES: Final[dict[str, str]] = {
    "XNYS": "NYSE",
    "NYSE": "NYSE",
    "XNAS": "NASDAQ",
    "NASDAQ": "NASDAQ",
}
"""MIC codes the Domain uses, mapped to library calendar names.

Deliberately a closed list. Passing an unrecognised name straight through would
let a typo silently select a different exchange's schedule.
"""

_FIRST_SUPPORTED_YEAR: Final = 1970

# Scanning limit for "find the next session". Comfortably longer than any
# exchange closure; a hit past this is a bug, not a long weekend.
_MAX_SCAN_DAYS: Final = 30


class _Row:
    """One trading day's boundary instants, all UTC."""

    __slots__ = ("close", "day", "open", "post", "pre")

    def __init__(
        self, day: date, pre: datetime, open_: datetime, close: datetime, post: datetime
    ) -> None:
        self.day = day
        self.pre = pre
        self.open = open_
        self.close = close
        self.post = post


class PandasMarketCalendar:
    """Adapter satisfying `alphagate.core.time_model.TradingCalendar`."""

    def __init__(self, clock: Clock, forward_horizon_years: int = 2) -> None:
        self._clock = clock
        self._forward_horizon_years = forward_horizon_years
        self._years: dict[tuple[str, int], dict[date, _Row]] = {}

    # -- port ---------------------------------------------------------------

    def sessions_for(self, exchange: str, day: date) -> Sequence[SessionWindow]:
        name = self._resolve(exchange)
        self._check_horizon(day)
        row = self._year(name, day.year).get(day)
        if row is None:
            return ()
        return self._windows(row)

    def trading_day_of(self, exchange: str, instant: datetime) -> date:
        moment = ensure_utc(instant, field="instant")
        name = self._resolve(exchange)
        self._check_horizon(moment.date())

        # The first day whose extended session has not yet ended owns this
        # instant. That gives the intuitive answer inside a session, and rolls
        # an overnight, weekend or holiday instant forward to the session it
        # precedes (ADR 0004 D6).
        for row in self._rows_from(name, moment.date()):
            if moment < row.post:
                return row.day
        raise CalendarHorizonExceeded(
            f"{exchange}: no session found within {_MAX_SCAN_DAYS} days of {moment.isoformat()}"
        )

    def next_session_open(self, exchange: str, instant: datetime) -> datetime:
        moment = ensure_utc(instant, field="instant")
        name = self._resolve(exchange)
        self._check_horizon(moment.date())

        for row in self._rows_from(name, moment.date()):
            if row.open > moment:
                return row.open
        raise CalendarHorizonExceeded(
            f"{exchange}: no session open found within {_MAX_SCAN_DAYS} days of "
            f"{moment.isoformat()}"
        )

    # -- internals ----------------------------------------------------------

    def _resolve(self, exchange: str) -> str:
        try:
            return _EXCHANGE_ALIASES[exchange.upper()]
        except KeyError:
            supported = ", ".join(sorted(_EXCHANGE_ALIASES))
            raise UnknownExchange(
                f"no calendar configured for {exchange!r}; supported: {supported}"
            ) from None

    def _horizon_end_year(self) -> int:
        return self._clock.now().year + self._forward_horizon_years

    def _check_horizon(self, day: date) -> None:
        limit = self._horizon_end_year()
        if day.year > limit:
            raise CalendarHorizonExceeded(
                f"{day.isoformat()} is beyond the calendar's forward horizon "
                f"(through {limit}). Future sessions past that point are "
                "rule-extrapolated, not published (ADR 0006 D4)."
            )

    @staticmethod
    def _windows(row: _Row) -> tuple[SessionWindow, ...]:
        candidates = (
            (SessionKind.PRE, row.pre, row.open),
            (SessionKind.REGULAR, row.open, row.close),
            (SessionKind.POST, row.close, row.post),
        )
        return tuple(
            SessionWindow(kind=kind, open_utc=start, close_utc=end, session_date=row.day)
            for kind, start, end in candidates
            if end > start
        )

    def _year(self, name: str, year: int) -> dict[date, _Row]:
        cached = self._years.get((name, year))
        if cached is not None:
            return cached
        rows = self._load_year(name, year)
        self._years[(name, year)] = rows
        return rows

    def _load_year(self, name: str, year: int) -> dict[date, _Row]:
        if year < _FIRST_SUPPORTED_YEAR:
            return {}
        schedule = mcal.get_calendar(name).schedule(
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            market_times="all",
        )
        rows: dict[date, _Row] = {}
        for index, record in schedule.iterrows():
            day = index.date()
            rows[day] = _Row(
                day=day,
                pre=_to_utc(record["pre"]),
                open_=_to_utc(record["market_open"]),
                close=_to_utc(record["market_close"]),
                post=_to_utc(record["post"]),
            )
        return rows

    def _rows_from(self, name: str, start: date) -> Iterator[_Row]:
        """Trading days at or after `start`, in order, bounded by `_MAX_SCAN_DAYS`.

        Starts a day early: an instant in the small hours may still fall inside
        the previous day's post-market window.
        """
        current = start.toordinal() - 1
        for _ in range(_MAX_SCAN_DAYS + 1):
            day = date.fromordinal(current)
            if day.year <= self._horizon_end_year():
                row = self._year(name, day.year).get(day)
                if row is not None:
                    yield row
            current += 1


def _to_utc(value: object) -> datetime:
    """Convert a pandas Timestamp to a plain, UTC, timezone-aware datetime.

    Converted at the boundary so no pandas type escapes this module.
    """
    moment = value.to_pydatetime()  # type: ignore[attr-defined]
    return ensure_utc(moment, field="session boundary").astimezone(UTC)
