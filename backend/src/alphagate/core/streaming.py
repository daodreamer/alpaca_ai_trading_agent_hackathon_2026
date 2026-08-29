"""Consuming a bar stream, safely — ADR 0007 D1/D6/D7, ADR 0008 D6.

Everything that walks a bar series one bar at a time — indicators, the structure
engine, and whatever Phase 5 brings — needs the same three guards, and needs
them to be the same guards, so that "this component saw a bar it should not
have" is impossible rather than merely unlikely:

* an instance is bound to one series key and rejects a bar from another;
* committed bars arrive in strictly increasing start-time order — a duplicate,
  a late bar or a revision means "rebuild from `BarSeries`", because recursive
  state cannot un-see a bar;
* a partial bar is either refused (default) or evaluated against a *copy* of the
  state, so a still-forming bar never becomes part of the recursion.

`ComponentSpec` is the identity these components are named by: a stable string
like `ema(period=20,source=CLOSE)` or `structure(left=3,right=3,...)` that a
cache row, a DTO field and a chart series can all agree on.
"""

from __future__ import annotations

import copy
import dataclasses
from abc import ABC, abstractmethod
from datetime import datetime

from alphagate.core.bar import AdjustmentMode, Bar, Feed
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.time_model import Timeframe

__all__ = ["BarConsumer", "ComponentSpec", "SeriesKey", "require_period"]


def require_period(period: int, *, field: str = "period") -> int:
    if period < 1:
        raise InvariantViolation(f"{field}: must be at least 1 bar, got {period}")
    return period


@dataclasses.dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A stream component's identity: what it is, and how it is configured.

    Parameters are ordered and pre-rendered so the key cannot change because a
    dict iterated differently.
    """

    kind: str
    parameters: tuple[tuple[str, str], ...] = ()

    @property
    def key(self) -> str:
        rendered = ",".join(f"{name}={value}" for name, value in self.parameters)
        return f"{self.kind}({rendered})"

    def __str__(self) -> str:
        return self.key


type SeriesKey = tuple[Ticker, Timeframe, Feed, AdjustmentMode]
"""What makes two bars part of the same series (specs/12-storage.md)."""


class BarConsumer[T](ABC):
    """Base for anything that folds a bar stream into state.

    Mutable by design — it accumulates a stream — and single-series: one
    instance follows one `(symbol, timeframe, feed, adjustment_mode)`.

    Subclasses implement `_consume`, which may assume the bar is finalized, in
    order, and from the right series.
    """

    def __init__(self, *, allow_partial: bool = False) -> None:
        self.allow_partial = allow_partial
        """Intrabar mode (specs/03-market-data.md). Off by default: a partial bar
        must not seed a recursive calculation unless the caller asked for it."""

        self._series: SeriesKey | None = None
        self._last_start: datetime | None = None
        self._value: T | None = None
        self._committed = 0

    # -- identity ----------------------------------------------------------

    @property
    @abstractmethod
    def spec(self) -> ComponentSpec: ...

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars required before the component can produce anything."""

    def __repr__(self) -> str:
        return f"<{self.spec.key} committed={self._committed}>"

    # -- reading -----------------------------------------------------------

    @property
    def value(self) -> T | None:
        """The result of the last committed bar. Provisional readings never land here."""
        return self._value

    @property
    def is_warm(self) -> bool:
        return self._value is not None

    @property
    def committed_bars(self) -> int:
        return self._committed

    @property
    def series(self) -> SeriesKey | None:
        """The series this instance is bound to, once it has seen a bar."""
        return self._series

    # -- writing -----------------------------------------------------------

    def update(self, bar: Bar) -> T | None:
        """Consume one bar and return its result, or `None` during warmup."""
        self._check_series(bar)
        self._check_order(bar)

        if not bar.is_final:
            return self._provisional(bar)

        value = self._consume(bar)
        self._last_start = bar.start_time_utc
        self._committed += 1
        self._value = value
        return value

    @abstractmethod
    def _consume(self, bar: Bar) -> T | None:
        """Advance the component's own state by one finalized bar."""

    def _provisional(self, bar: Bar) -> T | None:
        """Evaluate a still-forming bar without committing it (ADR 0007 D7).

        Runs against a deep copy, so the live state is bit-identical afterwards
        and the eventual closing result is exactly what it would have been had
        the intrabar ticks never arrived.
        """
        if not self.allow_partial:
            raise InvariantViolation(
                f"{self.spec.key}: refusing a partial bar at "
                f"{bar.start_time_utc.isoformat()}. Pass allow_partial=True for "
                "intrabar mode (specs/03-market-data.md)."
            )
        return copy.deepcopy(self).update(bar.finalized())

    def _check_series(self, bar: Bar) -> None:
        key: SeriesKey = (bar.symbol, bar.timeframe, bar.feed, bar.adjustment_mode)
        if self._series is None:
            self._series = key
        elif key != self._series:
            held = self._series
            raise InvariantViolation(
                f"{self.spec.key}: bar belongs to a different series — this component "
                f"follows {held[0]} {held[1].code} {held[2].value} {held[3].value}, "
                f"the bar is {key[0]} {key[1].code} {key[2].value} {key[3].value}"
            )

    def _check_order(self, bar: Bar) -> None:
        if self._last_start is not None and bar.start_time_utc <= self._last_start:
            raise InvariantViolation(
                f"{self.spec.key}: bar starts at {bar.start_time_utc.isoformat()}, at or "
                f"before the last committed bar at {self._last_start.isoformat()}; bar "
                "starts must be strictly increasing. A duplicate, late or revised bar "
                "means rebuilding from BarSeries, not replaying into component state "
                "(ADR 0007 D6)."
            )
