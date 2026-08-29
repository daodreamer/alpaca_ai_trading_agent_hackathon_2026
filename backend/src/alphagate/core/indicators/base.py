"""The indicator engine contract — ADR 0007.

An indicator here is an **online state machine**: it consumes one bar at a time
and returns the value for that bar, or `None` while it is still warming up.
Whole-series computation (`compute_series`) is a fold of that same `update`,
never a second implementation — which is what makes a replayed value and a live
value the same value, and what keeps a future bar structurally out of reach
(CLAUDE.md §12).

The stream discipline itself — series binding, ordering, partial bars — lives in
`alphagate.core.streaming.BarConsumer`, because the structure engine needs exactly
the same guards and two copies of a rule is one copy too many.

Values are `float`: indicators live in the approximate numeric domain, including
the price-valued ones, and cross back to `Decimal` only at a persistence or API
boundary (ADR 0005 D1).
"""

from __future__ import annotations

import dataclasses
import math
from abc import abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.identifiers import Ticker
from alphagate.core.streaming import BarConsumer, ComponentSpec, require_period
from alphagate.core.time_model import Timeframe

__all__ = [
    "IndicatorPoint",
    "IndicatorSeries",
    "IndicatorSpec",
    "OnlineIndicator",
    "PriceSource",
    "RollingWindow",
    "compute_series",
    "require_period",
]

IndicatorSpec = ComponentSpec
"""An indicator's identity (ADR 0007 D8).

The same class every stream component is named by, under the name the indicator
specs use. A plain alias rather than a subclass: an indicator's identity is not
a different *kind* of thing from a structure engine's.
"""


class PriceSource(Enum):
    """Which price of a bar an indicator reads.

    Named combinations only. An indicator that needs something else needs its
    own field, not a lambda: the source is part of the indicator's identity
    (ADR 0007 D8) and has to be nameable in a cache key.
    """

    OPEN = "OPEN"
    HIGH = "HIGH"
    LOW = "LOW"
    CLOSE = "CLOSE"
    HL2 = "HL2"
    HLC3 = "HLC3"
    OHLC4 = "OHLC4"

    def of(self, bar: Bar) -> float:
        """Extract this source from `bar`, crossing into the approximate domain."""
        match self:
            case PriceSource.OPEN:
                return float(bar.open)
            case PriceSource.HIGH:
                return float(bar.high)
            case PriceSource.LOW:
                return float(bar.low)
            case PriceSource.CLOSE:
                return float(bar.close)
            case PriceSource.HL2:
                return (float(bar.high) + float(bar.low)) / 2
            case PriceSource.HLC3:
                return (float(bar.high) + float(bar.low) + float(bar.close)) / 3
            case PriceSource.OHLC4:
                return (float(bar.open) + float(bar.high) + float(bar.low) + float(bar.close)) / 4


@dataclasses.dataclass(frozen=True, slots=True)
class IndicatorPoint[T]:
    """One output sample, timestamped by the bar that produced it.

    `value` is `None` while the indicator is warming up. There is no separate
    "missing" marker: an indicator that has nothing to say says nothing.
    """

    start_time_utc: datetime
    value: T | None


@dataclasses.dataclass(frozen=True, slots=True)
class IndicatorSeries[T]:
    """A computed indicator over one bar series (specs/04-domain-model.md).

    One point per input bar, in input order (ADR 0007 D3). The series is
    bar-indexed, not time-indexed: a bar that never printed is simply not here,
    and no value was interpolated to stand in for it.
    """

    symbol: Ticker
    timeframe: Timeframe
    spec: ComponentSpec
    warmup: int
    points: tuple[IndicatorPoint[T], ...]
    source_data_version: str | None = None
    """Fingerprint of the bars this was computed from, when the caller has one.

    Supplied rather than derived: the Domain does not know what a data version
    means. It becomes load-bearing when indicator caches land in Phase 7.
    """

    def __len__(self) -> int:
        return len(self.points)

    @property
    def values(self) -> tuple[T | None, ...]:
        return tuple(point.value for point in self.points)

    @property
    def is_warm(self) -> bool:
        """Whether the series produced any value at all."""
        return any(point.value is not None for point in self.points)


class OnlineIndicator[T](BarConsumer[T]):
    """Base class for every indicator. Subclasses implement `_consume`.

    Adds nothing to `BarConsumer` mechanically, and two things semantically: the
    result is a number in the approximate domain, and `warmup` is a bar count
    after which a value is emitted for every bar (ADR 0007 D4). A component that
    is not a number-per-bar — the structure engine, for one — is not an
    indicator and must not inherit this (CLAUDE.md §4).
    """

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars required before the first value is emitted (ADR 0007 D4)."""


class RollingWindow:
    """The last `size` values, and their mean.

    Sums with `math.fsum` over the retained values rather than keeping a running
    accumulator: `fsum` is correctly rounded, while a running total silently
    drops any term below half an ulp of the total so far and never recovers it
    (ADR 0007 D2).
    """

    __slots__ = ("_values", "size")

    def __init__(self, size: int) -> None:
        self.size = require_period(size, field="window size")
        self._values: deque[float] = deque(maxlen=size)

    @property
    def is_full(self) -> bool:
        return len(self._values) == self.size

    def push(self, value: float) -> float | None:
        """Add a value; return the window mean once the window is full."""
        self._values.append(value)
        return self.mean()

    def mean(self) -> float | None:
        if not self.is_full:
            return None
        return math.fsum(self._values) / self.size


def compute_series[T](indicator: OnlineIndicator[T], bars: Sequence[Bar]) -> IndicatorSeries[T]:
    """Fold `indicator` over `bars` and collect the result (ADR 0007 D1/D3).

    Takes an unused indicator instance and consumes it: the instance carries the
    parameters, and reusing one would silently continue an earlier series.
    """
    if not bars:
        raise InvariantViolation(
            f"{indicator.spec.key}: an indicator series needs at least one bar; "
            "an empty result could not be attributed to a symbol or timeframe"
        )
    if indicator.committed_bars:
        raise InvariantViolation(
            f"{indicator.spec.key}: compute_series needs a fresh indicator, but this "
            f"one has already consumed {indicator.committed_bars} bars"
        )

    points: Iterable[IndicatorPoint[T]] = [
        IndicatorPoint(start_time_utc=bar.start_time_utc, value=indicator.update(bar))
        for bar in bars
    ]
    first = bars[0]
    return IndicatorSeries(
        symbol=first.symbol,
        timeframe=first.timeframe,
        spec=indicator.spec,
        warmup=indicator.warmup,
        points=tuple(points),
    )
