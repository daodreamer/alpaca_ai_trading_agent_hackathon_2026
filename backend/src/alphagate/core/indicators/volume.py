"""Session VWAP and volume moving average — ADR 0007 D4/D5.

`SessionVwap` contract:

* input — high/low/close and volume, plus the bar's `session_date` and
  `session`, which are what make it calendar-aware without the Domain owning a
  calendar (ADR 0004 D6/D7);
* output — `float`, price-valued, `None` while the current session has traded
  no volume;
* warmup — one participating bar with non-zero volume;
* missing data — a bar outside the configured sessions is passed over and
  leaves the accumulation untouched; a zero-volume bar contributes nothing;
* reset — whenever `session_date` changes. There is no rolling variant here;
  anchored VWAP is a Full-version feature with its own anchor rules;
* duplicates/out-of-order — refused by the base class (ADR 0007 D6);
* causal — yes.

`VolumeSma` is an SMA over `bar.volume`. Raw volume needs no indicator: it is a
field on the bar.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

from alphagate.core.bar import Bar
from alphagate.core.errors import InvariantViolation
from alphagate.core.indicators.base import (
    IndicatorSpec,
    OnlineIndicator,
    PriceSource,
    RollingWindow,
    require_period,
)
from alphagate.core.time_model import SessionKind

__all__ = ["SessionVwap", "VolumeSma"]

_SESSION_ORDER = tuple(SessionKind)


class SessionVwap(OnlineIndicator[float]):
    """Volume-weighted average price, anchored at the start of the trading day.

    The typical price is `(high + low + close) / 3`. A provider-supplied
    `Bar.vwap` is deliberately ignored: it is that provider's aggregate over a
    different trade population, and mixing it in would make the value depend on
    which feed served the bars (ADR 0007 D5).

    The session's terms are retained and re-summed with `math.fsum` on every bar
    rather than kept in two running accumulators (ADR 0007 D2). A running sum
    loses the low-order bits of a small term added to a large total, and never
    gets them back, so the value would depend on the *order and magnitudes* of
    the day's bars and not only on their contents. `fsum` is correctly rounded:
    the result is the nearest float to the exact sum, whatever the spread of
    magnitudes. A session holds a few hundred terms, so nothing about this is
    expensive.
    """

    def __init__(
        self,
        *,
        sessions: Sequence[SessionKind] = (SessionKind.REGULAR,),
        source: PriceSource = PriceSource.HLC3,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        if not sessions:
            raise InvariantViolation("sessions: at least one session kind must participate")
        self.sessions = tuple(kind for kind in _SESSION_ORDER if kind in sessions)
        self.source = source
        self._session_date: date | None = None
        self._notional: list[float] = []
        self._volumes: list[float] = []

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec(
            "session_vwap", (("sessions", "|".join(kind.value for kind in self.sessions)),)
        )

    @property
    def warmup(self) -> int:
        """One participating bar — provided it traded."""
        return 1

    def _consume(self, bar: Bar) -> float | None:
        if bar.session not in self.sessions:
            return self._value  # unchanged; a pre-market print is not this session

        if bar.session_date != self._session_date:
            self._session_date = bar.session_date
            self._notional.clear()
            self._volumes.clear()

        volume = float(bar.volume)
        if volume > 0.0:
            self._notional.append(self.source.of(bar) * volume)
            self._volumes.append(volume)

        if not self._volumes:
            return None  # a volume-weighted average of no volume is undefined
        return math.fsum(self._notional) / math.fsum(self._volumes)


class VolumeSma(OnlineIndicator[float]):
    """Simple moving average of bar volume."""

    def __init__(self, *, period: int, allow_partial: bool = False) -> None:
        super().__init__(allow_partial=allow_partial)
        self.period = require_period(period)
        self._window = RollingWindow(self.period)

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec("volume_sma", (("period", str(self.period)),))

    @property
    def warmup(self) -> int:
        return self.period

    def _consume(self, bar: Bar) -> float | None:
        return self._window.push(float(bar.volume))
