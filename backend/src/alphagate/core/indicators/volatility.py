"""ATR — ADR 0007 D4/D5.

Contract:

* input — the full bar (high, low, and the *previous* close);
* output — `float`, price-valued, never negative;
* warmup — `period + 1` bars: the first bar has no previous close, so it has no
  true range and contributes nothing;
* missing data — bar-indexed; the "previous close" is the previous *bar's*
  close, whatever gap in time sits between them. That is the point: an
  overnight gap is exactly what true range is meant to capture;
* duplicates/out-of-order — refused by the base class (ADR 0007 D6);
* tolerance — rel 1e-9 internally, 1e-6 against a third-party implementation;
* causal — yes.
"""

from __future__ import annotations

from alphagate.core.bar import Bar
from alphagate.core.indicators.base import IndicatorSpec, OnlineIndicator, require_period
from alphagate.core.indicators.smoothing import WilderCore

__all__ = ["Atr", "true_range"]


def true_range(high: float, low: float, previous_close: float) -> float:
    """Wilder's true range: the bar's own range, extended to reach the last close."""
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


class Atr(OnlineIndicator[float]):
    """Average True Range, Wilder-smoothed, default period 14.

    The first bar's true range is *skipped* rather than approximated as
    `high - low`. Approximating it would make the seed depend on where the
    caller happened to start the series, which is exactly the kind of hidden
    input that breaks a deterministic rebuild (ADR 0007 D5).
    """

    def __init__(self, *, period: int = 14, allow_partial: bool = False) -> None:
        super().__init__(allow_partial=allow_partial)
        self.period = require_period(period)
        self._core = WilderCore(self.period)
        self._previous_close: float | None = None

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec("atr", (("period", str(self.period)),))

    @property
    def warmup(self) -> int:
        return self.period + 1

    def _consume(self, bar: Bar) -> float | None:
        previous_close = self._previous_close
        self._previous_close = float(bar.close)
        if previous_close is None:
            return None
        return self._core.feed(true_range(float(bar.high), float(bar.low), previous_close))
