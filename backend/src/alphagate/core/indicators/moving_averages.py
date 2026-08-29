"""SMA and EMA — ADR 0007 D4/D5.

Contract for both:

* input — one price per bar, selected by `PriceSource` (default close);
* output — `float`, price-valued, in the approximate domain (ADR 0005 D1);
* warmup — `period` bars; nothing is emitted before the window is complete;
* missing data — bar-indexed, so a gap in the calendar is not a gap in the
  window: "the last n bars", never "the last n minutes";
* duplicates/out-of-order — refused by the base class (ADR 0007 D6);
* tolerance — rel 1e-9 against a reference implementation (ADR 0005 D7);
* causal — yes; no value depends on a later bar.
"""

from __future__ import annotations

from alphagate.core.bar import Bar
from alphagate.core.indicators.base import (
    IndicatorSpec,
    OnlineIndicator,
    PriceSource,
    RollingWindow,
    require_period,
)
from alphagate.core.indicators.smoothing import EmaCore

__all__ = ["Ema", "Sma"]


class Sma(OnlineIndicator[float]):
    """Simple moving average: the arithmetic mean of the last `period` prices."""

    def __init__(
        self,
        *,
        period: int,
        source: PriceSource = PriceSource.CLOSE,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.period = require_period(period)
        self.source = source
        self._window = RollingWindow(self.period)

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec("sma", (("period", str(self.period)), ("source", self.source.value)))

    @property
    def warmup(self) -> int:
        return self.period

    def _consume(self, bar: Bar) -> float | None:
        return self._window.push(self.source.of(bar))


class Ema(OnlineIndicator[float]):
    """Exponential moving average, `alpha = 2/(period+1)`, seeded with `SMA(period)`.

    Seeding with the SMA rather than with the first price is what makes the first
    emitted value informed by a full window (ADR 0007 D5) and what makes the
    output comparable with mainstream charting packages.
    """

    def __init__(
        self,
        *,
        period: int,
        source: PriceSource = PriceSource.CLOSE,
        allow_partial: bool = False,
    ) -> None:
        super().__init__(allow_partial=allow_partial)
        self.period = require_period(period)
        self.source = source
        self._core = EmaCore(self.period)

    @property
    def spec(self) -> IndicatorSpec:
        return IndicatorSpec("ema", (("period", str(self.period)), ("source", self.source.value)))

    @property
    def warmup(self) -> int:
        return self.period

    def _consume(self, bar: Bar) -> float | None:
        return self._core.feed(self.source.of(bar))
